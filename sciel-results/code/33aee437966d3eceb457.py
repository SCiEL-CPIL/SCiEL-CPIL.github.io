import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    # Parameters for the task: v=24, k=7, t=3
    # But we should use the passed arguments.
    
    # Precompute triples
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute 3D index lookup for speed
    idx_3d = np.full((v, v, v), -1, dtype=np.int32)
    for a in range(v):
        for b in range(v):
            for c in range(v):
                if a == b or a == c or b == c:
                    continue
                t1, t2, t3 = sorted((a, b, c))
                idx_3d[a, b, c] = triple_to_idx[(t1, t2, t3)]
                
    def get_triple_idx(a, b, c):
        return int(idx_3d[a, b, c])

    def get_block_triple_indices(block):
        """Returns list of triple indices covered by a block"""
        return [triple_to_idx[triple] for triple in itertools.combinations(block, t)]

    def greedy_cover(random_state):
        rng = random.Random(random_state)
        uncovered = np.ones(num_triples, dtype=np.bool_)
        num_uncovered = num_triples
        blocks = []
        
        while num_uncovered > 0:
            # Find an uncovered triple
            # Random start scan
            start_idx = rng.randint(0, num_triples - 1)
            target_idx = -1
            for offset in range(num_triples):
                idx = (start_idx + offset) % num_triples
                if uncovered[idx]:
                    target_idx = idx
                    break
            if target_idx == -1:
                break
            
            target_triple = triples[target_idx]
            block = list(target_triple)
            
            # Mark target triple covered
            uncovered[target_idx] = False
            num_uncovered -= 1
            
            # Add remaining k - t points
            while len(block) < k:
                best_gain = -1
                best_point = -1
                block_set = set(block)
                
                # Calculate gain for each candidate x
                gains = {}
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            u = block[i]
                            w = block[j]
                            idx = get_triple_idx(x, u, w)
                            if uncovered[idx]:
                                gain += 1
                    gains[x] = gain
                
                max_gain = max(gains.values()) if gains else 0
                best_candidates = [x for x, g in gains.items() if g == max_gain]
                
                if not best_candidates:
                    break
                    
                best_point = rng.choice(best_candidates)
                block.append(best_point)
                
                # Update uncovered
                old_block = block[:-1]
                for i in range(len(old_block)):
                    for j in range(i+1, len(old_block)):
                        u = old_block[i]
                        w = old_block[j]
                        idx = get_triple_idx(best_point, u, w)
                        if uncovered[idx]:
                            uncovered[idx] = False
                            num_uncovered -= 1
                            
            blocks.append(tuple(sorted(block)))
            
        return blocks

    def trim_blocks(blocks):
        """Remove blocks that are not essential (cover no unique triple)"""
        if not blocks:
            return blocks
            
        triple_counts = [0] * num_triples
        block_triples = []
        
        for block in blocks:
            bt = get_block_triple_indices(block)
            block_triples.append(bt)
            for idx in bt:
                triple_counts[idx] += 1
                
        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not active[i]:
                    continue
                essential = False
                for idx in block_triples[i]:
                    if triple_counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    changed = True
                    for idx in block_triples[i]:
                        triple_counts[idx] -= 1
        return [blocks[i] for i in range(len(blocks)) if active[i]]

    def local_search_sa(blocks, time_limit):
        """
        Simulated Annealing to minimize number of blocks.
        Moves: Remove block (if valid), Swap point in block (if valid).
        """
        blocks = [list(b) for b in blocks]
        n_blocks = len(blocks)
        
        # Precompute triple indices for each block
        block_triples = [get_block_triple_indices(b) for b in blocks]
        
        # Initialize counts
        triple_counts = np.zeros(num_triples, dtype=np.int32)
        for bt in block_triples:
            for idx in bt:
                triple_counts[idx] += 1
        
        # Ensure initial state is valid (should be)
        if np.any(triple_counts == 0):
             # Fallback: just return trimmed blocks
             return trim_blocks(blocks)

        best_blocks = blocks
        best_len = n_blocks
        best_counts = triple_counts.copy()
        best_bts = block_triples
        
        rng = random.Random(int(time.time() * 1000))
        start_time = time.time()
        deadline = start_time + time_limit
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < deadline:
            if T < T_min:
                # Restart temperature occasionally? Or just stop?
                # Let's just stop SA and return best
                break
                
            move_type = rng.random()
            new_blocks = blocks
            new_bts = block_triples
            new_counts = triple_counts
            delta_cost = 0
            valid = True
            
            if move_type < 0.5 and n_blocks > 0:
                # Try Remove
                i = rng.randint(0, n_blocks - 1)
                # Check if removal is valid
                # A removal is valid if no triple count drops to 0
                # i.e., min(triple_counts - block_triples[i]) >= 1
                # We can check this efficiently
                # But let's just simulate
                for idx in block_triples[i]:
                    if triple_counts[idx] == 1:
                        valid = False
                        break
                
                if valid:
                    # Apply removal
                    for idx in block_triples[i]:
                        triple_counts[idx] -= 1
                    blocks.pop(i)
                    block_triples.pop(i)
                    n_blocks -= 1
                    delta_cost = -1
            else:
                # Try Swap
                if n_blocks == 0:
                    continue
                i = rng.randint(0, n_blocks - 1)
                b = blocks[i]
                if len(b) < 2:
                    continue
                x = b[rng.randint(0, len(b)-1)]
                # Choose y not in b
                y = rng.randint(0, v-1)
                while y in b:
                    y = rng.randint(0, v-1)
                
                # New block
                new_b = b.copy()
                new_b.remove(x)
                new_b.append(y)
                new_b.sort()
                
                # Check if new block is already in list?
                # If it is, swapping effectively removes one block and adds another identical one?
                # No, we are modifying block i. If new_b is same as another block j, we have duplicates.
                # Duplicates are allowed in intermediate states?
                # The problem says "blocks must be distinct".
                # So we must ensure new_b is not in blocks (except at index i).
                # And new_b must not be equal to any other block.
                # Also, the set of points in new_b must be distinct (guaranteed by construction).
                
                # Check distinctness
                is_distinct = True
                for j in range(n_blocks):
                    if j == i:
                        continue
                    if sorted(blocks[j]) == new_b:
                        is_distinct = False
                        break
                if not is_distinct:
                    continue
                
                # Calculate new triple indices for new_b
                new_bt = get_block_triple_indices(new_b)
                
                # Check validity
                # Old block i is removed, new block is added.
                # Net change: - old_bt[i] + new_bt
                # We need to ensure no count drops to 0.
                # Counts affected:
                # 1. Triples in old_bt[i] but NOT in new_bt: count decreases by 1.
                # 2. Triples in new_bt but NOT in old_bt[i]: count increases by 1.
                # 3. Triples in both: count unchanged.
                
                old_set = set(block_triples[i])
                new_set = set(new_bt)
                
                # Check decreases
                for idx in old_set:
                    if idx not in new_set:
                        if triple_counts[idx] == 1:
                            valid = False
                            break
                if not valid:
                    continue
                
                # If valid, apply swap
                # Update counts
                for idx in old_set:
                    if idx not in new_set:
                        triple_counts[idx] -= 1
                for idx in new_set:
                    if idx not in old_set:
                        triple_counts[idx] += 1
                        
                blocks[i] = new_b
                block_triples[i] = new_bt
                # n_blocks unchanged, delta_cost = 0
                delta_cost = 0
            
            if valid:
                if delta_cost < 0:
                    # Always accept improvement
                    if n_blocks < best_len:
                        best_len = n_blocks
                        best_blocks = [b[:] for b in blocks]
                        best_counts = triple_counts.copy()
                        best_bts = [bt[:] for bt in block_triples]
                else:
                    # Accept worse or equal with probability
                    # For delta_cost = 0, we accept to explore.
                    # We should accept 0-cost moves to move around the plateau.
                    # But we must avoid cycles.
                    # Simple acceptance: if rng.random() < exp(-delta_cost / T)
                    # Since delta_cost is 0, exp(0)=1, so always accept.
                    # This means we random walk on the plateau.
                    if rng.random() < 1.0: # Always accept 0-cost
                         pass # State already updated
                T *= cooling
            else:
                # Revert state if it was modified?
                # In our logic above, we only modify state if valid.
                # If invalid, we didn't modify.
                # So no revert needed.
                pass
        
        return best_blocks

    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    seed_counter = 0
    while time.time() < deadline:
        # Generate initial covering
        blocks = greedy_cover(seed_counter)
        blocks = trim_blocks(blocks)
        
        # Local search
        remaining = deadline - time.time()
        if remaining > 5:
            ls_time = min(10, remaining - 2)
            blocks = local_search_sa(blocks, ls_time)
            blocks = trim_blocks(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        if best_len == 69:
            break
            
        seed_counter += 1
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
