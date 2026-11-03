import itertools
import time
import random
import math

def construct(v, k, t):
    # Parameters for the task: v=24, k=7, t=3
    # But we should use the passed arguments.
    
    # Precompute triples
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute 3D index lookup for speed: idx_3d[a][b][c] = index of triple (a,b,c)
    idx_3d = [[[-1]*v for _ in range(v)] for _ in range(v)]
    for a in range(v):
        for b in range(v):
            for c in range(v):
                if a == b or a == c or b == c:
                    continue
                t1, t2, t3 = sorted((a, b, c))
                idx_3d[a][b][c] = triple_to_idx[(t1, t2, t3)]
        
    def get_triple_idx(a, b, c):
        return idx_3d[a][b][c]

    def greedy_cover(random_state):
        rng = random.Random(random_state)
        uncovered = bytearray([1] * num_triples)
        num_uncovered = num_triples
        blocks = []
        
        points = list(range(v))
        
        while num_uncovered > 0:
            # Find first uncovered
            start_idx = 0
            while start_idx < num_triples and uncovered[start_idx] == 0:
                start_idx += 1
            if start_idx >= num_triples:
                break
                
            target_triple = triples[start_idx]
            block = list(target_triple)
            uncovered[start_idx] = 0
            num_uncovered -= 1
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in points if x not in block]
                rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            p1 = block[i]
                            p2 = block[j]
                            idx = get_triple_idx(x, p1, p2)
                            if uncovered[idx]:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    
                if best_point == -1:
                    break
                block.append(best_point)
                
                # Update uncovered
                for i in range(len(block)):
                    if block[i] == best_point:
                        continue
                    for j in range(i+1, len(block)):
                        if block[j] == best_point:
                            continue
                        p1 = block[i]
                        p2 = block[j]
                        idx = get_triple_idx(best_point, p1, p2)
                        if uncovered[idx]:
                            uncovered[idx] = 0
                            num_uncovered -= 1
                            
            blocks.append(tuple(sorted(block)))
            
        return blocks

    def local_search(initial_blocks, rng, max_iters=2000, time_limit=5.0):
        """
        Simulated Annealing local search.
        State: list of blocks (tuples of points)
        Fitness: len(blocks) + 10 * num_uncovered
        Moves:
        1. Remove redundant block
        2. Add random block (to fix coverage)
        3. Swap block i with random block j
        """
        blocks = [list(b) for b in initial_blocks]
        
        # Precompute triple lists for current blocks
        block_triples = []
        triple_counts = [0] * num_triples
        
        def rebuild_counts():
            nonlocal triple_counts
            triple_counts = [0] * num_triples
            block_triples = []
            for b in blocks:
                bt = []
                for triple in itertools.combinations(b, t):
                    idx = triple_to_idx[triple]
                    triple_counts[idx] += 1
                    bt.append(idx)
                block_triples.append(bt)
            return triple_counts
        
        rebuild_counts()
        
        def get_uncovered_count():
            return sum(1 for c in triple_counts if c == 0)
        
        current_uncovered = get_uncovered_count()
        current_fitness = len(blocks) + 10 * current_uncovered
        
        best_blocks = [list(b) for b in blocks]
        best_fitness = current_fitness
        
        T = 10.0
        T_min = 0.1
        alpha = 0.999
        
        start_time = time.time()
        i = 0
        while i < max_iters and time.time() - start_time < time_limit:
            i += 1
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to remove a redundant block
                # Find a block where all triples have count > 1
                # To be fast, pick a random block and check
                if len(blocks) == 0: continue
                idx = rng.randrange(len(blocks))
                # Check if block idx is redundant
                redundant = True
                for tri_idx in block_triples[idx]:
                    if triple_counts[tri_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    removed_block = blocks.pop(idx)
                    removed_triples = block_triples.pop(idx)
                    for tri_idx in removed_triples:
                        triple_counts[tri_idx] -= 1
                    
                    new_uncovered = get_uncovered_count()
                    new_fitness = len(blocks) + 10 * new_uncovered
                    
                    if new_fitness <= current_fitness or rng.random() < math.exp(-(new_fitness - current_fitness) / T):
                        current_fitness = new_fitness
                        current_uncovered = new_uncovered
                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
                            best_blocks = [list(b) for b in blocks]
                    else:
                        # Revert
                        blocks.insert(idx, removed_block)
                        block_triples.insert(idx, removed_triples)
                        for tri_idx in removed_triples:
                            triple_counts[tri_idx] += 1
                
            elif move_type < 0.7:
                # Try to add a block that covers an uncovered triple
                # Find an uncovered triple
                uncovered_indices = [i for i, c in enumerate(triple_counts) if c == 0]
                if not uncovered_indices:
                    continue
                target_triple_idx = rng.choice(uncovered_indices)
                target_triple = triples[target_triple_idx]
                
                # Generate a random block containing this triple
                remaining_points = [x for x in range(v) if x not in target_triple]
                rng.shuffle(remaining_points)
                new_block = list(target_triple) + remaining_points[:k-t]
                new_block.sort()
                
                # Check if this block is already in the list? 
                # For simplicity, we just add it. If duplicate, it's wasteful but SA might remove it later.
                # To be smarter, we could check if it's identical to an existing block.
                # Given v=24, k=7, duplicates are possible but not too frequent.
                
                # Calculate gain: how many new uncovered triples does it cover?
                # And how many existing covered ones?
                new_triples = []
                gain_uncovered = 0
                for triple in itertools.combinations(new_block, t):
                    idx = triple_to_idx[triple]
                    new_triples.append(idx)
                    if triple_counts[idx] == 0:
                        gain_uncovered += 1
                
                # Only add if it helps? Or always try?
                # If gain_uncovered is 0, it's likely bad unless it allows a swap later.
                # Let's only add if it covers at least 1 uncovered triple.
                if gain_uncovered == 0:
                    continue
                
                # Apply move
                blocks.append(new_block)
                block_triples.append(new_triples)
                for tri_idx in new_triples:
                    triple_counts[tri_idx] += 1
                
                new_uncovered = get_uncovered_count()
                new_fitness = len(blocks) + 10 * new_uncovered
                
                if new_fitness <= current_fitness or rng.random() < math.exp(-(new_fitness - current_fitness) / T):
                    current_fitness = new_fitness
                    current_uncovered = new_uncovered
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_blocks = [list(b) for b in blocks]
                else:
                    # Revert
                    blocks.pop()
                    block_triples.pop()
                    for tri_idx in new_triples:
                        triple_counts[tri_idx] -= 1
            
            else:
                # Swap: Replace block i with a new random block
                if len(blocks) == 0: continue
                idx = rng.randrange(len(blocks))
                
                # Generate new block
                # Ideally, the new block should cover some uncovered triples.
                uncovered_indices = [i for i, c in enumerate(triple_counts) if c == 0]
                if uncovered_indices:
                    target_triple_idx = rng.choice(uncovered_indices)
                    target_triple = triples[target_triple_idx]
                    remaining_points = [x for x in range(v) if x not in target_triple]
                    rng.shuffle(remaining_points)
                    new_block = list(target_triple) + remaining_points[:k-t]
                else:
                    # Random block
                    new_block = rng.sample(range(v), k)
                
                new_block.sort()
                
                # If new block is same as old, skip
                if tuple(new_block) == tuple(blocks[idx]):
                    continue
                
                # Apply swap
                old_block = blocks[idx]
                old_triples = block_triples[idx]
                
                for tri_idx in old_triples:
                    triple_counts[tri_idx] -= 1
                
                new_triples = []
                for triple in itertools.combinations(new_block, t):
                    idx_t = triple_to_idx[triple]
                    new_triples.append(idx_t)
                    triple_counts[idx_t] += 1
                
                blocks[idx] = new_block
                block_triples[idx] = new_triples
                
                new_uncovered = get_uncovered_count()
                new_fitness = len(blocks) + 10 * new_uncovered
                
                if new_fitness <= current_fitness or rng.random() < math.exp(-(new_fitness - current_fitness) / T):
                    current_fitness = new_fitness
                    current_uncovered = new_uncovered
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_blocks = [list(b) for b in blocks]
                else:
                    # Revert
                    for tri_idx in new_triples:
                        triple_counts[tri_idx] -= 1
                    for tri_idx in old_triples:
                        triple_counts[tri_idx] += 1
                    blocks[idx] = old_block
                    block_triples[idx] = old_triples
            
            T *= alpha
            
        # Final trim: remove any redundant blocks
        # We do a simple pass to remove redundant blocks from best_blocks
        final_blocks = [list(b) for b in best_blocks]
        if final_blocks:
             # Rebuild counts for final_blocks
             fc_counts = [0] * num_triples
             fc_triples = []
             for b in final_blocks:
                 bt = []
                 for triple in itertools.combinations(b, t):
                     idx = triple_to_idx[triple]
                     fc_counts[idx] += 1
                     bt.append(idx)
                 fc_triples.append(bt)
             
             active = [True] * len(final_blocks)
             changed = True
             while changed:
                 changed = False
                 for i in range(len(final_blocks)):
                     if not active[i]: continue
                     redundant = True
                     for tri_idx in fc_triples[i]:
                         if fc_counts[tri_idx] == 1:
                             redundant = False
                             break
                     if redundant:
                         active[i] = False
                         changed = True
                         for tri_idx in fc_triples[i]:
                             fc_counts[tri_idx] -= 1
             
             final_blocks = [final_blocks[i] for i in range(len(final_blocks)) if active[i]]
             
        return final_blocks

    def trim_blocks(blocks):
        # Simple trim for fallback
        counts = [0] * num_triples
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active_blocks = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i, block in enumerate(blocks):
                if not active_blocks[i]:
                    continue
                essential = False
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active_blocks[i] = False
                    changed = True
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active_blocks[i]]

    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 140 # 10s buffer
    
    rng = random.Random(42)
    
    seed_counter = 0
    while time.time() < deadline:
        # 1. Greedy construction
        greedy_blocks = greedy_cover(seed_counter)
        
        # 2. Local Search (Simulated Annealing)
        # We allocate more time to local search if we have a good starting point?
        # For now, fixed time limit per SA run.
        ls_blocks = local_search(greedy_blocks, rng, max_iters=5000, time_limit=8.0)
        
        # 3. Update best
        if len(ls_blocks) < best_len:
            best_len = len(ls_blocks)
            best_blocks = ls_blocks
            
        if best_len == 69:
            break
            
        seed_counter += 1
        
    # Convert tuples to lists for output
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
