import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    # Parameters for the task: v=24, k=7, t=3
    # But we should use the passed arguments.
    
    # Precompute triples
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute 3D index lookup for speed
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
        uncovered = np.ones(num_triples, dtype=np.uint8)
        num_uncovered = num_triples
        blocks = []
        
        # List of points
        points = list(range(v))
        
        # Precompute: for each point x, list of triple indices where x appears
        # triple_indices_by_point[x] = list of (idx, other1, other2) for triples (x, other1, other2)
        triple_indices_by_point = [[] for _ in range(v)]
        for idx, triple in enumerate(triples):
            for p in triple:
                others = [q for q in triple if q != p]
                triple_indices_by_point[p].append((idx, others[0], others[1]))
        
        while num_uncovered > 0:
            # Find first uncovered triple
            start_idx = np.argmax(uncovered)
            if uncovered[start_idx] == 0:
                break
                
            # Get the triple
            target_triple = triples[start_idx]
            block = list(target_triple)
            
            # Mark as covered
            uncovered[start_idx] = 0
            num_uncovered -= 1
            
            # Add remaining k - t points
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in points if x not in block]
                
                # Shuffle candidates for random tie-breaking
                rng.shuffle(candidates)
                
                # Build list of (block_i, block_j) pairs for gain computation
                block_pairs = []
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        block_pairs.append((block[i], block[j]))
                
                for x in candidates:
                    gain = 0
                    # Count triples (x, u, v) where u, v in block
                    for (p1, p2) in block_pairs:
                        idx = get_triple_idx(x, p1, p2)
                        if uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    
                if best_point == -1:
                    break
                
                block.append(best_point)
                
                # Update uncovered status for all triples covered by the new point
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

    def trim_blocks(blocks):
        # Count frequency of each triple
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

    def local_search_trim(blocks, random_state, time_limit=5.0):
        """
        Attempt to further reduce blocks by random reordering and deletion.
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        start_time = time.time()
        
        # Precompute triple counts for current blocks
        counts = [0] * num_triples
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active = [True] * len(blocks)
        
        while time.time() - start_time < time_limit:
            active_indices = [i for i in range(len(blocks)) if active[i]]
            rng.shuffle(active_indices)
            
            found_removal = False
            for i in active_indices:
                if not active[i]:
                    continue
                essential = False
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    found_removal = True
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
            if not found_removal:
                break
                
        return [blocks[i] for i in range(len(blocks)) if active[i]]

    def swap_local_search(blocks, random_state, time_limit=3.0):
        """
        Try to replace a block with another block that covers the same essential
        triples while potentially allowing removal of other blocks.
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        n_blocks = len(blocks)
        start_time = time.time()
        
        # Precompute triple counts
        counts = [0] * num_triples
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
        
        active = [True] * n_blocks
        
        # Precompute all possible blocks (7-subsets of v)
        all_blocks = list(itertools.combinations(range(v), k))
        
        # For each point, precompute which triples it's in with each pair
        # Actually, let's just iterate through candidates
        
        while time.time() - start_time < time_limit:
            # Try to find a removable block
            active_indices = [i for i in range(n_blocks) if active[i]]
            if not active_indices:
                break
            
            rng.shuffle(active_indices)
            made_change = False
            
            for i in active_indices:
                if not active[i]:
                    continue
                
                # Check if block i is removable
                essential_triples = []
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential_triples.append(idx)
                
                if not essential_triples:
                    # Block is removable
                    active[i] = False
                    made_change = True
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                    continue
                
                # Block has essential triples; try to find a replacement block
                # that covers all essential triples and has minimal overlap with others
                best_replacement = None
                best_score = float('inf')
                
                # Sample some candidate blocks
                num_candidates = min(50, len(all_blocks))
                candidates = rng.sample(all_blocks, num_candidates)
                
                for cand in candidates:
                    cand_set = set(cand)
                    # Check if cand covers all essential triples
                    covers_essential = True
                    for idx in essential_triples:
                        triple = triples[idx]
                        if not all(x in cand_set for x in triple):
                            covers_essential = False
                            break
                    
                    if not covers_essential:
                        continue
                    
                    # Score: number of new triples covered (lower is better for replacement)
                    # Actually, we want a block that doesn't add too much redundancy
                    new_triples = 0
                    for triple in itertools.combinations(cand, t):
                        idx = triple_to_idx[triple]
                        if counts[idx] == 0:
                            new_triples += 1
                    
                    # Prefer blocks with fewer new triples (less redundant)
                    if new_triples < best_score:
                        best_score = new_triples
                        best_replacement = cand
                
                if best_replacement is not None:
                    # Replace block i with best_replacement
                    old_block = blocks[i]
                    blocks[i] = best_replacement
                    
                    # Update counts
                    for triple in itertools.combinations(old_block, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                    for triple in itertools.combinations(best_replacement, t):
                        idx = triple_to_idx[triple]
                        counts[idx] += 1
                    
                    made_change = True
                    # After replacement, try to remove more blocks
                    break
            
            if not made_change:
                break
        
        return [blocks[i] for i in range(n_blocks) if active[i]]

    def final_polish(blocks, random_state, time_limit=3.0):
        """
        High-precision polish: repeatedly scan all blocks in random order
        and remove any that are not essential (have no uniquely covered triple).
        This is an exhaustive check unlike the timed local_search_trim.
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        start_time = time.time()
        
        # Precompute triple counts
        counts = [0] * num_triples
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
        
        active = [True] * len(blocks)
        
        # Multiple passes until no more removals or time runs out
        while time.time() - start_time < time_limit:
            active_indices = [i for i in range(len(blocks)) if active[i]]
            if not active_indices:
                break
            rng.shuffle(active_indices)
            
            found_removal = False
            for i in active_indices:
                if not active[i]:
                    continue
                essential = False
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    found_removal = True
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
            
            if not found_removal:
                break
        
        return [blocks[i] for i in range(len(blocks)) if active[i]]
    
    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 100  # 20s buffer for safety
    
    seed_counter = 0
    while time.time() < deadline:
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        
        # Scale time limits based on remaining time
        if remaining > 10:
            trim_t = 1.5
            swap_t = 2.5
            polish_t = 2.0
        elif remaining > 5:
            trim_t = 0.8
            swap_t = 1.2
            polish_t = 1.0
        else:
            trim_t = 0.3
            swap_t = 0.5
            polish_t = 0.5
        
        blocks = greedy_cover(seed_counter)
        blocks = trim_blocks(blocks)
        blocks = local_search_trim(blocks, seed_counter, time_limit=trim_t)
        blocks = swap_local_search(blocks, seed_counter, time_limit=swap_t)
        blocks = final_polish(blocks, seed_counter, time_limit=polish_t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        if best_len == 69:
            break
            
        seed_counter += 1
    
    # Final polish on the best result with whatever time is left
    if best_blocks:
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time > 0.5:
            best_blocks = final_polish(best_blocks, 9999, time_limit=min(remaining_time, 5.0))
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
