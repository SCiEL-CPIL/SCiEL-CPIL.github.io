import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    points = list(range(v))
    all_t_sets = list(itertools.combinations(points, t))
    num_t_sets = len(all_t_sets)
    t_set_to_idx = {s: i for i, s in enumerate(all_t_sets)}
    
    # Precompute lookup for (t-1 set, point) -> t-set index
    lookup = {}
    for S in itertools.combinations(points, t-1):
        S_set = set(S)
        for x in points:
            if x not in S_set:
                T = tuple(sorted(S + (x,)))
                lookup[(S, x)] = t_set_to_idx[T]
                
    def get_gains(block, candidates, uncovered_mask):
        """Calculate gain for each candidate point."""
        gains = np.zeros(len(candidates))
        if len(block) < t - 1:
            return gains
            
        t_minus_1_subsets = list(itertools.combinations(block, t-1))
        
        for i, x in enumerate(candidates):
            indices = []
            for S in t_minus_1_subsets:
                idx_t = lookup.get((S, x), -1)
                if idx_t != -1:
                    indices.append(idx_t)
            if indices:
                gains[i] = np.sum(uncovered_mask[indices])
        return gains

    def greedy_cover(uncovered_mask, seed_idx=None):
        blocks = []
        if seed_idx is None:
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                return blocks
            seed_idx = uncovered_indices[0]
            
        seed_set = all_t_sets[seed_idx]
        block = [seed_set[0]]
        block_set = set(block)
        
        for _ in range(k - 1):
            if time.time() > deadline:
                break
            candidates = [x for x in points if x not in block_set]
            if not candidates:
                break
                
            gains = get_gains(block, candidates, uncovered_mask)
            
            if np.max(gains) == 0:
                best_point = random.choice(candidates)
            else:
                # Randomize tie-breaking slightly
                max_gain = np.max(gains)
                top_candidates = [c for c, g in zip(candidates, gains) if g == max_gain]
                best_point = random.choice(top_candidates)
                
            block.append(best_point)
            block_set.add(best_point)
            
        block.sort()
        blocks.append(block)
        
        for S in itertools.combinations(block, t):
            idx = t_set_to_idx[S]
            uncovered_mask[idx] = False
            
        return blocks

    def local_search(blocks, max_iterations=50):
        """Try to remove blocks and repair."""
        current_blocks = [list(b) for b in blocks]
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Shuffle blocks to try different orders
            random.shuffle(current_blocks)
            
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    break
                    
                # Try removing block i
                block_to_remove = current_blocks[i]
                remaining = current_blocks[:i] + current_blocks[i+1:]
                
                # Find uncovered sets
                # Reset mask
                mask = np.ones(num_t_sets, dtype=bool)
                for b in remaining:
                    for S in itertools.combinations(b, t):
                        idx = t_set_to_idx[S]
                        mask[idx] = False
                        
                uncovered_indices = np.where(mask)[0]
                if len(uncovered_indices) == 0:
                    # Block was redundant!
                    current_blocks.pop(i)
                    improved = True
                    break # Restart loop as list changed
                    
                # Try to cover uncovered sets with new blocks
                # Use greedy to cover just these
                new_blocks = []
                temp_mask = mask.copy()
                
                # Limit new blocks to 1 or 2 to ensure net gain or neutral
                # If we add 1 block, net change 0. 
                # If we add 0, net change -1.
                # We only accept if net change < 0 or (net change == 0 and some heuristic improves).
                # For simplicity, only accept if we can cover with 0 blocks (impossible if uncovered exists) 
                # or if we can cover with 1 block AND that block is "better" (hard to define).
                # Let's just try to cover with 1 block.
                
                # Pick a seed from uncovered
                seed_idx = uncovered_indices[0]
                seed_set = all_t_sets[seed_idx]
                new_block = [seed_set[0]]
                new_block_set = set(new_block)
                
                for _ in range(k - 1):
                    if time.time() > deadline:
                        break
                    candidates = [x for x in points if x not in new_block_set]
                    gains = get_gains(new_block, candidates, temp_mask)
                    if np.max(gains) == 0:
                        best_point = random.choice(candidates)
                    else:
                        max_gain = np.max(gains)
                        top_candidates = [c for c, g in zip(candidates, gains) if g == max_gain]
                        best_point = random.choice(top_candidates)
                    new_block.append(best_point)
                    new_block_set.add(best_point)
                    
                new_block.sort()
                
                # Check if this new block covers all uncovered sets
                # It's a single block, so it covers C(k,t) sets.
                # Uncovered sets might be more than that.
                # If len(uncovered_indices) <= C(k,t), it's possible.
                
                # Check coverage
                all_covered = True
                for S in itertools.combinations(new_block, t):
                    idx = t_set_to_idx[S]
                    temp_mask[idx] = False
                    
                # If any uncovered remain, this single block wasn't enough.
                if np.any(temp_mask):
                    continue # Skip this removal attempt
                
                # If we got here, 1 new block covered all holes.
                # Net change: -1 (remove) + 1 (add) = 0.
                # We accept if the new block is different from the old one?
                # Or just accept to explore state space.
                # To avoid cycles, we can require the new block to be different.
                if new_block != block_to_remove:
                    current_blocks[i] = new_block
                    improved = True
                    break # Restart loop
                    
        return current_blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run greedy multiple times
    for r in range(5):
        if time.time() > deadline:
            break
        random.seed(100 + r)
        mask = np.ones(num_t_sets, dtype=bool)
        blocks = greedy_cover(mask)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
        # Apply local search
        if time.time() < deadline - 20:
            improved_blocks = local_search(blocks, max_iterations=20)
            if len(improved_blocks) < best_size:
                best_size = len(improved_blocks)
                best_blocks = improved_blocks

    return best_blocks if best_blocks is not None else []

