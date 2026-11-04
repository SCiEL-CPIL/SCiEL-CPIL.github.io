import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # 40s safety margin for 150s limit

    # Precompute all t-subsets and blocks as bitmasks
    # v=19, so bitmasks fit in Python int
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_mask_to_idx = {}
    tsubset_masks = []
    for i, ts in enumerate(all_tsubsets):
        mask = 0
        for p in ts:
            mask |= (1 << p)
        tsubset_masks.append(mask)
        tsubset_mask_to_idx[mask] = i

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_masks = []
    # Precompute which t-subset indices each block covers
    block_covered_ts_indices = []
    
    for blk in all_blocks:
        mask = 0
        for p in blk:
            mask |= (1 << p)
        block_masks.append(mask)
        
        # Find t-subsets contained in this block
        covered_indices = []
        # Iterate over all t-subsets of the block
        for ts in itertools.combinations(blk, t):
            ts_mask = 0
            for p in ts:
                ts_mask |= (1 << p)
            idx = tsubset_mask_to_idx[ts_mask]
            covered_indices.append(idx)
        block_covered_ts_indices.append(covered_indices)

    # Function to run one greedy+reduce cycle
    def run_greedy_reduce(seed_val):
        random.seed(seed_val)
        
        # Greedy Phase
        # counts[i] = number of selected blocks covering t-subset i
        counts = np.zeros(n_tsubsets, dtype=np.int8)
        selected_block_indices = []
        remaining = set(range(n_tsubsets))
        
        # To speed up greedy, we can maintain a list of active blocks
        # But C(19,8) is small enough to scan all each time?
        # 75k blocks * 56 ts * 350 steps = 1.4B ops. 
        # In Python, a loop of 1.4B is ~10-20s. 
        # We can optimize by only considering blocks that cover at least one remaining ts.
        # But maintaining that list is tricky.
        
        # Optimization: Pre-compute the list of block indices for each t-subset?
        # ts_to_blocks[ts_idx] = list of block indices covering it.
        # Then, for each remaining ts, we know which blocks can cover it.
        # But we want the block covering the MOST remaining ts.
        
        # Let's try the direct scan first, it might be fast enough with numpy?
        # No, the inner loop is Python.
        
        # Alternative: Use a heuristic.
        # Pick a random remaining t-subset. 
        # Pick the block covering it that covers the most other remaining ts.
        # This is "Greedy with Pivot".
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Pick a pivot t-subset (random or first)
            # Random pivot helps with diversity in restarts
            pivot_ts = random.choice(list(remaining))
            
            # Find blocks that cover pivot_ts
            # We need a mapping from ts to blocks.
            # Let's precompute ts_to_blocks once.
            
            best_block_idx = -1
            best_gain = -1
            
            # Get candidate blocks covering the pivot
            candidates = ts_to_blocks[pivot_ts]
            
            for b_idx in candidates:
                # Calculate gain: how many remaining ts does this block cover?
                gain = 0
                for ts_idx in block_covered_ts_indices[b_idx]:
                    if ts_idx in remaining:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                # Should not happen if valid
                break
                
            selected_block_indices.append(best_block_idx)
            
            # Update counts and remaining
            for ts_idx in block_covered_ts_indices[best_block_idx]:
                counts[ts_idx] += 1
                if counts[ts_idx] == 1:
                    remaining.discard(ts_idx)
            
        # Reduction Phase
        # Remove redundant blocks
        # A block is redundant if all ts it covers have count > 1
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_block_indices) - 1, -1, -1):
                b_idx = selected_block_indices[i]
                is_redundant = True
                for ts_idx in block_covered_ts_indices[b_idx]:
                    if counts[ts_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove block
                    selected_block_indices.pop(i)
                    for ts_idx in block_covered_ts_indices[b_idx]:
                        counts[ts_idx] -= 1
                    changed = True
                    if time.time() > deadline:
                        return selected_block_indices
        
        return selected_block_indices

    # Precompute ts_to_blocks
    ts_to_blocks = [[] for _ in range(n_tsubsets)]
    for b_idx, ts_list in enumerate(block_covered_ts_indices):
        for ts_idx in ts_list:
            ts_to_blocks[ts_idx].append(b_idx)

    best_solution = None
    best_len = float('inf')
    
    # Multiple restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Use different seeds for diversity
        sol = run_greedy_reduce(i * 12345)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol[:]
            
        # If we hit a very low number, we can stop early or continue
        # Best known is 335. If we get below 335, great.
        if best_len <= 330:
             # Try a few more to see if we can go lower, but don't waste time
             pass

    # Convert block indices back to lists of points
    final_blocks = []
    if best_solution:
        for b_idx in best_solution:
            final_blocks.append(list(all_blocks[b_idx]))
    else:
        # Fallback: simple greedy without restarts if something went wrong
        # Or just return empty? No, must be valid.
        # Let's do one deterministic run
        sol = run_greedy_reduce(0)
        for b_idx in sol:
            final_blocks.append(list(all_blocks[b_idx]))

    return final_blocks
