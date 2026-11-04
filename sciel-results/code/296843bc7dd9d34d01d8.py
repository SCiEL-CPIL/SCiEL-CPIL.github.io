import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # 1. Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute all k-blocks and their coverage
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Build coverage matrix: rows = blocks, cols = t-subsets
    # Use sparse representation via bitmasks for efficiency
    block_masks = [0] * n_k_blocks
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # 3. First pass: Greedy set cover to get a good initial solution
    def greedy_cover():
        uncovered = (1 << n_t_subs) - 1
        selected = []
        while uncovered != 0:
            if time.time() > deadline - 60:
                break
            best_idx = -1
            best_gain = -1
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
        return selected
    
    initial_solution = greedy_cover()
    best_solution = initial_solution[:]
    best_size = len(initial_solution)
    
    # 4. Redundancy removal (remove blocks that aren't critical)
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        cover_count = [0] * n_t_subs
        masks = [block_masks[i] for i in blocks]
        for mask in masks:
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            new_masks = []
            for i in range(len(blocks)):
                mask = masks[i]
                is_critical = False
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        is_critical = True
                        break
                    m ^= lsb
                if is_critical:
                    new_blocks.append(blocks[i])
                    new_masks.append(mask)
                else:
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    changed = True
            blocks = new_blocks
            masks = new_masks
        return blocks
    
    best_solution = remove_redundant(best_solution)
    best_size = len(best_solution)
    
    # 5. LP-based approach: Solve set cover LP relaxation, then round
    if time.time() < deadline - 80:
        # Build A matrix for LP: A[i,j] = 1 if block i covers t-subset j
        # We want min sum(x_i) s.t. A @ x >= 1, 0 <= x <= 1
        # This is a large LP, so we'll use a subset of blocks for tractability
        # Use blocks from the greedy solution plus some random ones
        
        # Select a pool of candidate blocks
        pool_size = min(n_k_blocks, 2000)
        if pool_size < n_k_blocks:
            # Use greedy solution blocks plus random blocks
            greedy_set = set(best_solution)
            remaining = [i for i in range(n_k_blocks) if i not in greedy_set]
            random.shuffle(remaining)
            pool = list(greedy_set) + remaining[:pool_size - len(greedy_set)]
        else:
            pool = list(range(n_k_blocks))
        
        pool_set = set(pool)
        pool_to_idx = {b: i for i, b in enumerate(pool)}
        n_pool = len(pool)
        
        # Build constraint matrix for LP
        # A[i,j] = 1 if pool block i covers t-subset j
        A = np.zeros((n_t_subs, n_pool), dtype=np.float64)
        for pi, bi in enumerate(pool):
            mask = block_masks[bi]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                A[idx, pi] = 1.0
                m ^= lsb
        
        # Solve LP: min c^T x s.t. A^T x >= 1, 0 <= x <= 1
        c = np.ones(n_pool, dtype=np.float64)
        b = np.ones(n_t_subs, dtype=np.float64)
        
        # Bounds: 0 <= x_i <= 1
        bounds = [(0, 1)] * n_pool
        
        try:
            result = linprog(c, A_ub=-A.T, b_ub=-b, bounds=bounds, method='highs', maxiter=5000)
            if result.success and result.x is not None:
                x_lp = result.x
                
                # Randomized rounding: sample blocks with probability x_lp[i]
                # Do multiple rounds to get good solutions
                for round_num in range(5):
                    if time.time() > deadline - 20:
                        break
                    
                    # Sample blocks based on LP weights
                    sampled = []
                    for i in range(n_pool):
                        if x_lp[i] > 0.01 and random.random() < x_lp[i]:
                            sampled.append(pool[i])
                    
                    # Complete the solution with greedy for uncovered
                    uncovered = (1 << n_t_subs) - 1
                    for bi in sampled:
                        uncovered &= ~block_masks[bi]
                    
                    if uncovered != 0:
                        # Greedy completion
                        while uncovered != 0:
                            if time.time() > deadline - 15:
                                break
                            best_idx = -1
                            best_gain = -1
                            for i in range(n_k_blocks):
                                if i in set(sampled):
                                    continue
                                gain = (block_masks[i] & uncovered).bit_count()
                                if gain > best_gain:
                                    best_gain = gain
                                    best_idx = i
                            if best_idx == -1 or best_gain == 0:
                                break
                            sampled.append(best_idx)
                            uncovered &= ~block_masks[best_idx]
                    
                    # Remove redundancy
                    sampled = remove_redundant(sampled)
                    
                    if len(sampled) < best_size:
                        best_size = len(sampled)
                        best_solution = sampled[:]
                        
        except Exception as e:
            pass
    
    # 6. Local search: Try swapping blocks to reduce count
    if time.time() < deadline - 10:
        # For each block in the solution, try removing it and see if we can
        # cover the newly uncovered subsets with one other block
        current = best_solution[:]
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Compute coverage counts
            cover_count = [0] * n_t_subs
            masks = [block_masks[i] for i in current]
            for mask in masks:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] += 1
                    m ^= lsb
            
            # Try removing each block
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                mask = masks[i]
                # Find subsets that would become uncovered
                newly_uncovered = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        newly_uncovered.append(idx)
                    m ^= lsb
                
                if not newly_uncovered:
                    # Can remove this block
                    current.pop(i)
                    masks.pop(i)
                    for idx in newly_uncovered:
                        cover_count[idx] -= 1
                    improved = True
                    continue
                
                # Try to find a single block that covers all newly_uncovered
                # Convert newly_uncovered to a bitmask
                target_mask = 0
                for idx in newly_uncovered:
                    target_mask |= (1 << idx)
                
                # Find a block that covers target_mask
                replacement = -1
                for j in range(n_k_blocks):
                    if j in set(current):
                        continue
                    if (block_masks[j] & target_mask) == target_mask:
                        # Check if this block is "worth it" - it should cover
                        # enough of the target to be useful
                        replacement = j
                        break
                
                if replacement != -1:
                    # Swap: remove block i, add replacement
                    # This keeps the same count, but might enable further removals
                    # Only do this if it seems promising
                    pass  # Skip swaps that keep count same for now
    
    # Final redundancy removal
    best_solution = remove_redundant(best_solution)
    
    # Convert to list of lists
    blocks = [list(all_k_blocks[i]) for i in best_solution]
    
    return blocks
