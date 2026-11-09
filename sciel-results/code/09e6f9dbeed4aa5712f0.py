import itertools
import time
import random
import numpy as np
import scipy.sparse
import scipy.optimize

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # --- Try MILP ---
    try:
        # Build sparse matrix A
        rows = []
        cols = []
        data = []
        
        # Precompute subset indices for each block to speed up matrix building
        # Actually, just iterate
        for b_idx, block in enumerate(all_blocks):
            for s in itertools.combinations(block, t):
                s_idx = t_subset_to_idx[s]
                rows.append(s_idx)
                cols.append(b_idx)
                data.append(1)
                
        A = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(num_t_subsets, num_blocks))
        
        c = np.ones(num_blocks)
        # Constraint: A x >= 1  =>  -A x <= -1
        constraints = scipy.optimize.LinearConstraint(-A, ub=-1, lb=-np.inf)
        integrality = np.ones(num_blocks)
        
        options = {
            "time_limit": 100,
            "mip_rel_gap": 1e-6, # Accept small gap
            "presolve": True
        }
        
        res = scipy.optimize.milp(c, constraints=constraints, integrality=integrality, options=options)
        
        if res.status in [0, 1]: # Optimal or Feasible
            x = res.x
            if x is not None:
                block_indices = [i for i in range(num_blocks) if x[i] > 0.5]
                # Quick verification
                if is_valid_cover(all_blocks, block_indices, all_t_subsets, t_subset_to_idx, t):
                    return [list(all_blocks[i]) for i in block_indices]
                
    except Exception:
        pass

    # --- Fallback Heuristic ---
    # Precompute masks
    block_masks = [0] * num_blocks
    block_subsets = [[] for _ in range(num_blocks)]
    for i, block in enumerate(all_blocks):
        mask = 0
        subs = []
        for s in itertools.combinations(block, t):
            idx = t_subset_to_idx[s]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks[i] = mask
        block_subsets[i] = subs
        
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for idx in block_subsets[i]:
            subset_to_blocks[idx].append(i)

    def greedy_construct(randomize=False):
        uncovered = (1 << num_t_subsets) - 1
        chosen = []
        used = [False] * num_blocks
        
        while uncovered != 0:
            best_gain = -1
            best_idx = -1
            
            # Optimization: Iterate only over blocks that cover at least one uncovered subset?
            # Hard to do efficiently without scanning.
            # Scan all.
            for i in range(num_blocks):
                if used[i]:
                    continue
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and randomize and gain > 0:
                    if random.random() < 0.1: # 10% chance to pick tie
                        best_idx = i

            if best_idx == -1:
                break
                
            chosen.append(best_idx)
            used[best_idx] = True
            uncovered &= ~block_masks[best_idx]
            
        return chosen

    def prune_solution(chosen_indices):
        if not chosen_indices:
            return []
        
        counts = [0] * num_t_subsets
        for idx in chosen_indices:
            for s_idx in block_subsets[idx]:
                counts[s_idx] += 1
        
        current_list = list(chosen_indices)
        changed = True
        while changed:
            changed = False
            new_list = []
            for idx in current_list:
                essential = False
                for s_idx in block_subsets[idx]:
                    if counts[s_idx] == 1:
                        essential = True
                        break
                if not essential:
                    for s_idx in block_subsets[idx]:
                        counts[s_idx] -= 1
                    changed = True
                else:
                    new_list.append(idx)
            current_list = new_list
        
        return current_list

    def try_swaps(chosen_indices, max_iters=50):
        current_solution = list(chosen_indices)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            counts = [0] * num_t_subsets
            for idx in current_solution:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1
            
            for i, idx in enumerate(current_solution):
                essential_subs = [s for s in block_subsets[idx] if counts[s] == 1]
                if not essential_subs:
                    # Remove
                    for s in block_subsets[idx]:
                        counts[s] -= 1
                    current_solution.pop(i)
                    improved = True
                    break
                
                # Find candidate
                pool = subset_to_blocks[essential_subs[0]]
                best_cand = -1
                best_score = -1
                
                for cand_idx in pool:
                    if cand_idx in current_solution:
                        continue
                    # Check coverage of essential_subs
                    if all((block_masks[cand_idx] >> s) & 1 for s in essential_subs):
                        # Score: redundancy
                        score = 0
                        for s in block_subsets[cand_idx]:
                            if counts[s] > 1:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_cand = cand_idx
                            
                if best_cand != -1:
                    # Swap and check if it allows further pruning
                    old_idx = idx
                    current_solution[i] = best_cand
                    # Prune
                    pruned = prune_solution(current_solution)
                    if len(pruned) < len(current_solution):
                        current_solution = pruned
                        improved = True
                        break
                    else:
                        # Revert
                        current_solution[i] = old_idx
                        
            if not improved:
                break
                
        return current_solution

    best_solution = []
    best_len = float('inf')
    
    # Deterministic Greedy
    sol = greedy_construct(randomize=False)
    sol = prune_solution(sol)
    sol = try_swaps(sol)
    if len(sol) < best_len:
        best_len = len(sol)
        best_solution = sol

    # Randomized Greedy
    while time.time() < deadline - 10:
        sol = greedy_construct(randomize=True)
        sol = prune_solution(sol)
        sol = try_swaps(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            if best_len <= 131:
                break

    return [list(all_blocks[i]) for i in best_solution]

def is_valid_cover(all_blocks, indices, all_t_subsets, t_subset_to_idx, t):
    covered = set()
    for idx in indices:
        block = all_blocks[idx]
        for s in itertools.combinations(block, t):
            covered.add(s)
    return len(covered) == len(all_t_subsets)
