import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill
    
    # 1. Map t-subsets to IDs
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_map = {sub: i for i, sub in enumerate(t_subs)}
    num_t = len(t_subs)
    
    # 2. Pre-compute block -> t-subset IDs
    k_subs = list(itertools.combinations(range(v), k))
    num_k = len(k_subs)
    
    t_in_k = list(itertools.combinations(range(k), t))
    num_t_in_k = len(t_in_k)
    
    block_t_ids = np.zeros((num_k, num_t_in_k), dtype=np.int32)
    
    for i, block in enumerate(k_subs):
        for j, t_indices in enumerate(t_in_k):
            global_t_sub = tuple(sorted(block[x] for x in t_indices))
            block_t_ids[i, j] = t_sub_map[global_t_sub]
    
    # Build index mapping
    block_to_idx = {block: i for i, block in enumerate(k_subs)}
    
    # 3. Greedy Cover to get initial solution
    covered = np.zeros(num_t, dtype=bool)
    used_blocks = set()
    solution = []
    
    while not covered.all():
        if time.time() > deadline - 30.0:
            break
            
        G = np.sum(~covered[block_t_ids], axis=1)
        G[list(used_blocks)] = -1
        best_idx = np.argmax(G)
        
        if G[best_idx] <= 0:
            break
            
        used_blocks.add(best_idx)
        solution.append(list(k_subs[best_idx]))
        covered[block_t_ids[best_idx]] = True
    
    # 4. Local Search: Remove redundant blocks
    def remove_redundant(sol):
        if not sol:
            return sol, False
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in sol:
            idx = block_to_idx[tuple(b)]
            cover_count[block_t_ids[idx]] += 1
            
        new_sol = []
        changed = False
        for b in sol:
            idx = block_to_idx[tuple(b)]
            min_count = np.min(cover_count[block_t_ids[idx]])
            if min_count > 1:
                changed = True
                cover_count[block_t_ids[idx]] -= 1
            else:
                new_sol.append(b)
        return new_sol, changed
    
    for _ in range(10):
        if time.time() > deadline - 30.0:
            break
        solution, changed = remove_redundant(solution)
        if not changed:
            break
    
    # 5. Pairwise merge local search
    def pairwise_merge(sol):
        if len(sol) < 2:
            return sol, False
        if time.time() > deadline - 15.0:
            return sol, False
        
        n = len(sol)
        # Compute coverage counts
        cover_count = np.zeros(num_t, dtype=np.int32)
        sol_indices = [block_to_idx[tuple(b)] for b in sol]
        for idx in sol_indices:
            cover_count[block_t_ids[idx]] += 1
        
        # For each block, find t-subsets it uniquely covers
        unique_covered = []
        for idx in sol_indices:
            tids = block_t_ids[idx]
            unique = [tid for tid in tids if cover_count[tid] == 1]
            unique_covered.append(set(unique))
        
        changed = False
        new_sol = list(sol)
        
        # Try all pairs
        for i in range(n):
            if time.time() > deadline - 15.0:
                break
            for j in range(i + 1, n):
                if new_sol[i] is None or new_sol[j] is None:
                    continue
                # Union of uniquely covered t-subsets
                union_unique = unique_covered[i] | unique_covered[j]
                if not union_unique:
                    continue
                
                # Check if there exists a k-block containing all these t-subsets
                all_elements = set()
                for tid in union_unique:
                    all_elements.update(t_subs[tid])
                
                if len(all_elements) > k:
                    continue
                
                remaining = set(range(v)) - all_elements
                need = k - len(all_elements)
                
                if need < 0:
                    continue
                
                sorted_all = sorted(all_elements)
                sorted_rem = sorted(remaining)
                if need == 0:
                    new_block = sorted_all
                else:
                    new_block = sorted(sorted_all + sorted_rem[:need])
                
                new_block_tuple = tuple(new_block)
                if new_block_tuple in block_to_idx:
                    new_idx = block_to_idx[new_block_tuple]
                    new_tids = block_t_ids[new_idx]
                    new_tid_set = set(new_tids.tolist())
                    if union_unique.issubset(new_tid_set):
                        new_sol[i] = None
                        new_sol[j] = None
                        new_sol.append(list(new_block_tuple))
                        changed = True
                        break
            if changed:
                break
        
        if changed:
            new_sol = [b for b in new_sol if b is not None]
        
        return new_sol, changed
    
    # Apply pairwise merge repeatedly
    for _ in range(50):
        if time.time() > deadline - 15.0:
            break
        solution, changed = pairwise_merge(solution)
        if not changed:
            break
    
    # After merges, remove redundant again
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
        solution, changed = remove_redundant(solution)
        if not changed:
            break
    
    # 6. WalkSAT-style local search
    included = np.zeros(num_k, dtype=bool)
    for b in solution:
        included[block_to_idx[tuple(b)]] = True
    
    cover_count = np.zeros(num_t, dtype=np.int32)
    for i in range(num_k):
        if included[i]:
            cover_count[block_t_ids[i]] += 1
    
    num_uncovered = np.sum(cover_count == 0)
    best_num_uncovered = num_uncovered
    best_included = included.copy()
    
    noise = 0.3
    max_steps = 3000
    
    step = 0
    while step < max_steps and time.time() < deadline - 10.0:
        step += 1
        
        if random.random() < noise:
            i = random.randint(0, num_k - 1)
        else:
            delta = np.zeros(num_k, dtype=np.int32)
            
            inc_mask = included
            if np.any(inc_mask):
                inc_block_tids = block_t_ids[inc_mask]
                delta[inc_mask] = np.sum(cover_count[inc_block_tids] == 1, axis=1)
            
            not_inc_mask = ~included
            if np.any(not_inc_mask):
                not_inc_block_tids = block_t_ids[not_inc_mask]
                delta[not_inc_mask] = -np.sum(cover_count[not_inc_block_tids] == 0, axis=1)
            
            best_i = np.argmin(delta)
            i = int(best_i)
        
        if included[i]:
            cover_count[block_t_ids[i]] -= 1
            included[i] = False
        else:
            cover_count[block_t_ids[i]] += 1
            included[i] = True
        
        num_uncovered = np.sum(cover_count == 0)
        
        if num_uncovered < best_num_uncovered:
            best_num_uncovered = num_uncovered
            best_included = included.copy()
            if best_num_uncovered == 0:
                break
    
    best_solution = []
    for i in range(num_k):
        if best_included[i]:
            best_solution.append(list(k_subs[i]))
    
    # 7. Final redundancy removal
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        best_solution, changed = remove_redundant(best_solution)
        if not changed:
            break
    
    # 8. Final pairwise merge pass
    for _ in range(30):
        if time.time() > deadline - 3.0:
            break
        best_solution, changed = pairwise_merge(best_solution)
        if not changed:
            break
    
    for _ in range(10):
        if time.time() > deadline - 2.0:
            break
        best_solution, changed = remove_redundant(best_solution)
        if not changed:
            break
    
    if len(best_solution) < len(solution):
        solution = best_solution
    
    return solution
