import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    rows = []
    cols = []
    data = []
    
    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
    
    best_solution = None
    best_size = float('inf')
    
    # 2. LP Relaxation for duals
    duals = None
    try:
        if time.time() > deadline - 5:
            raise TimeoutError()
            
        M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
        
        c = np.ones(n_k)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds = [(0, 1)] * n_k
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            duals = np.abs(res.ineqlin.marginals)
            if np.sum(duals) == 0:
                duals = None
    except Exception:
        duals = None
    
    if duals is None:
        duals = np.ones(n_t)
    
    # 3. Multi-restart heap-based greedy with LP duals
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        block_weights = np.zeros(n_k)
        for j in range(n_k):
            if subsets_of_block[j]:
                block_weights[j] = np.sum(duals[subsets_of_block[j]])
                
        if restart > 0:
            block_weights += np.random.uniform(0, 0.01, n_k)
            
        heap = [(-block_weights[j], j) for j in range(n_k)]
        heapq.heapify(heap)
        
        covered = np.zeros(n_t, dtype=bool)
        picked = np.zeros(n_k, dtype=bool)
        solution_blocks = []
        
        while heap:
            if time.time() > deadline - 5:
                break
                
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            
            if picked[j]:
                continue
            if w > block_weights[j] + 1e-9:
                continue
            if w <= 1e-9:
                break
                
            picked[j] = True
            solution_blocks.append(j)
            
            for s_idx in subsets_of_block[j]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    y_s = duals[s_idx]
                    if y_s > 0:
                        for b_prime in blocks_of_subset[s_idx]:
                            if not picked[b_prime]:
                                block_weights[b_prime] -= y_s
                                heapq.heappush(heap, (-block_weights[b_prime], b_prime))
                                
            if np.all(covered):
                break
                
        if len(solution_blocks) < best_size and np.all(covered):
            best_size = len(solution_blocks)
            best_solution = solution_blocks.copy()
    
    # Fallback
    if best_solution is None:
        uncovered = set(range(n_t))
        res_blocks = []
        while uncovered:
            best_b = -1
            max_cov = -1
            for j in range(n_k):
                cnt = sum(1 for s in subsets_of_block[j] if s in uncovered)
                if cnt > max_cov:
                    max_cov = cnt
                    best_b = j
            if best_b == -1 or max_cov == 0:
                break
            res_blocks.append(best_b)
            for s in subsets_of_block[best_b]:
                if s in uncovered:
                    uncovered.remove(s)
        best_solution = res_blocks
        best_size = len(best_solution)
    
    # 4. Deletion Pass
    if best_solution is not None:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
                
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block[b_idx]
                can_remove = True
                for s_idx in subs_b:
                    if cover_count[s_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    cover_count[subs_b] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)
            best_solution = new_solution
            best_size = len(best_solution)
    
    # 5. 2-for-1 / 1-for-1 Swap Local Search
    # Try to improve by replacing blocks or removing and re-covering
    if best_solution is not None and time.time() < deadline - 3:
        # Compute cover counts
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            new_solution = best_solution.copy()
            
            # Try 1-for-1 replacement: remove block i, find if another block can cover all its unique subsets
            for i, b_idx in enumerate(best_solution):
                if time.time() > deadline - 3:
                    break
                subs_b = subsets_of_block[b_idx]
                
                # Find subsets that would become uncovered if we remove b_idx
                unique_subs = [s for s in subs_b if cover_count[s] == 1]
                
                if not unique_subs:
                    # Can remove entirely (should have been caught by deletion, but just in case)
                    cover_count[subs_b] -= 1
                    new_solution.pop(i)
                    improved = True
                    # Recompute cover counts
                    cover_count = np.zeros(n_t, dtype=np.int32)
                    for b_idx2 in new_solution:
                        for s_idx in subsets_of_block[b_idx2]:
                            cover_count[s_idx] += 1
                    break
                else:
                    # Try to find a replacement block that covers all unique_subs
                    # and doesn't conflict
                    best_replacement = -1
                    best_gain = 0
                    for b_new in range(n_k):
                        if b_new in best_solution:
                            continue
                        # Check if b_new covers all unique_subs
                        new_subs = set(subsets_of_block[b_new])
                        if all(s in new_subs for s in unique_subs):
                            # Gain: how many subsets does b_new cover that are currently covered only by b_idx?
                            # Actually, we just need it to cover the unique ones
                            # The gain is 0 for 1-for-1, but we might reduce redundancy
                            # For now, just accept any valid replacement
                            best_replacement = b_new
                            break
                    
                    if best_replacement != -1:
                        # Remove b_idx, add best_replacement
                        cover_count[subs_b] -= 1
                        new_solution[i] = best_replacement
                        for s_idx in subsets_of_block[best_replacement]:
                            cover_count[s_idx] += 1
                        improved = True
                        break
            
            if improved:
                best_solution = new_solution
                best_size = len(best_solution)
    
    # 6. Final deletion pass after swaps
    if best_solution is not None and time.time() < deadline - 1:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
                
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block[b_idx]
                can_remove = True
                for s_idx in subs_b:
                    if cover_count[s_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    cover_count[subs_b] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)
            best_solution = new_solution
            best_size = len(best_solution)
    
    return [list(k_subsets[idx]) for idx in best_solution]
