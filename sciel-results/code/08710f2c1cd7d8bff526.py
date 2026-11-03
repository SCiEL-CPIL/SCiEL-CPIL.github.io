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
    
    # Precompute subsets_of_block and blocks_of_subset
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    # Build adjacency efficiently
    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
    
    # Convert to numpy arrays for faster indexing
    subsets_of_block_arr = [np.array(subs, dtype=np.int32) for subs in subsets_of_block]
    blocks_of_subset_arr = [np.array(subs, dtype=np.int32) for subs in blocks_of_subset]
    
    best_solution = None
    best_size = float('inf')
    
    # 2. Try LP Relaxation to get duals (with time limit)
    duals = None
    try:
        if time.time() > deadline - 15:
            raise TimeoutError()
            
        # Build sparse matrix
        rows = []
        cols = []
        for j in range(n_k):
            for i in subsets_of_block[j]:
                rows.append(i)
                cols.append(j)
        
        M = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_t, n_k))
        
        c = np.ones(n_k)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds = [(0, 1)] * n_k
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                     options={'presolve': True, 'time_limit': 8})
        
        if res.success and res.ineqlin is not None:
            duals = np.abs(res.ineqlin.marginals)
            if np.sum(duals) < 1e-10:
                duals = None
    except Exception:
        duals = None
        
    if duals is None:
        duals = np.ones(n_t)
    
    # 3. Improved Dynamic Greedy with better weight management
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize weights from duals
        block_weights = np.zeros(n_k)
        for j in range(n_k):
            if len(subsets_of_block[j]) > 0:
                block_weights[j] = np.sum(duals[subsets_of_block_arr[j]])
                
        # Perturbation strategy for diversity
        if restart == 0:
            pass  # No perturbation for first run
        elif restart == 1:
            # Small random noise
            block_weights += np.random.uniform(0, 0.005, n_k)
        elif restart == 2:
            # Scaled noise
            block_weights *= (1 + np.random.uniform(-0.02, 0.02, n_k))
        elif restart == 3:
            # Random subset of duals zeroed
            mask = np.random.random(n_t) < 0.1
            d = duals.copy()
            d[mask] = 0
            block_weights = np.zeros(n_k)
            for j in range(n_k):
                if len(subsets_of_block[j]) > 0:
                    block_weights[j] = np.sum(d[subsets_of_block_arr[j]])
        elif restart == 4:
            # Amplify high-weight subsets
            d = duals.copy()
            d = d ** 1.5
            d = d / d.max() if d.max() > 0 else d
            block_weights = np.zeros(n_k)
            for j in range(n_k):
                if len(subsets_of_block[j]) > 0:
                    block_weights[j] = np.sum(d[subsets_of_block_arr[j]])
        elif restart == 5:
            # Random tie-breaking bias
            block_weights += np.random.permutation(np.arange(n_k)) * 1e-6
        elif restart == 6:
            # Geometric noise
            block_weights *= np.exp(np.random.normal(0, 0.01, n_k))
        else:
            # Mixed strategy
            block_weights += np.random.uniform(0, 0.003, n_k)
            block_weights *= (1 + np.random.uniform(-0.01, 0.01, n_k))
            
        # Use a priority queue with lazy deletion
        covered = np.zeros(n_t, dtype=bool)
        picked = np.zeros(n_k, dtype=bool)
        solution_blocks = []
        
        # Build initial heap
        heap = [(-block_weights[j], j) for j in range(n_k) if block_weights[j] > 1e-12]
        heapq.heapify(heap)
        
        while heap:
            if time.time() > deadline - 3:
                break
                
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            
            if picked[j]:
                continue
            if w > block_weights[j] + 1e-9:
                continue
            if w <= 1e-12:
                break
                
            # Select block j
            picked[j] = True
            solution_blocks.append(j)
            
            # Update: for each newly covered t-subset, decrement weights of blocks containing it
            for s_idx in subsets_of_block_arr[j]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    y_s = duals[s_idx]
                    if y_s > 1e-15:
                        for b_prime in blocks_of_subset_arr[s_idx]:
                            if not picked[b_prime]:
                                block_weights[b_prime] -= y_s
                                if block_weights[b_prime] > 1e-12:
                                    heapq.heappush(heap, (-block_weights[b_prime], b_prime))
                                
            if np.all(covered):
                break
                
        if np.all(covered) and len(solution_blocks) < best_size:
            best_size = len(solution_blocks)
            best_solution = solution_blocks.copy()
            
    # Fallback
    if best_solution is None:
        best_solution = []
        covered = np.zeros(n_t, dtype=bool)
        while not np.all(covered):
            best_b = -1
            max_new = -1
            for j in range(n_k):
                if any(not covered[s] for s in subsets_of_block[j]):
                    new_cov = sum(1 for s in subsets_of_block[j] if not covered[s])
                    if new_cov > max_new:
                        max_new = new_cov
                        best_b = j
            if best_b == -1:
                break
            best_solution.append(best_b)
            for s in subsets_of_block[best_b]:
                covered[s] = True
        best_size = len(best_solution)
    
    # 4. Deletion pass
    if best_solution:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block_arr[b_idx]] += 1
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            new_sol = []
            for b_idx in best_solution:
                subs = subsets_of_block_arr[b_idx]
                if np.all(cover_count[subs] > 1):
                    cover_count[subs] -= 1
                    changed = True
                else:
                    new_sol.append(b_idx)
            best_solution = new_sol
            best_size = len(best_solution)
    
    # 5. 2-swap improvement: try replacing two blocks with one
    if best_solution and time.time() < deadline - 3:
        sol_set = set(best_solution)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in best_solution:
                cover_count[subsets_of_block_arr[b_idx]] += 1
            
            for i in range(len(best_solution)):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, len(best_solution)):
                    if time.time() > deadline - 2:
                        break
                    b1, b2 = best_solution[i], best_solution[j]
                    # Check if removing both leaves some subsets uncovered
                    cover_without = cover_count.copy()
                    cover_without[subsets_of_block_arr[b1]] -= 1
                    cover_without[subsets_of_block_arr[b2]] -= 1
                    if np.any(cover_without < 1):
                        continue
                    # Try to find a single block that covers all subsets that were only covered by b1 or b2
                    # Subsets that become uncovered or have count 1 after removing both
                    critical = np.where((cover_without <= 1) & (cover_count > 0))[0]
                    if len(critical) == 0:
                        # Can just remove both
                        new_sol = [b for idx, b in enumerate(best_solution) if idx not in (i, j)]
                        best_solution = new_sol
                        best_size = len(best_solution)
                        improved = True
                        break
                    # Find a block that covers all critical subsets
                    found = False
                    for cand in range(n_k):
                        if cand in sol_set:
                            continue
                        if all(c in set(subsets_of_block[cand]) for c in critical):
                            # Replace b1, b2 with cand
                            new_sol = [b for idx, b in enumerate(best_solution) if idx not in (i, j)]
                            new_sol.append(cand)
                            best_solution = new_sol
                            best_size = len(best_solution)
                            sol_set = set(best_solution)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break
            if improved:
                # Re-run deletion
                cover_count = np.zeros(n_t, dtype=np.int32)
                for b_idx in best_solution:
                    cover_count[subsets_of_block_arr[b_idx]] += 1
                changed2 = True
                while changed2:
                    changed2 = False
                    new_sol = []
                    for b_idx in best_solution:
                        subs = subsets_of_block_arr[b_idx]
                        if np.all(cover_count[subs] > 1):
                            cover_count[subs] -= 1
                            changed2 = True
                        else:
                            new_sol.append(b_idx)
                    best_solution = new_sol
                    best_size = len(best_solution)
    
    return [list(k_subsets[idx]) for idx in best_solution]
