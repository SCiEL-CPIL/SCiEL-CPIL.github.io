import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog, milp, LinearConstraint
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Precompute subsets_of_block: for each block j, list of t-subset indices
    subsets_of_block = [[] for _ in range(n_k)]
    # Precompute blocks_of_subset: for each t-subset i, list of block indices
    blocks_of_subset = [[] for _ in range(n_t)]
    
    # Build sparse matrix for LP
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
    
    # 2. Try MILP directly with time limit
    try:
        if time.time() > deadline - 5:
            raise TimeoutError()
            
        M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
        
        # Minimize sum(x_j)
        # Subject to: M.T @ x >= ones(n_t)  =>  M.T @ x >= 1
        # x in {0, 1}
        
        c = np.ones(n_k)
        A = M.T.tocsr()  # Shape (n_t, n_k)
        b_lower = np.ones(n_t)
        b_upper = np.inf * np.ones(n_t)
        
        constraints = LinearConstraint(A, b_lower, b_upper)
        integrality = np.ones(n_k)  # All variables are binary
        
        # Use HiGHS solver with time limit
        time_limit = max(1, int(deadline - time.time() - 2))
        
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=[(0, 1)] * n_k,
            options={"time_limit": time_limit}
        )
        
        if res.success or res.status == 3:  # 3 = time limit reached, best solution found
            x = res.x
            if x is not None:
                # Round to binary
                x_bin = np.round(x).astype(int)
                # Ensure validity: only keep blocks that are actually selected
                selected_blocks = [j for j in range(n_k) if x_bin[j] == 1]
                
                # Verify coverage
                if selected_blocks:
                    covered = np.zeros(n_t, dtype=bool)
                    for b_idx in selected_blocks:
                        for s_idx in subsets_of_block[b_idx]:
                            covered[s_idx] = True
                    
                    if np.all(covered):
                        if len(selected_blocks) < best_size:
                            best_size = len(selected_blocks)
                            best_solution = selected_blocks.copy()
        
    except Exception:
        pass
    
    # 3. Fallback: Dynamic Greedy with LP Duals
    if best_solution is None or time.time() < deadline - 10:
        duals = None
        try:
            if time.time() > deadline - 5:
                raise TimeoutError()
                
            M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
            
            # Minimize sum(x_j)
            # Subject to: M.T @ x >= ones(n_t)  =>  -M.T @ x <= -ones(n_t)
            c = np.ones(n_k)
            A_ub = -M.T.tocsr()
            b_ub = -np.ones(n_t)
            bounds = [(0, 1)] * n_k
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                duals = res.ineqlin.marginals
                duals = np.abs(duals)
                
                if np.sum(duals) == 0:
                    duals = None
            else:
                duals = None
                
        except Exception:
            duals = None
            
        if duals is None:
            duals = np.ones(n_t)
            
        num_restarts = 3 if best_solution is not None else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize weights
            block_weights = np.zeros(n_k)
            for j in range(n_k):
                if subsets_of_block[j]:
                    block_weights[j] = np.sum(duals[subsets_of_block[j]])
                    
            # Add noise for diversification
            if restart > 0:
                block_weights += np.random.uniform(0, 0.01, n_k)
                
            # Heap of (-weight, index)
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
                    
                # Lazy deletion check
                if w > block_weights[j] + 1e-9:
                    continue
                    
                if w <= 1e-9:
                    break
                    
                # Select block j
                picked[j] = True
                solution_blocks.append(j)
                
                # Update weights
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
        # Standard greedy
        uncovered = set(range(n_t))
        res_blocks = []
        while uncovered:
            best_b = -1
            max_cov = -1
            for j in range(n_k):
                cnt = 0
                for s in subsets_of_block[j]:
                    if s in uncovered:
                        cnt += 1
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
        best_size = len(res_blocks)
        
    # 4. Post-processing: Deletion Pass
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
            
    # 5. Local Search: Swap and Delete
    if best_solution is not None and time.time() < deadline - 5:
        improved = True
        max_local_steps = 20
        step = 0
        
        while improved and step < max_local_steps and time.time() < deadline - 5:
            improved = False
            step += 1
            
            # Recalculate cover counts
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in best_solution:
                for s_idx in subsets_of_block[b_idx]:
                    cover_count[s_idx] += 1
            
            # Identify blocks with unique coverage
            for i, b_idx in enumerate(best_solution):
                if time.time() > deadline - 5:
                    break
                    
                unique_subs = []
                for s_idx in subsets_of_block[b_idx]:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                    
                if not unique_subs:
                    # This block is redundant? 
                    for s_idx in subsets_of_block[b_idx]:
                        cover_count[s_idx] -= 1
                    best_solution.pop(i)
                    best_size -= 1
                    improved = True
                    break # Restart loop as indices changed
                    
                # Check if we can swap this block
                union_points = set()
                for s_idx in unique_subs:
                    union_points.update(t_subsets[s_idx])
                    
                if len(union_points) > k:
                    continue # No single block can cover these unique subsets
                    
                # Find candidate replacement blocks
                candidate_blocks = set(blocks_of_subset[unique_subs[0]])
                candidate_blocks.discard(b_idx)
                
                for s_idx in unique_subs[1:]:
                    candidate_blocks.intersection_update(blocks_of_subset[s_idx])
                    if not candidate_blocks:
                        break
                        
                if not candidate_blocks:
                    continue
                    
                best_candidate = -1
                best_score = -1
                
                cand_list = list(candidate_blocks)
                random.shuffle(cand_list)
                for cand_idx in cand_list[:5]:
                    score = len(subsets_of_block[cand_idx])
                    if score > best_score:
                        best_score = score
                        best_candidate = cand_idx
                        
                if best_candidate != -1:
                    best_solution[i] = best_candidate
                    improved = True
                    break # Restart loop to recalculate cover counts
        
    # Final Deletion Pass
    if best_solution is not None and time.time() < deadline - 2:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
                
        changed = True
        while changed and time.time() < deadline - 2:
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
