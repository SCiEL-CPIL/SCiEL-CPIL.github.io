import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety
    
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
    
    # 2. Try LP Relaxation to get duals
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
            # Duals for the constraints A_ub x <= b_ub
            duals = res.ineqlin.marginals
            duals = np.abs(duals)
            
            if np.sum(duals) == 0:
                duals = None
        else:
            duals = None
            
    except Exception:
        duals = None
        
    # 3. Dynamic Greedy with LP Duals
    if duals is None:
        duals = np.ones(n_t)
        
    num_restarts = 5
    
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
    # Try to swap blocks to enable further deletions
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
                    # This block is redundant
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
                
                # If unique_subs is large, the intersection of blocks_of_subset for all of them is small.
                for s_idx in unique_subs[1:]:
                    candidate_blocks.intersection_update(blocks_of_subset[s_idx])
                    if not candidate_blocks:
                        break
                
                if not candidate_blocks:
                    continue
                
                # Try swapping with the first valid candidate
                best_candidate = -1
                best_score = -1
                
                # Limit candidates to check for speed
                cand_list = list(candidate_blocks)
                random.shuffle(cand_list)
                for cand_idx in cand_list[:5]: # Check up to 5 candidates
                    # Heuristic: Prefer candidates that cover many other subsets.
                    score = len(subsets_of_block[cand_idx])
                    if score > best_score:
                        best_score = score
                        best_candidate = cand_idx
                
                if best_candidate != -1:
                    # Perform swap
                    best_solution[i] = best_candidate
                    improved = True
                    break # Restart loop to recalculate cover counts
        
    # 6. Lagrangian Relaxation Local Search
    # Formulate as min sum x_j + lambda * sum (1 - sum_{j in S_i} x_j) for uncovered subsets
    # We adjust lambda dynamically to guide the search.
    
    if best_solution is not None and time.time() < deadline - 10:
        # Compute initial cover counts
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
        
        # Penalty weights for uncovered subsets
        # If a subset is uncovered, penalty is high. If covered, 0.
        # We want to minimize: |Solution| + sum_{i: uncovered} penalty[i]
        # But we enforce coverage, so we use this to guide swaps.
        
        # Dynamic penalty: start low, increase for stubbornly uncovered subsets
        # Since we maintain a valid cover, "uncovered" means "singly covered" in the context of deletion potential?
        # Actually, let's use the concept of "criticality".
        # A subset with cover_count == 1 is critical.
        # We want to reduce the number of critical subsets, or find swaps that eliminate them.
        
        # Let's try a stochastic local search where we pick a random block to remove,
        # and then find the best replacement to restore coverage, preferring ones that
        # reduce the total "criticality" or allow further deletions.
        
        current_solution = best_solution.copy()
        current_size = len(current_solution)
        
        # Precompute for each block, the list of subsets it covers (already have subsets_of_block)
        
        # Penalty vector: high weight for subsets with low cover count
        # We'll update this dynamically.
        penalties = np.zeros(n_t)
        
        for _ in range(50): # 50 iterations of Lagrangian-guided local search
            if time.time() > deadline - 5:
                break
            
            # Update penalties: higher for subsets with cover_count == 1
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in current_solution:
                for s_idx in subsets_of_block[b_idx]:
                    cover_count[s_idx] += 1
            
            # Set penalty: 10 * (1 - 1/cover_count) ? No, simpler:
            # Penalty is high if cover_count is low.
            # We want to avoid creating new critical subsets.
            # Let's just use cover_count directly as a guide.
            
            # Pick a random block to try removing
            if not current_solution:
                break
                
            idx_to_remove = random.randrange(len(current_solution))
            b_removed = current_solution[idx_to_remove]
            
            # Find all subsets that become uncovered (cover_count becomes 0)
            # Actually, we need to find subsets where cover_count[b_removed] was 1.
            # Let's calculate which subsets are now uncovered.
            
            # To be efficient, we can just simulate the removal and check.
            # But we need to find a replacement block that covers ALL newly uncovered subsets.
            
            # Identify critical subsets in the removed block
            critical_subs = []
            for s_idx in subsets_of_block[b_removed]:
                if cover_count[s_idx] == 1:
                    critical_subs.append(s_idx)
            
            if not critical_subs:
                # Block was redundant, remove it
                current_solution.pop(idx_to_remove)
                current_size -= 1
                continue
            
            # Find replacement block that covers all critical_subs
            candidate_blocks = set(blocks_of_subset[critical_subs[0]])
            candidate_blocks.discard(b_removed)
            
            for s_idx in critical_subs[1:]:
                candidate_blocks.intersection_update(blocks_of_subset[s_idx])
                if not candidate_blocks:
                    break
            
            if not candidate_blocks:
                continue
            
            # Score candidates:
            # We want to minimize the number of NEW critical subsets created.
            # Or maximize the number of subsets that become "safe" (cover_count > 1).
            
            best_cand = -1
            best_score = float('inf')
            
            cand_list = list(candidate_blocks)
            random.shuffle(cand_list)
            # Check up to 10 candidates
            for cand_idx in cand_list[:10]:
                # Calculate score
                # Score = number of subsets in cand_idx that would become critical if we add it?
                # No, we are ADDING cand_idx.
                # The subsets in cand_idx will have their cover_count increased by 1.
                # The subsets NOT in cand_idx but in b_removed will have cover_count decreased by 1.
                
                # Wait, we remove b_removed and add cand_idx.
                # Subsets in critical_subs are covered by cand_idx (by construction).
                # Other subsets in b_removed (non-critical) will have cover_count - 1.
                # Subsets in cand_idx (not in b_removed) will have cover_count + 1.
                
                # We want to avoid creating new critical subsets (cover_count becomes 1).
                # A subset s becomes critical if:
                # 1. It is in b_removed but not in cand_idx, and cover_count[s] was 2. (Becomes 1)
                # 2. It is not in b_removed and not in cand_idx? No, unaffected.
                # 3. It is in cand_idx but not in b_removed? Cover_count increases, so cannot become critical (unless it was -1, impossible).
                
                # So we only need to check subsets in b_removed \ cand_idx.
                
                # Also, we want to prefer candidates that cover many subsets (general goodness).
                
                # Let's calculate:
                # new_critical_count = 0
                # for s in subsets_of_block[b_removed]:
                #     if s not in cand_idx_set:
                #         if cover_count[s] == 2:
                #             new_critical_count += 1
                
                # This is O(k*t) per candidate.
                # To speed up, we can precompute a set for cand_idx?
                # subsets_of_block is a list of ints.
                
                # Let's just do a simple heuristic:
                # Score = len(critical_subs) - (number of subsets in cand_idx that are currently cover_count==1)
                # This encourages replacing a block that has many critical subsets with one that covers them
                # and also covers other critical subsets from elsewhere?
                
                # Actually, simpler:
                # Score = sum(1 for s in subsets_of_block[cand_idx] if cover_count[s] == 1)
                # We want to MAXIMIZE this, because covering other critical subsets is good.
                
                score = 0
                for s in subsets_of_block[cand_idx]:
                    if cover_count[s] == 1:
                        score += 1
                
                # Also penalize if cand_idx is the same as b_removed (already discarded)
                
                if score > best_score:
                    best_score = score
                    best_cand = cand_idx
            
            if best_cand != -1:
                # Perform swap
                current_solution[idx_to_remove] = best_cand
                # current_size unchanged
                
                # Optional: Run a quick deletion pass on the new solution
                # to see if it allows size reduction
                cover_count_new = np.zeros(n_t, dtype=np.int32)
                for b_idx in current_solution:
                    for s_idx in subsets_of_block[b_idx]:
                        cover_count_new[s_idx] += 1
                
                # Try to remove redundant blocks
                changed = True
                while changed and time.time() < deadline - 5:
                    changed = False
                    new_sol = []
                    for b_idx in current_solution:
                        can_remove = True
                        for s_idx in subsets_of_block[b_idx]:
                            if cover_count_new[s_idx] == 1:
                                can_remove = False
                                break
                        if can_remove:
                            cover_count_new[subsets_of_block[b_idx]] -= 1
                            changed = True
                        else:
                            new_sol.append(b_idx)
                    current_solution = new_sol
                    current_size = len(current_solution)
                
                if current_size < best_size:
                    best_size = current_size
                    best_solution = current_solution.copy()
    
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
