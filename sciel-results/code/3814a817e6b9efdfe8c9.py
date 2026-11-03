import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin (evaluator kills at 150s, but we want safety)

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # Build data structures for fast access
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    rows = []
    cols = []
    data = []

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            rows.append(i)
            cols.append(j)
            data.append(1.0)

    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
    M_T = M.T  # CSC matrix, efficient for dot product with dense vector

    best_solution = None
    best_size = float('inf')

    # 2. LP Relaxation to get duals (weights)
    duals = None
    try:
        if time.time() > deadline - 20:
            raise TimeoutError()

        c_lp = np.ones(n_k)
        b_ub = -np.ones(n_t)
        bounds_lp = [(0, 1)] * n_k

        res = linprog(c_lp, A_ub=-M_T, b_ub=b_ub, bounds=bounds_lp, method='highs', 
                      options={"presolve": True, "time_limit": 2})

        if res.success:
            duals = res.ineqlin.marginals
            duals = np.abs(duals)
            if np.sum(duals) == 0:
                duals = None
    except Exception:
        duals = None

    if duals is None:
        duals = np.ones(n_t)

    # Pre-compute base weights for blocks based on duals
    base_block_weights = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            base_block_weights[j] = np.sum(duals[subsets_of_block[j]])

    # 3. Multi-start Heap-based Greedy Search with Local Search
    num_restarts = 300
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break

        if restart == 0:
            gains = base_block_weights.copy()
        else:
            noise_scale = np.max(base_block_weights) * 0.05
            gains = base_block_weights + rng.uniform(0, noise_scale, n_k)

        # Heap-based lazy-update greedy
        # Heap stores (-gain, block_idx, version)
        # We track current gain per block and a version counter to detect stale entries
        current_gain = gains.copy()
        version = np.zeros(n_k, dtype=np.int32)
        heap = [(-current_gain[j], j, 0) for j in range(n_k)]
        heapq.heapify(heap)
        
        uncovered_mask = np.ones(n_t, dtype=bool)
        remaining_uncovered = n_t
        solution_blocks = []
        
        while remaining_uncovered > 0:
            if time.time() > deadline - 5:
                break

            # Pop until we find a valid (non-stale) entry
            best_idx = -1
            best_gain_val = 0.0
            while heap:
                neg_g, idx, ver = heapq.heappop(heap)
                if ver == version[idx] and current_gain[idx] > 1e-9:
                    best_idx = idx
                    best_gain_val = current_gain[idx]
                    break
                # Stale entry, discard
            
            if best_idx == -1:
                # No block has positive gain; fall back to scanning
                found = False
                for j in range(n_k):
                    if current_gain[j] > 1e-9:
                        best_idx = j
                        best_gain_val = current_gain[j]
                        found = True
                        break
                if not found:
                    # Try any block that covers at least one uncovered subset
                    for j in range(n_k):
                        if any(uncovered_mask[s] for s in subsets_of_block[j]):
                            best_idx = j
                            found = True
                            break
                    if not found:
                        break
                else:
                    break

            solution_blocks.append(best_idx)
            
            subs_b = subsets_of_block[best_idx]
            newly_covered_indices = [s for s in subs_b if uncovered_mask[s]]
            
            if not newly_covered_indices:
                continue

            uncovered_mask[newly_covered_indices] = False
            remaining_uncovered -= len(newly_covered_indices)
            
            # Update gains: for each newly covered subset, all blocks covering it lose that subset's dual weight
            for s_idx in newly_covered_indices:
                w = duals[s_idx]
                for b_idx in blocks_of_subset[s_idx]:
                    current_gain[b_idx] -= w
                    version[b_idx] += 1
                    heapq.heappush(heap, (-current_gain[b_idx], b_idx, version[b_idx]))

            if remaining_uncovered == 0:
                break

        # Check if solution is valid and better
        if remaining_uncovered == 0:
            # Run local search on this solution
            improved = local_search(solution_blocks, subsets_of_block, blocks_of_subset, n_t, n_k, deadline)
            if len(improved) < best_size:
                best_size = len(improved)
                best_solution = improved.copy()

    # 4. Fallback if no solution found
    if best_solution is None:
        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        while np.any(uncovered_mask):
            best_idx = -1
            max_cov = -1
            for j in range(n_k):
                cnt = np.sum(uncovered_mask[subsets_of_block[j]])
                if cnt > max_cov:
                    max_cov = cnt
                    best_idx = j
            if best_idx == -1:
                break
            solution_blocks.append(best_idx)
            uncovered_mask[subsets_of_block[best_idx]] = False
        best_solution = solution_blocks
        best_size = len(best_solution)

    # 5. Post-processing: Deletion Pass
    if best_solution is not None and time.time() < deadline - 1:
        best_solution = deletion_pass(best_solution, subsets_of_block, n_t, deadline)
        best_size = len(best_solution)

    return [list(k_subsets[idx]) for idx in best_solution]


def local_search(solution, subsets_of_block, blocks_of_subset, n_t, n_k, deadline):
    """1-exchange and 2-exchange local search."""
    if not solution:
        return solution
    
    # Build coverage count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b_idx in solution:
        cover_count[subsets_of_block[b_idx]] += 1
    
    # Set of blocks in solution for O(1) lookup
    in_solution = set(solution)
    
    improved = True
    current_solution = list(solution)
    
    while improved and time.time() < deadline - 3:
        improved = False
        
        # --- 1-exchange pass: try removing one block, optionally adding one ---
        for i in range(len(current_solution)):
            if time.time() > deadline - 3:
                break
            
            b_remove = current_solution[i]
            subs_remove = subsets_of_block[b_remove]
            
            # Check if removing this block leaves any subset uncovered
            can_remove = True
            for s_idx in subs_remove:
                if cover_count[s_idx] == 1:
                    can_remove = False
                    break
            
            if not can_remove:
                continue
            
            # Temporarily remove
            for s_idx in subs_remove:
                cover_count[s_idx] -= 1
            
            # Find best block to add
            best_add = -1
            best_gain = 0
            for j in range(n_k):
                if j in in_solution:
                    continue
                gain = np.sum(cover_count[subsets_of_block[j]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_add = j
            
            if best_add >= 0 and best_gain > 0:
                # Add the replacement
                current_solution[i] = best_add
                in_solution.discard(b_remove)
                in_solution.add(best_add)
                
                for s_idx in subsets_of_block[best_add]:
                    cover_count[s_idx] += 1
                
                improved = True
                break
            else:
                # Restore
                for s_idx in subs_remove:
                    cover_count[s_idx] += 1
        
        # --- 2-exchange pass: try removing two blocks, adding one or two ---
        if time.time() < deadline - 5:
            for i in range(len(current_solution)):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, len(current_solution)):
                    if time.time() > deadline - 3:
                        break
                    
                    b1 = current_solution[i]
                    b2 = current_solution[j]
                    subs1 = subsets_of_block[b1]
                    subs2 = subsets_of_block[b2]
                    
                    # Check if removing both leaves any subset uncovered
                    can_remove = True
                    for s_idx in subs1:
                        if cover_count[s_idx] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        for s_idx in subs2:
                            if cover_count[s_idx] <= 1:
                                can_remove = False
                                break
                    
                    if not can_remove:
                        continue
                    
                    # Temporarily remove both
                    for s_idx in subs1:
                        cover_count[s_idx] -= 1
                    for s_idx in subs2:
                        cover_count[s_idx] -= 1
                    
                    # Try adding one block that covers all newly-uncovered subsets
                    best_add1 = -1
                    best_gain1 = 0
                    for c in range(n_k):
                        if c in in_solution:
                            continue
                        gain = np.sum(cover_count[subsets_of_block[c]] == 0)
                        if gain > best_gain1:
                            best_gain1 = gain
                            best_add1 = c
                    
                    if best_add1 >= 0 and best_gain1 > 0:
                        # Add one replacement
                        current_solution[i] = best_add1
                        current_solution.pop(j)
                        in_solution.discard(b1)
                        in_solution.discard(b2)
                        in_solution.add(best_add1)
                        
                        for s_idx in subsets_of_block[best_add1]:
                            cover_count[s_idx] += 1
                        
                        improved = True
                        break
                    else:
                        # Try adding two blocks
                        best_add2 = -1
                        best_gain2 = 0
                        for c in range(n_k):
                            if c in in_solution:
                                continue
                            gain = np.sum(cover_count[subsets_of_block[c]] == 0)
                            if gain > best_gain2:
                                best_gain2 = gain
                                best_add2 = c
                        
                        if best_add2 >= 0 and best_gain2 > 0:
                            # Temporarily add best_add2
                            for s_idx in subsets_of_block[best_add2]:
                                cover_count[s_idx] += 1
                            
                            best_add3 = -1
                            best_gain3 = 0
                            for c in range(n_k):
                                if c in in_solution or c == best_add2:
                                    continue
                                gain = np.sum(cover_count[subsets_of_block[c]] == 0)
                                if gain > best_gain3:
                                    best_gain3 = gain
                                    best_add3 = c
                            
                            # Restore best_add2
                            for s_idx in subsets_of_block[best_add2]:
                                cover_count[s_idx] -= 1
                            
                            if best_add3 >= 0 and best_gain3 > 0:
                                # Add two replacements
                                current_solution[i] = best_add2
                                current_solution[j] = best_add3
                                in_solution.discard(b1)
                                in_solution.discard(b2)
                                in_solution.add(best_add2)
                                in_solution.add(best_add3)
                                
                                for s_idx in subsets_of_block[best_add2]:
                                    cover_count[s_idx] += 1
                                for s_idx in subsets_of_block[best_add3]:
                                    cover_count[s_idx] += 1
                                
                                improved = True
                                break
                            else:
                                # Restore
                                for s_idx in subs1:
                                    cover_count[s_idx] += 1
                                for s_idx in subs2:
                                    cover_count[s_idx] += 1
                        else:
                            # Restore
                            for s_idx in subs1:
                                cover_count[s_idx] += 1
                            for s_idx in subs2:
                                cover_count[s_idx] += 1
                    
                    if improved:
                        break
                if improved:
                    break
    
    return current_solution


def deletion_pass(solution, subsets_of_block, n_t, deadline):
    """Aggressive deletion pass: repeatedly remove redundant blocks."""
    if not solution:
        return solution
    
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b_idx in solution:
        cover_count[subsets_of_block[b_idx]] += 1
    
    changed = True
    current = list(solution)
    
    while changed and time.time() < deadline - 1:
        changed = False
        new_solution = []
        for b_idx in current:
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
        current = new_solution
    
    # Final verification pass: ensure all subsets are still covered
    cover_count2 = np.zeros(n_t, dtype=np.int32)
    for b_idx in current:
        cover_count2[subsets_of_block[b_idx]] += 1
    
    if np.any(cover_count2 == 0):
        # Something went wrong; restore original
        return solution
    
    return current
