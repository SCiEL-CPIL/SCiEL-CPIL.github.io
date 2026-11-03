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

    best_solution = None
    best_size = float('inf')

    # 2. LP Relaxation to get duals (weights)
    # Use the dual problem: max sum(y_s) s.t. sum_{s in B} y_s <= 1 for all blocks B, y >= 0
    # This gives us weights that prioritize covering "hard" subsets
    duals = None
    try:
        if time.time() > deadline - 25:
            raise TimeoutError()

        # Solve the dual LP: maximize sum(y) subject to M @ y <= 1, y >= 0
        c_dual = -np.ones(n_t)  # minimize -sum(y) = maximize sum(y)
        A_eq = None
        b_eq = None
        A_ub_lp = M.tocsr()  # M @ y <= 1
        b_ub_lp = np.ones(n_k)
        bounds_dual = [(0, None)] * n_t

        res = linprog(c_dual, A_ub=A_ub_lp, b_ub=b_ub_lp, bounds=bounds_dual, method='highs',
                      options={"presolve": True, "time_limit": 25})

        if res.success:
            duals = np.abs(res.x)
            # Normalize to have reasonable scale
            if np.sum(duals) > 0:
                duals = duals / np.max(duals)
            else:
                duals = None
    except Exception:
        duals = None

    if duals is None:
        # Fallback: use frequency-based weights (rarer subsets get higher weight)
        duals = np.ones(n_t)
        # Weight by inverse frequency of containing blocks
        freq = np.array([len(blocks_of_subset[i]) for i in range(n_t)], dtype=float)
        duals = 1.0 / (freq + 1.0)
        duals = duals / np.max(duals)

    # Pre-compute base weights for blocks based on duals
    base_block_weights = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            base_block_weights[j] = np.sum(duals[subsets_of_block[j]])

    # 3. Multi-start Lazy Greedy Search
    # Uses a priority queue with cached gains and submodular lazy evaluation.
    # When a block is popped, we recompute its actual gain only if the cached
    # gain might be stale (i.e., some of its subsets were covered since it was last updated).
    
    # Adaptive number of restarts based on remaining time
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 10  # Reserve 10s for post-processing
    if remaining < 10:
        num_restarts = 1
    else:
        num_restarts = min(500, max(50, int(remaining * 3)))
    
    rng = np.random.default_rng(42)

    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break

        if restart == 0:
            init_weights = base_block_weights.copy()
        elif restart < num_restarts // 2:
            # Early restarts: moderate noise
            noise_scale = np.max(base_block_weights) * 0.08
            init_weights = base_block_weights + rng.uniform(0, noise_scale, n_k)
        else:
            # Later restarts: more aggressive exploration
            noise_scale = np.max(base_block_weights) * 0.15
            init_weights = base_block_weights * rng.uniform(0.5, 1.5, n_k)

        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        
        # Lazy greedy with priority queue
        # Heap entries: (-cached_gain, block_idx, version)
        # version tracks how many times we've updated this block's gain
        versions = np.zeros(n_k, dtype=np.int32)
        cached_gains = init_weights.copy()
        
        # Build initial heap
        heap = []
        for j in range(n_k):
            if cached_gains[j] > 1e-9:
                heapq.heappush(heap, (-cached_gains[j], j, 0))
        
        while heap and time.time() < deadline - 5:
            neg_gain, best_idx, ver = heapq.heappop(heap)
            
            if ver != versions[best_idx]:
                # Stale entry, skip
                continue
            
            # Recompute actual gain for this block
            subs_b = subsets_of_block[best_idx]
            actual_gain = 0.0
            for s in subs_b:
                if uncovered_mask[s]:
                    actual_gain += duals[s]
            
            if actual_gain <= 1e-9:
                # No benefit, skip this block
                continue
            
            # Accept this block
            solution_blocks.append(best_idx)
            
            # Update uncovered_mask
            newly_covered = []
            for s in subs_b:
                if uncovered_mask[s]:
                    newly_covered.append(s)
                    uncovered_mask[s] = False
            
            if not newly_covered:
                continue
            
            # Update cached gains for all blocks that share newly covered subsets
            # For each newly covered subset s, find all blocks containing s
            # and subtract duals[s] from their cached gain
            for s in newly_covered:
                w = duals[s]
                for j2 in blocks_of_subset[s]:
                    if j2 == best_idx:
                        continue
                    cached_gains[j2] -= w
                    versions[j2] += 1
                    # Push updated entry
                    if cached_gains[j2] > 1e-9:
                        heapq.heappush(heap, (-cached_gains[j2], j2, versions[j2]))
            
            # Check if all covered
            if not np.any(uncovered_mask):
                break
            
            # Early termination: if we already have a solution, stop expanding
            if len(solution_blocks) >= best_size and best_size < float('inf'):
                break

        # Check if solution is valid and better
        if len(solution_blocks) < best_size:
            if not np.any(uncovered_mask):
                best_size = len(solution_blocks)
                best_solution = solution_blocks.copy()

    # 4. Fallback if no solution found
    if best_solution is None:
        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        while np.any(uncovered_mask) and time.time() < deadline - 1:
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

    # 5. Post-processing: Vectorized Deletion Pass with multiple passes
    if best_solution is not None and time.time() < deadline - 5:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block[b_idx]] += 1

        max_deletion_passes = 10
        for _ in range(max_deletion_passes):
            if time.time() > deadline - 5:
                break
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block[b_idx]
                # Check if any subset is uniquely covered by this block
                if not np.any(cover_count[subs_b] == 1):
                    # Can remove
                    cover_count[subs_b] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)

            best_solution = new_solution
            best_size = len(best_solution)
            if not changed:
                break

    # 6. Local Search: Try swapping blocks to improve coverage
    # Enhanced with 1-opt and 2-opt moves
    if best_solution is not None and time.time() < deadline - 3:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block[b_idx]] += 1
        
        improved = True
        max_local_iter = 50
        local_iter = 0
        while improved and time.time() < deadline - 3 and local_iter < max_local_iter:
            local_iter += 1
            improved = False
            new_solution = list(best_solution)
            
            for i in range(len(new_solution)):
                if time.time() > deadline - 3:
                    break
                
                b_old = new_solution[i]
                subs_old = subsets_of_block[b_old]
                
                # Find subsets uniquely covered by b_old
                unique_subs = [s for s in subs_old if cover_count[s] == 1]
                
                if len(unique_subs) == 0:
                    # Can just remove b_old
                    cover_count[subs_old] -= 1
                    new_solution.pop(i)
                    improved = True
                    break
                else:
                    # Try 1-opt: replace with a single block
                    if len(unique_subs) <= 4:
                        candidates = set(blocks_of_subset[unique_subs[0]])
                        for s in unique_subs[1:]:
                            candidates &= set(blocks_of_subset[s])
                        candidates.discard(b_old)
                        
                        best_candidate = None
                        best_score = -1
                        for cand in candidates:
                            subs_cand = subsets_of_block[cand]
                            # Score: number of subsets that would become more covered
                            score = np.sum(cover_count[subs_cand] <= 1)
                            # Bonus for blocks that cover many unique subsets
                            score += 0.1 * sum(1 for s in unique_subs if s in set(subs_cand))
                            if score > best_score:
                                best_score = score
                                best_candidate = cand
                        
                        if best_candidate is not None and best_score > len(unique_subs):
                            cover_count[subs_old] -= 1
                            subs_new = subsets_of_block[best_candidate]
                            cover_count[subs_new] += 1
                            new_solution[i] = best_candidate
                            improved = True
                            break
                    else:
                        # Try 2-opt: remove b_old and add two new blocks
                        # Find two blocks that together cover all unique_subs
                        if len(unique_subs) <= 6:
                            best_pair = None
                            best_pair_score = -1
                            
                            # Get candidate blocks for first unique subset
                            cand_first = list(blocks_of_subset[unique_subs[0]])
                            if len(cand_first) > 20:
                                cand_first = cand_first[:20]
                            
                            for b1 in cand_first:
                                if b1 == b_old:
                                    continue
                                subs_b1 = set(subsets_of_block[b1])
                                # Check which unique subsets b1 covers
                                remaining = [s for s in unique_subs if s not in subs_b1]
                                if not remaining:
                                    continue
                                
                                # Find blocks covering all remaining
                                cand_second = set(blocks_of_subset[remaining[0]])
                                for s in remaining[1:]:
                                    cand_second &= set(blocks_of_subset[s])
                                cand_second.discard(b_old)
                                cand_second.discard(b1)
                                
                                if not cand_second:
                                    continue
                                
                                for b2 in list(cand_second)[:10]:
                                    subs_b2 = set(subsets_of_block[b2])
                                    # Check if b1 and b2 together cover all unique_subs
                                    if all(s in subs_b1 or s in subs_b2 for s in unique_subs):
                                        # Score the pair
                                        score = 0
                                        for s in subs_b1:
                                            if cover_count[s] <= 1:
                                                score += 1
                                        for s in subs_b2:
                                            if cover_count[s] <= 1 and s not in subs_b1:
                                                score += 1
                                        if score > best_pair_score:
                                            best_pair_score = score
                                            best_pair = (b1, b2)
                            
                            if best_pair is not None and best_pair_score > len(unique_subs):
                                b1, b2 = best_pair
                                cover_count[subs_old] -= 1
                                cover_count[subsets_of_block[b1]] += 1
                                cover_count[subsets_of_block[b2]] += 1
                                new_solution[i] = b1
                                new_solution.append(b2)
                                improved = True
                                break
            
            if improved:
                best_solution = new_solution
                best_size = len(best_solution)

    return [list(k_subsets[idx]) for idx in best_solution]
