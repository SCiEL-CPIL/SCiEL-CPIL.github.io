import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin for safety

    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)

    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # 3. Precompute block_t_indices: (n_k, C(k, t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)
    
    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    
    # Filling the array
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]
            
    # 4. Calculate initial Weights for Weighted Greedy
    # Frequency of each t-subset across all k-blocks
    t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
    t_counts[t_counts == 0] = 1
    t_weights = 1.0 / t_counts.astype(np.float64)
    
    # 5. Build sub_to_blocks for local search
    # sub_to_blocks[j] is a list of block indices that cover t-subset j
    sub_to_blocks = [[] for _ in range(n_t)]
    for b_idx in range(n_k):
        for s_idx in block_t_indices[b_idx]:
            sub_to_blocks[s_idx].append(b_idx)
            
    # Helper to perform greedy set cover with specific weights
    def greedy_solve(weights):
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        rng = random.Random(42)
        
        while True:
            if time.time() > deadline - 10.0:
                break
            if not uncovered.any():
                break
                
            # Calculate weighted gains
            gains = (uncovered[block_t_indices] * weights[block_t_indices]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best_idx = rng.choice(candidates)
            selected_blocks.append(best_idx)
            uncovered[block_t_indices[best_idx]] = False
            
        return selected_blocks

    # Initial Greedy
    selected_blocks = greedy_solve(t_weights)
    
    if not selected_blocks:
        return []
        
    best_solution = [k_subsets[i] for i in selected_blocks]
    best_size = len(selected_blocks)

    # Local Search: Swap-based Optimization
    def local_search_improve(selected_indices, deadline, rng):
        # selected_indices is a list of block indices
        if not selected_indices:
            return [], 0
            
        # Build coverage counts
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for bi in selected_indices:
            coverage_count[block_t_indices[bi]] += 1
            
        current_size = len(selected_indices)
        improved = True
        no_improve_streak = 0
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Phase 1: Remove redundant blocks
            i = 0
            while i < len(selected_indices):
                if time.time() > deadline - 2.0:
                    return selected_indices, len(selected_indices)
                    
                bi = selected_indices[i]
                ts = block_t_indices[bi]
                # Check if all covered subsets have count > 1
                if np.all(coverage_count[ts] > 1):
                    # Remove block
                    coverage_count[ts] -= 1
                    selected_indices.pop(i)
                    current_size -= 1
                    improved = True
                else:
                    i += 1
                    
            # Phase 2: Try swaps
            if time.time() > deadline - 5.0:
                break
                
            for i in range(len(selected_indices)):
                if time.time() > deadline - 2.0:
                    break
                    
                bi = selected_indices[i]
                ts_b = block_t_indices[bi]
                
                # Find critical subsets: those with count == 1
                critical_mask = (coverage_count[ts_b] == 1)
                critical_subsets = ts_b[critical_mask]
                
                if len(critical_subsets) == 0:
                    # This block is redundant
                    coverage_count[ts_b] -= 1
                    selected_indices.pop(i)
                    current_size -= 1
                    improved = True
                    break
                    
                # If too many critical subsets, finding a replacement is hard.
                if len(critical_subsets) > 4:
                    continue
                    
                # Find candidate blocks covering ALL critical subsets
                cands_lists = [sub_to_blocks[s] for s in critical_subsets]
                cands_lists.sort(key=len)
                
                if not cands_lists:
                    continue
                
                # Start with the smallest list
                candidates = set(cands_lists[0])
                
                for lst in cands_lists[1:]:
                    if not candidates:
                        break
                    if len(candidates) < len(lst):
                        candidates = candidates.intersection(lst)
                    else:
                        candidates = set(lst).intersection(candidates)
                        
                if not candidates:
                    continue
                
                candidates.discard(bi)
                
                if not candidates:
                    continue
                
                cand_list = list(candidates)
                
                if len(cand_list) > 50:
                    cand_list = rng.sample(cand_list, 50)
                
                best_cand = None
                best_score = -1
                
                # Heuristic: Pick the one that minimizes disruption (max overlap) or random
                # To escape local minima, random is often better, but let's try overlap first
                # Actually, random choice from a small set is fast and effective.
                best_cand = rng.choice(cand_list)
                
                if best_cand is not None:
                    # Perform swap
                    coverage_count[ts_b] -= 1
                    selected_indices[i] = best_cand
                    
                    ts_bc = block_t_indices[best_cand]
                    coverage_count[ts_bc] += 1
                    
                    improved = True
                    break
                    
        return selected_indices, len(selected_indices)

    rng = random.Random(123)
    
    # Run local search
    current_indices = list(selected_blocks)
    current_indices, current_size = local_search_improve(current_indices, deadline, rng)
    
    if current_size < best_size:
        best_size = current_size
        best_solution = [k_subsets[i] for i in current_indices]
        
    # Lagrangian Relaxation Phase
    # We update weights based on "scarcity" of coverage in the current solution.
    # High weight -> subset is hard to cover (or currently under-covered in terms of redundancy potential).
    # Actually, for Lagrangian relaxation of Set Cover:
    # Minimize sum(x_j) subject to sum_{j in S_i} x_j >= 1.
    # Lagrangian L(x, u) = sum x_j + sum_i u_i (1 - sum_{j in S_i} x_j)
    # = sum x_j (1 - sum_{i in T_j} u_i) + sum u_i
    # To minimize L, we pick blocks with negative reduced cost: c'_j = 1 - sum_{i in T_j} u_i < 0.
    # Or simply, we use u_i to weight the "gain" in greedy.
    # If a subset is currently covered only once (critical), increasing its weight u_i makes blocks covering it more "valuable" (higher gain) in the next greedy pass?
    # Wait, standard greedy picks max gain. Gain = sum_{i uncovered} w_i.
    # If we increase w_i for critical subsets, blocks covering them get higher priority.
    
    if time.time() < deadline - 20.0:
        # Iterated Lagrangian Greedy
        # Start with weights based on current coverage counts
        # u_i is proportional to 1 / count_i ? Or just 1 if count_i == 1?
        # Let's try: u_i = 1.0 if count_i == 1, else 0.1.
        # This emphasizes covering the "weak points".
        
        # We will do a few iterations of:
        # 1. Compute weights from current best solution's coverage counts.
        # 2. Run Greedy with these weights.
        # 3. Polish with Local Search.
        # 4. Keep best.
        
        # We also incorporate "Subgradient" like updates:
        # If a subset is uncovered in the greedy solution (shouldn't happen if greedy finishes), increase weight.
        # If a subset is over-covered, decrease weight?
        # Since greedy always covers everything, we look at redundancy.
        # Subsets with count 1 are "tight". Subsets with count > 1 are "slack".
        # We want to encourage blocks that cover tight subsets.
        
        for iter_lagr in range(5):
            if time.time() > deadline - 10.0:
                break
                
            # Compute weights based on current best_solution
            coverage_count_best = np.zeros(n_t, dtype=np.int32)
            for bi in current_indices:
                coverage_count_best[block_t_indices[bi]] += 1
            
            # Weight: High for critical (count=1), Low for redundant (count>1)
            # Normalize so that average weight is 1?
            # w_i = 1.0 / max(1, count_i)
            lag_weights = 1.0 / np.maximum(1, coverage_count_best).astype(np.float64)
            
            # Run Greedy with these weights
            lag_selected = greedy_solve(lag_weights)
            
            # Polish
            lag_selected, lag_size = local_search_improve(lag_selected, deadline - 5.0, rng)
            
            if lag_size < best_size:
                best_size = lag_size
                best_solution = [k_subsets[i] for i in lag_selected]
                current_indices = lag_selected
                
            # If no improvement, we might break or continue with different weights
            # Let's add a small perturbation to weights to escape local optima
            if time.time() > deadline - 15.0:
                break

    # Iterated Local Search (ILS) if time remains
    if time.time() < deadline - 30.0:
        perturbation_strength = 0.05
        max_iterations = 10
        
        for _ in range(max_iterations):
            if time.time() > deadline - 10.0:
                break
                
            n_perturb = max(1, int(len(current_indices) * perturbation_strength))
            if n_perturb >= len(current_indices):
                n_perturb = max(1, len(current_indices) // 2)
                
            remove_indices = set(rng.sample(range(len(current_indices)), n_perturb))
            perturbed_indices = [idx for i, idx in enumerate(current_indices) if i not in remove_indices]
            
            # Repair
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in perturbed_indices:
                coverage_count[block_t_indices[bi]] += 1
            uncovered = coverage_count == 0
            
            # Use current weights for repair? Or default?
            # Let's use weights derived from the *perturbed* state to guide repair
            # or just use the static t_weights (inverse frequency) which is a good heuristic.
            repair_weights = t_weights
            
            # Simple greedy repair
            while uncovered.any() and time.time() < deadline - 5.0:
                gains = (uncovered[block_t_indices] * repair_weights[block_t_indices]).sum(axis=1)
                
                selected_set = set(perturbed_indices)
                
                # Find best candidate
                best_gain = -1
                best_idx = -1
                
                # To speed up, sample top candidates
                # Get indices with gain > 0
                candidate_indices = np.where(gains > 0)[0]
                
                if len(candidate_indices) == 0:
                    break
                    
                # Top 50
                if len(candidate_indices) > 50:
                    top_idx = np.argsort(gains[candidate_indices])[::-1][:50]
                    top_candidates = candidate_indices[top_idx]
                else:
                    top_candidates = candidate_indices
                    
                for idx in top_candidates:
                    if idx not in selected_set:
                        if gains[idx] > best_gain:
                            best_gain = gains[idx]
                            best_idx = idx
                            
                if best_idx == -1:
                    # Fallback
                    for idx in candidate_indices:
                        if idx not in selected_set:
                            if gains[idx] > best_gain:
                                best_gain = gains[idx]
                                best_idx = idx
                    if best_idx == -1:
                        break
                        
                perturbed_indices.append(best_idx)
                selected_set.add(best_idx)
                coverage_count[block_t_indices[best_idx]] += 1
                uncovered[block_t_indices[best_idx]] = False
                
            # Polish
            if time.time() < deadline - 5.0:
                polished_indices, polished_size = local_search_improve(perturbed_indices, deadline - 5.0, rng)
                
                if polished_size < best_size:
                    best_size = polished_size
                    best_solution = [k_subsets[i] for i in polished_indices]
                    current_indices = polished_indices

    if best_solution is None:
        return []
        
    return [list(b) for b in best_solution]
