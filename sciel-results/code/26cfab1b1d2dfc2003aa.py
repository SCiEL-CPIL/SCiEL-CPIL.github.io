import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # Helper function for combinations count
    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structures
    # block_to_ts: list of lists, block_to_ts[j] contains indices of t-subsets in block j
    # ts_to_blocks: list of lists, ts_to_blocks[i] contains indices of blocks containing t-subset i
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
        
    # Map from block tuple to index for fast lookup
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def check_deadline():
        return time.time() > deadline - 5.0

    # --- Strategy: Lagrangian Relaxation / Subgradient Optimization ---
    # We maintain weights for each t-subset. Initially, weights are uniform.
    # In each iteration, we select blocks greedily based on "efficiency":
    # efficiency = sum(weights of uncovered t-subsets in block) / cost_of_block
    # Here cost is 1 for each block.
    # After constructing a solution, we update weights:
    # w_i += step * (1 - sum_{j in S} x_j * A_ij)
    # where A_ij is 1 if block j covers t-subset i, x_j is 1 if block j is in solution.
    # This increases weights of uncovered subsets and decreases weights of over-covered ones.
    
    def run_lagrangian_greedy(max_iters=50, step_size=None):
        if step_size is None:
            # Heuristic step size
            step_size = 0.5 / (num_t_subsets ** 0.5)
            
        # Initial weights: all 1.0
        weights = np.ones(num_t_subsets, dtype=np.float64)
        
        best_sol = None
        best_len = float('inf')
        
        for iter_num in range(max_iters):
            if check_deadline():
                break
                
            # 1. Greedy Construction based on current weights
            is_uncovered = np.full(num_t_subsets, True, dtype=bool)
            is_selected = np.full(num_k_blocks, False, dtype=bool)
            sol = []
            
            # Precompute gains for all blocks? No, too slow.
            # Instead, we can compute gain on the fly or use a priority queue.
            # Given num_k_blocks ~ 75582 and num_t_subsets ~ 11628, 
            # a simple loop might be okay if we are careful, but let's try a vectorized approach for gain calculation.
            
            # Vectorized Gain Calculation:
            # gain[j] = sum(weights[i] for i in block_to_ts[j] if is_uncovered[i])
            
            # We need to update gains efficiently.
            # Initial gains:
            # Since is_uncovered is all True, gain[j] = sum(weights[i] for i in block_to_ts[j])
            # We can precompute this once per iteration? 
            # weights change every iteration, so we recompute.
            
            # To speed up, we can use sparse matrix multiplication if we had one, 
            # but with numpy arrays it's a bit tricky without scipy.sparse.
            # Let's stick to a loop for gain calculation but optimize the inner loop.
            
            # Actually, let's just compute gains for all blocks at the start of each greedy step?
            # That's O(N_blocks * K_t) which is ~ 75k * 56 = 4M ops. 
            # 50 iterations * 4M = 200M ops. Might be slow in Python.
            
            # Alternative: Update gains incrementally.
            # When we select a block, we mark its t-subsets as covered.
            # For every other block b, if it shares any of those t-subsets, its gain decreases by weights[shared_ts].
            
            # Let's use the incremental update approach.
            
            # Initialize gains
            gains = np.zeros(num_k_blocks, dtype=np.float64)
            for j in range(num_k_blocks):
                # Sum weights of t-subsets in block j
                # block_to_ts[j] is a list of indices
                gains[j] = np.sum(weights[block_to_ts[j]])
            
            while True:
                if check_deadline():
                    return best_sol, best_len
                
                max_gain = np.max(gains)
                if max_gain <= 1e-9:
                    break
                    
                # Find all blocks with max gain (or close to it)
                # To avoid ties causing bias, pick randomly among top performers
                # Or just pick argmax. Let's pick argmax for determinism in this phase, 
                # but maybe randomize later.
                
                # Optimization: Only consider blocks with gain > 0
                candidates = np.where(gains == max_gain)[0]
                if len(candidates) == 0:
                    break
                    
                b_idx = candidates[0] # Deterministic pick for now
                
                sol.append(list(all_k_blocks[b_idx]))
                is_selected[b_idx] = True
                gains[b_idx] = -1 # Mark as selected
                
                # Update gains for other blocks
                # For each t-subset ts in b_idx:
                #   For each block b' containing ts:
                #     if not selected:
                #       gains[b'] -= weights[ts]
                
                for ts_idx in block_to_ts[b_idx]:
                    w = weights[ts_idx]
                    if w == 0:
                        continue
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gains[nb_idx] -= w
                            
            if not sol:
                break
                
            # Check if this solution is better
            if len(sol) < best_len:
                best_len = len(sol)
                best_sol = sol
                
            # 2. Update Weights (Subgradient Step)
            # Compute coverage vector
            cover_count = np.zeros(num_t_subsets, dtype=np.int32)
            for b in sol:
                idx = get_block_idx(b)
                if idx >= 0:
                    for ts_idx in block_to_ts[idx]:
                        cover_count[ts_idx] += 1
                        
            # Subgradient: g_i = 1 - cover_count[i]
            # If cover_count[i] == 0, g_i = 1 (increase weight)
            # If cover_count[i] >= 1, g_i <= 0 (decrease weight or keep)
            # Standard subgradient for set cover: 
            # w_i <- max(0, w_i + step * (1 - sum_{j in S} A_ij))
            
            # Calculate sum of selected blocks covering each ts
            # We already have cover_count.
            
            # Adaptive step size (Polyak-like or diminishing)
            # step = alpha / (iter_num + 1)
            current_step = step_size / (1 + iter_num * 0.1)
            
            # Update weights
            # w_i += current_step * (1 - cover_count[i])
            # Clip to 0
            updates = 1.0 - cover_count.astype(np.float64)
            weights += current_step * updates
            weights = np.maximum(weights, 0.0)
            
            # Optional: Normalize weights to prevent drift
            if np.max(weights) > 1000:
                weights /= 1000
                
        return best_sol, best_len

    # Run Lagrangian Greedy
    lag_sol, lag_len = run_lagrangian_greedy(max_iters=100)
    
    # If Lagrangian didn't find anything (shouldn't happen), fall back to standard greedy
    if lag_sol is None:
        # Standard Greedy
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        while True:
            if check_deadline():
                break
            max_idx = np.argmax(gain)
            if gain[max_idx] <= 0:
                break
            sol.append(list(all_k_blocks[max_idx]))
            is_selected[max_idx] = True
            gain[max_idx] = -1
            for ts_idx in block_to_ts[max_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for b_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[b_idx]:
                            gain[b_idx] -= 1
        lag_sol = sol
        lag_len = len(sol)

    best_solution = lag_sol
    best_len = lag_len

    # Local Search: Remove redundant blocks
    def optimize_redundancy(blocks):
        if not blocks:
            return blocks
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
            else:
                for ts in itertools.combinations(b, t):
                    cover_count[ts_index_map[ts]] += 1
                    
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if check_deadline():
                    return blocks
                
                idx = get_block_idx(blocks[i])
                if idx >= 0:
                    redundant = True
                    for ts_idx in block_to_ts[idx]:
                        if cover_count[ts_idx] == 1:
                            redundant = False
                            break
                else:
                    redundant = True
                    for ts in itertools.combinations(blocks[i], t):
                        if cover_count[ts_index_map[ts]] == 1:
                            redundant = False
                            break
                        
                if redundant:
                    block_to_remove = blocks[i]
                    blocks.pop(i)
                    idx = get_block_idx(block_to_remove)
                    if idx >= 0:
                        for ts_idx in block_to_ts[idx]:
                            cover_count[ts_idx] -= 1
                    else:
                        for ts in itertools.combinations(block_to_remove, t):
                            cover_count[ts_index_map[ts]] -= 1
                    changed = True
                    break
                    
        return blocks

    best_solution = optimize_redundancy(best_solution)
    
    # Local Search: 1-move swap optimization
    def local_search_swap(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
            else:
                for ts in itertools.combinations(b, t):
                    cover_count[ts_index_map[ts]] += 1
                    
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        for i in range(len(blocks)):
            if check_deadline():
                return blocks, improved
            
            idx = get_block_idx(blocks[i])
            if idx < 0:
                continue
                
            unique_ts = []
            for ts_idx in block_to_ts[idx]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
                    
            if not unique_ts:
                continue
            
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            candidates.discard(idx)
            
            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            
            if not valid_candidates:
                continue
                
            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])
            
            old_block = blocks[i]
            blocks[i] = new_block
            
            for ts_idx in block_to_ts[idx]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_block_idx]:
                cover_count[ts_idx] += 1
            
            sol_block_indices.discard(idx)
            sol_block_indices.add(new_block_idx)
            
            improved = True
            
        return blocks, improved

    # Run local search for a few iterations
    for _ in range(50):
        if check_deadline():
            break
        best_solution, improved = local_search_swap(best_solution)
        if not improved:
            break
        best_solution = optimize_redundancy(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # 2-Exchange Local Search
    def local_search_2exchange(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
            else:
                for ts in itertools.combinations(b, t):
                    cover_count[ts_index_map[ts]] += 1
                    
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        
        for i in range(len(blocks)):
            if check_deadline():
                return blocks, improved
            
            idx_i = get_block_idx(blocks[i])
            if idx_i < 0:
                continue
                
            for j in range(i + 1, len(blocks)):
                if check_deadline():
                    return blocks, improved
                    
                idx_j = get_block_idx(blocks[j])
                if idx_j < 0:
                    continue
                    
                unique_ts = set()
                for ts_idx in block_to_ts[idx_i]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.add(ts_idx)
                for ts_idx in block_to_ts[idx_j]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.add(ts_idx)
                        
                if not unique_ts:
                    continue
                    
                candidates = set(ts_to_blocks[list(unique_ts)[0]])
                for ts_idx in list(unique_ts)[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                        
                if not candidates:
                    continue
                    
                candidates.discard(idx_i)
                candidates.discard(idx_j)
                
                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                
                if not valid_candidates:
                    continue
                    
                new_block_idx = random.choice(valid_candidates)
                new_block = list(all_k_blocks[new_block_idx])
                
                # Check validity
                valid = True
                # Temporarily remove i and j
                for ts_idx in block_to_ts[idx_i]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[idx_j]:
                    cover_count[ts_idx] -= 1
                # Add new
                for ts_idx in block_to_ts[new_block_idx]:
                    cover_count[ts_idx] += 1
                    
                # Verify
                if np.any(cover_count < 1):
                    valid = False
                else:
                    valid = True
                
                if valid:
                    blocks.pop(j)
                    blocks.pop(i)
                    blocks.append(new_block)
                    
                    sol_block_indices.discard(idx_i)
                    sol_block_indices.discard(idx_j)
                    sol_block_indices.add(new_block_idx)
                    
                    improved = True
                    return blocks, improved 
                else:
                    # Rollback
                    for ts_idx in block_to_ts[new_block_idx]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[idx_i]:
                        cover_count[ts_idx] += 1
                    for ts_idx in block_to_ts[idx_j]:
                        cover_count[ts_idx] += 1
                    
        return blocks, improved

    for _ in range(20):
        if check_deadline():
            break
        best_solution, improved = local_search_2exchange(best_solution)
        if not improved:
            break
        best_solution = optimize_redundancy(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    best_solution = optimize_redundancy(best_solution)

    return best_solution

