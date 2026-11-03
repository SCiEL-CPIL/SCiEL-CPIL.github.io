import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 148.0

    def time_left():
        return deadline - (time.time() - start_time)

    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # For each block, compute which t-subsets it covers
    block_t_subsets = []
    for block in k_blocks:
        covered = [t_index[combo] for combo in itertools.combinations(block, t)]
        block_t_subsets.append(covered)

    # Build reverse mapping
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)

    t_cover_count = np.array([len(lst) for lst in t_to_blocks], dtype=np.int32)
    block_t_sets = [set(lst) for lst in block_t_subsets]

    def run_primal_dual(seed=42, max_iters=2000, epsilon=0.01):
        rng = np.random.default_rng(seed)
        
        # Dual prices for each t-subset
        dual = np.zeros(num_t, dtype=np.float64)
        # Cost of each block (all 1)
        cost = np.ones(num_k, dtype=np.float64)
        
        # Track which blocks are selected
        selected_mask = np.zeros(num_k, dtype=np.bool_)
        selected = []
        
        # Uncovered t-subsets
        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        
        # Precompute: for each block, the list of t-subset indices it covers
        # Use numpy arrays for faster operations
        block_cover_arrays = []
        for i, covered in enumerate(block_t_subsets):
            if covered:
                block_cover_arrays.append(np.array(covered, dtype=np.int32))
            else:
                block_cover_arrays.append(np.array([], dtype=np.int32))
        
        # Track dual-weight sum for each block (sum of dual prices of t-subsets it covers)
        block_dual_weight = np.zeros(num_k, dtype=np.float64)
        
        # Initial: compute initial dual weights
        for i, arr in enumerate(block_cover_arrays):
            if len(arr) > 0:
                block_dual_weight[i] = np.sum(dual[arr])
        
        # Precompute for efficiency: for each t-subset, which blocks cover it (as numpy array)
        t_to_blocks_arr = []
        for idx in range(num_t):
            if t_to_blocks[idx]:
                t_to_blocks_arr.append(np.array(t_to_blocks[idx], dtype=np.int32))
            else:
                t_to_blocks_arr.append(np.array([], dtype=np.int32))
        
        for iteration in range(max_iters):
            if time_left() < 25:
                break
            if uncovered_count == 0:
                break
            
            # Step 1: Raise dual prices for uncovered t-subsets
            # Find uncovered t-subsets
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Raise dual prices by epsilon for all uncovered t-subsets
            delta = epsilon
            dual[uncovered_indices] += delta
            
            # Update block dual weights: for each block, add delta for each newly uncovered t-subset it covers
            # More efficient: for each uncovered t-subset, add delta to all blocks covering it
            for t_idx in uncovered_indices:
                blocks_covering = t_to_blocks_arr[t_idx]
                if len(blocks_covering) > 0:
                    block_dual_weight[blocks_covering] += delta
            
            # Step 2: Find blocks with dual-weight >= cost (i.e., dual-weight/cost >= 1)
            # Select all blocks where block_dual_weight[i] >= cost[i] * (1 - small_epsilon)
            # Actually, in primal-dual for set cover, we select when dual-weight >= cost
            # But we want to be more selective: pick the best ratio
            
            # Compute ratio for non-selected blocks
            ratios = np.zeros(num_k, dtype=np.float64)
            non_selected = ~selected_mask
            if np.any(non_selected):
                # For numerical stability, avoid division by zero
                safe_cost = np.where(cost > 0, cost, 1.0)
                ratios[non_selected] = block_dual_weight[non_selected] / safe_cost[non_selected]
                ratios[selected_mask] = -1e10
            
            # Select the block with highest ratio that has ratio >= 1.0
            # Or if none has ratio >= 1.0, select the one with highest ratio
            best_idx = int(np.argmax(ratios))
            best_ratio = ratios[best_idx]
            
            if best_ratio < 1e-10:
                # No progress possible
                break
            
            # Select this block
            selected.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update uncovered status
            covered_arr = block_cover_arrays[best_idx]
            if len(covered_arr) > 0:
                newly_covered = uncovered[covered_arr]
                if np.any(newly_covered):
                    uncovered[covered_arr[newly_covered]] = False
                    uncovered_count -= int(np.sum(newly_covered))
            
            # Reset dual prices for newly covered t-subsets to prevent over-selection
            # Actually, in standard primal-dual, we don't reset. But we can clamp.
            # Let's keep dual prices as is for now.
        
        return selected

    def run_greedy(weight_mode='inverse', seed=42):
        rng = np.random.default_rng(seed)
        if weight_mode == 'inverse':
            safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
            t_weights = 1.0 / safe_counts
        elif weight_mode == 'random':
            t_weights = 0.3 / np.maximum(t_cover_count, 1).astype(np.float64) + 0.7 * rng.random(num_t)
        elif weight_mode == 'exp':
            safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
            t_weights = np.exp(-safe_counts)
        elif weight_mode == 'primal_dual':
            # Use primal-dual approach
            return run_primal_dual(seed=seed)
        else:
            t_weights = 1.0 / np.maximum(t_cover_count, 1).astype(np.float64)

        gain = np.zeros(num_k, dtype=np.float64)
        for i, covered in enumerate(block_t_subsets):
            if covered:
                gain[i] = np.sum(t_weights[covered])

        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)

        while uncovered_count > 0 and time_left() > 30:
            masked_gain = gain.copy()
            masked_gain[selected_mask] = -1e10
            best_idx = int(np.argmax(masked_gain))
            best_gain = gain[best_idx]
            if best_gain <= 0:
                break
            selected.append(best_idx)
            selected_mask[best_idx] = True
            for idx in block_t_subsets[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    w = t_weights[idx]
                    for b in t_to_blocks[idx]:
                        if not selected_mask[b]:
                            gain[b] -= w
        return selected

    def local_search(selected, max_time_frac=0.15):
        if not selected:
            return selected
        
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1
        selected_set = set(selected)

        # Phase 1: Repeated single-drop
        for _ in range(50):
            if time_left() < 15:
                break
            changed = False
            for si in range(len(selected) - 1, -1, -1):
                if time_left() < 15:
                    break
                block_idx = selected[si]
                can_remove = True
                for idx in block_t_subsets[block_idx]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for idx in block_t_subsets[block_idx]:
                        cover_count[idx] -= 1
                    selected.pop(si)
                    selected_set.discard(block_idx)
                    changed = True
            if not changed:
                break

        # Phase 2: 1-for-1 swaps with improved scoring
        max_rounds = 100 
        for round_num in range(max_rounds):
            if time_left() < 12:
                break
            improved = False
            order = list(range(len(selected)))
            random.shuffle(order)
            for si in order:
                if time_left() < 12:
                    break
                if improved:
                    break
                drop_block = selected[si]
                drop_set = block_t_subsets[drop_block]
                critical = [idx for idx in drop_set if cover_count[idx] == 1]
                if not critical:
                    for idx in drop_set:
                        cover_count[idx] -= 1
                    selected.pop(si)
                    selected_set.discard(drop_block)
                    improved = True
                    continue
                
                # Find candidates that cover all critical t-subsets
                candidate_sets = [set(t_to_blocks[idx]) for idx in critical]
                if not candidate_sets:
                    continue
                candidates = candidate_sets[0]
                for cs in candidate_sets[1:]:
                    candidates.intersection_update(cs)
                
                # Filter out already selected blocks
                candidates -= selected_set
                
                if not candidates:
                    continue
                    
                best_candidate = -1
                best_score = float('inf')
                
                for b in candidates:
                    b_set = block_t_sets[b]
                    lost = 0
                    for idx in drop_set:
                        if idx not in b_set and cover_count[idx] == 1:
                            lost += 1
                            if lost > best_score:
                                break
                    if lost > best_score:
                        continue
                            
                    if lost < best_score:
                        best_score = lost
                        best_candidate = b
                        if best_score == 0:
                            break
                
                if best_candidate != -1:
                    for idx in drop_set:
                        cover_count[idx] -= 1
                    for idx in block_t_subsets[best_candidate]:
                        cover_count[idx] += 1
                    selected[si] = best_candidate
                    selected_set.discard(drop_block)
                    selected_set.add(best_candidate)
                    improved = True
        return selected

    # Run multiple strategies
    best_selected = None
    best_len = float('inf')

    configs = [
        ('primal_dual', 42),
        ('primal_dual', 123),
        ('inverse', 42),
        ('random', 123),
        ('random', 777),
        ('exp', 555),
        ('random', 111),
        ('inverse', 999),
        ('primal_dual', 555),
        ('random', 2024),
        ('exp', 314),
    ]

    for i, (mode, seed) in enumerate(configs):
        if time_left() < 20:
            break
        if mode == 'primal_dual':
            sel = run_primal_dual(seed=seed)
        else:
            sel = run_greedy(mode, seed)
        sel = local_search(sel, 0.12)
        if len(sel) < best_len:
            best_len = len(sel)
            best_selected = sel[:]

    # Final polish
    if best_selected and time_left() > 15:
        best_selected = local_search(best_selected, 0.25)

    result = [list(k_blocks[i]) for i in best_selected]

    # Ensure validity
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in best_selected:
            for idx in block_t_subsets[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            selected_set = set(best_selected)
            for i in range(num_k):
                if time_left() < 3:
                    break
                if i in selected_set:
                    continue
                for idx in block_t_subsets[i]:
                    if final_cover[idx] == 0:
                        best_selected.append(i)
                        selected_set.add(i)
                        for idx2 in block_t_subsets[i]:
                            final_cover[idx2] += 1
                        break
            result = [list(k_blocks[i]) for i in best_selected]
    return result
