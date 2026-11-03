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
    block_t_arr = [np.array(lst, dtype=np.int32) for lst in block_t_subsets]

    # Symmetry: precompute block as frozenset of points for fast intersection
    block_points = [set(block) for block in k_blocks]

    def run_greedy(weight_mode='inverse', seed=42):
        rng = np.random.default_rng(seed)
        if weight_mode == 'inverse':
            safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
            t_weights = 1.0 / safe_counts
        elif weight_mode == 'random':
            t_weights = 0.3 / np.maximum(t_cover_count, 1).astype(np.float64) + 0.7 * rng.random(num_t)
        elif weight_mode == 'exp':
            # Exponential weighting: rare t-subsets get much higher weight
            safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
            t_weights = np.exp(-safe_counts)
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
        max_time = max_time_frac * 120
        ls_start = time.time()
        def ls_time_left():
            return max_time - (time.time() - ls_start) + (deadline - time.time())
        
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
        for round_num in range(200):
            if time_left() < 12:
                break
            improved = False
            # Shuffle order for diversity
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
                critical_set = set(critical)
                best_candidate = -1
                best_score = -1
                # Score: minimize new unique t-subsets covered, maximize overlap with already-covered
                for b in range(num_k):
                    if b in selected_set:
                        continue
                    b_set = block_t_sets[b]
                    if not critical_set.issubset(b_set):
                        continue
                    # Count t-subsets that would become newly uncovered (cover_count goes from 1 to 0)
                    # and newly covered
                    new_uncovered = 0
                    for idx in b_set:
                        if cover_count[idx] == 0:
                            new_uncovered += 1
                    # Also penalize if we lose coverage of things only this block covers
                    lost = 0
                    for idx in drop_set:
                        if cover_count[idx] == 1 and idx not in b_set:
                            lost += 1
                    score = new_uncovered + 2 * lost
                    if score < best_score or (best_candidate == -1):
                        best_score = score
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

    # Run multiple greedy restarts
    best_selected = None
    best_len = float('inf')

    configs = [
        ('inverse', 42),
        ('random', 123),
        ('random', 777),
        ('exp', 555),
        ('random', 111),
        ('inverse', 999),
        ('random', 2024),
        ('exp', 314),
    ]

    for i, (mode, seed) in enumerate(configs):
        if time_left() < 20:
            break
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
