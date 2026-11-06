import numpy as np
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: 150s limit. We stop searching at 145s to ensure return.
    deadline = start_time + 145.0

    N = 256

    # Precompute all 4-term arithmetic progressions
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)

    # For each position, store the other 3 positions in each AP containing it
    aps_at = [[] for _ in range(N)]
    for idx in range(num_aps):
        for pos in aps[idx]:
            aps_at[pos].append(idx)

    others_for_pos = [None] * N
    for i in range(N):
        if not aps_at[i]:
            others_for_pos[i] = np.array([], dtype=np.int32).reshape(0, 3)
        else:
            others_list = []
            for ap_idx in aps_at[i]:
                row = aps[ap_idx]
                others = [p for p in row if p != i]
                others_list.append(others)
            others_for_pos[i] = np.array(others_list, dtype=np.int32)

    def count_mono(col):
        c = col[aps]
        return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))

    def fast_delta_single(col, i):
        others = others_for_pos[i]
        if others.shape[0] == 0:
            return 0
        others_colors = col[others]
        all_same = (others_colors[:, 0] == others_colors[:, 1]) & (others_colors[:, 1] == others_colors[:, 2])
        if not np.any(all_same):
            return 0
        c_other = others_colors[:, 0]
        c_i = col[i]
        mono_before = c_other == c_i
        mono_after = c_other == (1 - c_i)
        count_before = np.sum(mono_before & all_same)
        count_after = np.sum(mono_after & all_same)
        return int(count_after - count_before)

    def fast_delta_pair(col, i, j):
        """Compute change in mono count if we flip positions i and j simultaneously."""
        # Collect all APs affected by flipping i or j
        affected = set()
        for ap_idx in aps_at[i]:
            affected.add(ap_idx)
        for ap_idx in aps_at[j]:
            affected.add(ap_idx)
        
        if not affected:
            return 0
        
        # Count mono before and after for affected APs
        before = 0
        after = 0
        col_copy = col.copy()
        col_copy[i] = 1 - col_copy[i]
        col_copy[j] = 1 - col_copy[j]
        
        for ap_idx in affected:
            row = aps[ap_idx]
            c_before = col[row]
            c_after = col_copy[row]
            if (c_before[0] == c_before[1]) and (c_before[1] == c_before[2]) and (c_before[2] == c_before[3]):
                before += 1
            if (c_after[0] == c_after[1]) and (c_after[1] == c_after[2]) and (c_after[2] == c_after[3]):
                after += 1
        
        return after - before

    def sa_run(seed, max_steps, start_temp, end_temp):
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        col = np_rng.randint(0, 2, size=N).astype(np.int32)
        current_score = count_mono(col)
        best_col = col.copy()
        best_score = current_score

        # Precompute list of positions for fast access
        positions = list(range(N))

        for step in range(max_steps):
            if time.time() > deadline - 3.0:
                break

            t = start_temp * (end_temp / start_temp) ** (step / max_steps)

            # With some probability, do a pair flip; otherwise single flip
            if rng.random() < 0.3:
                i = rng.randint(0, N - 1)
                j = rng.randint(0, N - 1)
                if i == j:
                    j = (j + 1) % N
                delta = fast_delta_pair(col, i, j)
                if delta <= 0:
                    col[i] = 1 - col[i]
                    col[j] = 1 - col[j]
                    current_score += delta
                    if current_score < best_score:
                        best_score = current_score
                        best_col = col.copy()
                else:
                    if t > 0 and rng.random() < np.exp(-delta / t):
                        col[i] = 1 - col[i]
                        col[j] = 1 - col[j]
                        current_score += delta
            else:
                i = rng.randint(0, N - 1)
                delta = fast_delta_single(col, i)
                if delta <= 0:
                    col[i] = 1 - col[i]
                    current_score += delta
                    if current_score < best_score:
                        best_score = current_score
                        best_col = col.copy()
                else:
                    if t > 0 and rng.random() < np.exp(-delta / t):
                        col[i] = 1 - col[i]
                        current_score += delta

        return best_col, best_score

    def local_search_single(col, current_score, max_passes=20):
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 5.0:
            improved = False
            passes += 1
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                delta = fast_delta_single(col, i)
                if delta < 0:
                    col[i] = 1 - col[i]
                    current_score += delta
                    improved = True
        return col, current_score

    def local_search_pair(col, current_score, max_passes=5):
        """Local search using pair flips."""
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 5.0:
            improved = False
            passes += 1
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                for j in order:
                    if j <= i:
                        continue
                    delta = fast_delta_pair(col, i, j)
                    if delta < 0:
                        col[i] = 1 - col[i]
                        col[j] = 1 - col[j]
                        current_score += delta
                        improved = True
        return col, current_score

    # Multiple restarts
    best_overall = None
    best_overall_score = float('inf')

    # 1. Try structured colorings
    structured_patterns = [
        np.array([(i // 2) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i // 4) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i // 8) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i // 16) % 2 for i in range(N)], dtype=np.int32),
        np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32),
        np.array([(i % 3) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i % 5) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i % 7) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i * 3 + 1) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i * 5 + 2) % 2 for i in range(N)], dtype=np.int32),
    ]

    for pat in structured_patterns:
        s = count_mono(pat)
        if s < best_overall_score:
            best_overall_score = s
            best_overall = pat.copy()

    # 2. SA runs with more restarts and longer runs
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
            break

        seed = 100 + r * 137
        max_steps = 60000
        start_temp = 3.0
        end_temp = 0.005

        best_col, best_score = sa_run(seed, max_steps, start_temp, end_temp)

        # Polish with single-flip local search
        best_col, best_score = local_search_single(best_col, best_score, max_passes=15)

        # Try pair-flip local search for further improvement
        if time.time() < deadline - 10.0:
            best_col, best_score = local_search_pair(best_col, best_score, max_passes=3)

        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall = best_col.copy()

    # 3. Final polish on the best overall
    if best_overall is not None and time.time() < deadline - 2.0:
        col = best_overall.copy()
        current_score = best_overall_score
        col, current_score = local_search_single(col, current_score, max_passes=30)

        if current_score < best_overall_score:
            best_overall_score = current_score
            best_overall = col

        # Final pair polish
        if time.time() < deadline - 1.0:
            col, current_score = local_search_pair(col, current_score, max_passes=2)
            if current_score < best_overall_score:
                best_overall_score = current_score
                best_overall = col

    return [int(c) for c in best_overall]
# EVOLVE-BLOCK-END
