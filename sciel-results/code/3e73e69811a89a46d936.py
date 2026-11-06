"""Combined program: diverse initializations + fast vectorized local search + multi-flip SA."""
import numpy as np
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    N = 256

    # Precompute all 4-term arithmetic progressions (0-indexed)
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)

    # For each position, precompute: list of (ap_index, column_index)
    pos_aps = [[] for _ in range(N)]
    for idx in range(num_aps):
        for col_idx in range(4):
            pos = aps[idx, col_idx]
            pos_aps[pos].append((idx, col_idx))

    pos_aps_arr = []
    for i in range(N):
        if pos_aps[i]:
            arr = np.array(pos_aps[i], dtype=np.int32)
            pos_aps_arr.append(arr)
        else:
            pos_aps_arr.append(np.empty((0, 2), dtype=np.int32))

    # Also precompute "others" for each position: for each AP containing i, the other 3 positions
    others_for_pos = [None] * N
    for i in range(N):
        pa = pos_aps_arr[i]
        k = pa.shape[0]
        if k == 0:
            others_for_pos[i] = np.array([], dtype=np.int32).reshape(0, 3)
        else:
            ap_indices = pa[:, 0]
            col_indices = pa[:, 1]
            others_list = []
            for j in range(k):
                row = aps[ap_indices[j]]
                others = [p for p in row if p != i]
                others_list.append(others)
            others_for_pos[i] = np.array(others_list, dtype=np.int32)

    def count_mono(col):
        """Count monochromatic 4-APs using vectorized all-equal check."""
        c = col[aps]
        all_zero = np.all(c == 0, axis=1)
        all_one = np.all(c == 1, axis=1)
        return int(np.sum(all_zero) + np.sum(all_one))

    def fast_delta(col, i, new_val):
        """Compute change in mono count if we set position i to new_val."""
        pa = pos_aps_arr[i]
        k = pa.shape[0]
        if k == 0:
            return 0
        ap_indices = pa[:, 0]
        col_indices = pa[:, 1]
        sub = col[aps[ap_indices]].copy()
        old_mono = np.all(sub == sub[:, 0:1], axis=1)
        old_count = int(np.sum(old_mono))
        sub[:, col_indices] = new_val
        new_mono = np.all(sub == sub[:, 0:1], axis=1)
        new_count = int(np.sum(new_mono))
        return new_count - old_count

    def fast_delta_others(col, i):
        """Faster delta computation using precomputed others."""
        others = others_for_pos[i]
        k = others.shape[0]
        if k == 0:
            return 0
        oc = col[others]  # shape (k, 3)
        all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
        if not np.any(all_same):
            return 0
        c_other = oc[:, 0]
        c_i = col[i]
        before = (c_other == c_i) & all_same
        after = (c_other == (1 - c_i)) & all_same
        return int(np.sum(after) - np.sum(before))

    def make_thue_morse():
        """Thue-Morse sequence: t(n) = popcount(n) % 2."""
        seq = np.zeros(N, dtype=np.int32)
        for i in range(N):
            seq[i] = bin(i).count('1') % 2
        return seq

    def make_block_coloring(block_size):
        return np.array([(i // block_size) % 2 for i in range(N)], dtype=np.int32)

    def make_mod_coloring(mod):
        return np.array([(i % mod) % 2 for i in range(N)], dtype=np.int32)

    def make_xor_pattern(x):
        """Color based on XOR of block index and sub-block index."""
        colors = np.zeros(N, dtype=np.int32)
        for i in range(N):
            block = i // x
            sub = (i % x) // (x // 2) if x >= 2 else 0
            colors[i] = (block % 2) ^ (sub % 2)
        return colors

    def make_random():
        return np.random.randint(0, 2, N).astype(np.int32)

    # Generate diverse structured candidates
    candidates = []
    candidates.append(make_thue_morse())
    candidates.append(1 - make_thue_morse())
    for bs in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 64]:
        candidates.append(make_block_coloring(bs))
    for mod in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32]:
        candidates.append(make_mod_coloring(mod))
    for x in [4, 8, 16, 32, 64]:
        candidates.append(make_xor_pattern(x))
    for _ in range(20):
        candidates.append(make_random())

    # Find best initial coloring
    best_col = None
    best_score = float('inf')
    for cand in candidates:
        s = count_mono(cand)
        if s < best_score:
            best_score = s
            best_col = cand.copy()

    def coordinate_descent(col, max_rounds=100):
        """Iterate single-flip improvements until convergence or time limit."""
        current_score = count_mono(col)
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for i in range(N):
                new_val = 1 - col[i]
                delta = fast_delta(col, i, new_val)
                if delta < 0:
                    col[i] = new_val
                    current_score += delta
                    improved = True
            if not improved:
                break
        return col, current_score

    def local_search_fast(col, score, max_passes=200):
        """Fast local search using precomputed deltas, multiple passes until convergence."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline - 2.0:
                break
            improved = False
            passes += 1
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 2.0:
                    break
                d = fast_delta_others(col, i)
                if d < 0:
                    col[i] = 1 - col[i]
                    score += d
                    improved = True
        return col, score

    def sa_from(col, seed, max_steps, start_temp, end_temp):
        """SA starting from given coloring."""
        rng = random.Random(seed)
        col = col.copy()
        current_score = count_mono(col)
        best_col_local = col.copy()
        best_score_local = current_score

        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            t = start_temp * (end_temp / start_temp) ** (step / max_steps)
            i = rng.randint(0, N - 1)
            new_val = 1 - col[i]
            delta = fast_delta(col, i, new_val)
            if delta <= 0:
                col[i] = new_val
                current_score += delta
                if current_score < best_score_local:
                    best_score_local = current_score
                    best_col_local = col.copy()
            else:
                if t > 0 and rng.random() < np.exp(-delta / t):
                    col[i] = new_val
                    current_score += delta
        return best_col_local, best_score_local

    def multi_flip_escape(col, score, num_flips, max_rounds=50):
        """Try flipping multiple bits simultaneously to escape local minima."""
        rng = random.Random(int(time.time() * 1000) % 100000)
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            indices = rng.sample(range(N), num_flips)
            new_col = col.copy()
            for i in indices:
                new_col[i] = 1 - new_col[i]
            new_score = count_mono(new_col)
            delta = new_score - score
            if delta < 0 or (delta <= 10 and rng.random() < np.exp(-delta / 3.0)):
                col = new_col
                score = new_score
        return col, score

    # Coordinate descent from best
    best_col, best_score = coordinate_descent(best_col.copy())

    # Multi-round SA from perturbations of the best
    num_sa_rounds = 15
    for r in range(num_sa_rounds):
        if time.time() > deadline - 10.0:
            break
        seed = 10000 + r * 7919
        rng = random.Random(seed)
        perturbed = best_col.copy()
        num_flips = rng.randint(3, 12)
        flip_positions = rng.sample(range(N), num_flips)
        for pos in flip_positions:
            perturbed[pos] = 1 - perturbed[pos]

        # SA with varying temperatures
        start_temp = 3.0
        end_temp = 0.005
        max_steps = 30000
        best_col_local, best_score_local = sa_from(perturbed, seed, max_steps, start_temp, end_temp)

        if best_score_local < best_score:
            best_score = best_score_local
            best_col = best_col_local.copy()
            # Coordinate descent on new best
            best_col, best_score = coordinate_descent(best_col.copy())

    # Final polish with coordinate descent
    if best_col is not None and time.time() < deadline - 5.0:
        best_col, best_score = coordinate_descent(best_col)

    # One more round of SA from the final best if time permits
    if time.time() < deadline - 15.0:
        seed = 99999
        perturbed = best_col.copy()
        rng = random.Random(seed)
        num_flips = rng.randint(2, 8)
        flip_positions = rng.sample(range(N), num_flips)
        for pos in flip_positions:
            perturbed[pos] = 1 - perturbed[pos]
        best_col_local, best_score_local = sa_from(perturbed, seed, 20000, 2.0, 0.01)
        if best_score_local < best_score:
            best_score = best_score_local
            best_col = best_col_local.copy()
            best_col, best_score = coordinate_descent(best_col)

    # Multi-flip escape attempts if time permits
    if time.time() < deadline - 10.0:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            col_local, score_local = multi_flip_escape(best_col.copy(), best_score, num_flips=4)
            if score_local < best_score:
                best_score = score_local
                best_col = col_local.copy()
                best_col, best_score = coordinate_descent(best_col)

    return [int(c) for c in best_col]
# EVOLVE-BLOCK-END
