"""Combined program: diverse structured initializations + fast vectorized coordinate descent + SA."""
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

    return [int(c) for c in best_col]
# EVOLVE-BLOCK-END
