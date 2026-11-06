"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    N = 256

    # Precompute all 4-term arithmetic progressions
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)

    # For each position, precompute:
    #   - list of AP indices that contain it
    #   - for each such AP, which column (0-3) the position occupies
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
        """Count monochromatic 4-APs."""
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

    def thue_morse(n):
        """Generate Thue-Morse sequence of length n."""
        seq = [0]
        for _ in range(n.bit_length() - 1):
            seq = seq + [1 - x for x in seq]
        return seq[:n]

    def coordinate_descent(col, max_rounds=50):
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

    def pair_flip_search(col, max_rounds=20):
        """Try flipping pairs of positions to escape local minima."""
        current_score = count_mono(col)
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            improved = False
            # Try random pairs
            for _ in range(200):
                if time.time() > deadline - 3.0:
                    break
                i, j = random.sample(range(N), 2)
                # Flip both
                old_i, old_j = col[i], col[j]
                col[i] = 1 - old_i
                col[j] = 1 - old_j
                new_score = count_mono(col)
                delta = new_score - current_score
                if delta < 0:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    col[i] = old_i
                    col[j] = old_j
            if not improved:
                break
        return col, current_score

    def sa_from(col, seed, max_steps, start_temp, end_temp):
        """SA starting from given coloring."""
        rng = random.Random(seed)
        col = col.copy()
        current_score = count_mono(col)
        best_col = col.copy()
        best_score = current_score

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
                if current_score < best_score:
                    best_score = current_score
                    best_col = col.copy()
            else:
                if t > 0 and rng.random() < np.exp(-delta / t):
                    col[i] = new_val
                    current_score += delta
        return best_col, best_score

    best_overall = None
    best_overall_score = float('inf')

    # Start with Thue-Morse - known to break APs effectively
    tm_seq = thue_morse(N)
    col_tm = np.array(tm_seq, dtype=np.int32)
    s_tm = count_mono(col_tm)
    if s_tm < best_overall_score:
        best_overall_score = s_tm
        best_overall = col_tm.copy()

    # Also try complement of Thue-Morse
    col_tm_comp = 1 - col_tm
    s_tm_comp = count_mono(col_tm_comp)
    if s_tm_comp < best_overall_score:
        best_overall_score = s_tm_comp
        best_overall = col_tm_comp.copy()

    # Try periodic blocks as alternatives
    for block_size in [2, 3, 4, 5, 8]:
        col_struct = np.array([(i // block_size) % 2 for i in range(N)], dtype=np.int32)
        s = count_mono(col_struct)
        if s < best_overall_score:
            best_overall_score = s
            best_overall = col_struct.copy()

    # Coordinate descent from best
    if best_overall is not None:
        best_overall, best_overall_score = coordinate_descent(best_overall.copy())

    # Pair flip search to escape local minima
    if time.time() < deadline - 10.0:
        best_overall, best_overall_score = pair_flip_search(best_overall.copy())
        # Coordinate descent again after pair flips
        best_overall, best_overall_score = coordinate_descent(best_overall.copy())

    # SA from multiple perturbations of the best
    for r in range(10):
        if time.time() > deadline - 10.0:
            break
        seed = 5000 + r * 104729
        rng = random.Random(seed)
        perturbed = best_overall.copy()
        # Flip 5-10 random positions
        num_flips = rng.randint(5, 10)
        flip_positions = rng.sample(range(N), num_flips)
        for pos in flip_positions:
            perturbed[pos] = 1 - perturbed[pos]
        
        best_col, best_score = sa_from(perturbed, seed, 20000, 2.0, 0.01)
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall = best_col.copy()
            # Coordinate descent on new best
            best_overall, best_overall_score = coordinate_descent(best_overall.copy())

    # Final polish
    if best_overall is not None and time.time() < deadline - 5.0:
        best_overall, best_overall_score = coordinate_descent(best_overall)

    return [int(c) for c in best_overall]
# EVOLVE-BLOCK-END
