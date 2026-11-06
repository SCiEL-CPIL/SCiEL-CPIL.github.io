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
    # This lets us compute delta without copying the full color array.
    pos_aps = [[] for _ in range(N)]   # pos_aps[i] = list of (ap_idx, col_idx)
    for idx in range(num_aps):
        for col_idx in range(4):
            pos = aps[idx, col_idx]
            pos_aps[pos].append((idx, col_idx))

    # Convert to numpy arrays: pos_aps_arr[i] = array of shape (k, 2)
    pos_aps_arr = []
    for i in range(N):
        if pos_aps[i]:
            arr = np.array(pos_aps[i], dtype=np.int32)  # (k, 2)
            pos_aps_arr.append(arr)
        else:
            pos_aps_arr.append(np.empty((0, 2), dtype=np.int32))

    def count_mono(col):
        """Count monochromatic 4-APs."""
        c = col[aps]  # (num_aps, 4)
        all_zero = np.all(c == 0, axis=1)
        all_one = np.all(c == 1, axis=1)
        return int(np.sum(all_zero) + np.sum(all_one))

    def fast_delta(col, i, new_val):
        """
        Compute change in mono count if we set position i to new_val.
        Uses precomputed pos_aps_arr[i] to only check affected APs.
        """
        pa = pos_aps_arr[i]
        k = pa.shape[0]
        if k == 0:
            return 0

        ap_indices = pa[:, 0]  # (k,)
        col_indices = pa[:, 1]  # (k,)

        # Get the 4 colors for each affected AP
        sub = col[aps[ap_indices]]  # (k, 4)

        # Before: check if all 4 are equal
        old_mono = np.all(sub == sub[:, 0:1], axis=1)  # (k,)
        old_count = int(np.sum(old_mono))

        # After: set the specific column to new_val
        sub[:, col_indices] = new_val
        new_mono = np.all(sub == sub[:, 0:1], axis=1)
        new_count = int(np.sum(new_mono))

        return new_count - old_count

    def sa_run(seed, max_steps, start_temp, end_temp):
        """Run simulated annealing."""
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        col = np_rng.randint(0, 2, size=N).astype(np.int32)
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

    def polish(col):
        """Iterate single-flip improvements until no improvement."""
        current_score = count_mono(col)
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(N):
                if time.time() > deadline - 3.0:
                    break
                new_val = 1 - col[i]
                delta = fast_delta(col, i, new_val)
                if delta < 0:
                    col[i] = new_val
                    current_score += delta
                    improved = True
        return col, current_score

    # Multiple restarts
    best_overall = None
    best_overall_score = float('inf')

    # Try some structured starting points
    for block_size in [2, 3, 4, 5, 8]:
        col_struct = np.array([(i // block_size) % 2 for i in range(N)], dtype=np.int32)
        s = count_mono(col_struct)
        if s < best_overall_score:
            best_overall_score = s
            best_overall = col_struct.copy()

    # SA with many restarts
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        seed = 1000 + r * 7919
        max_steps = 30000
        start_temp = 3.0
        end_temp = 0.01

        best_col, best_score = sa_run(seed, max_steps, start_temp, end_temp)

        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall = best_col.copy()

    # Final polish
    if best_overall is not None and time.time() < deadline - 5.0:
        best_overall, best_overall_score = polish(best_overall)

    return [int(c) for c in best_overall]
# EVOLVE-BLOCK-END
