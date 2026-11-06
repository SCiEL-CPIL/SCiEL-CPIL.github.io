"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    N = 256

    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= N, d >= 1
    # Store 0-indexed positions
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            # 0-indexed
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)

    # For each position, list of AP indices that contain it
    # aps_at[i] = list of AP indices where position i appears
    aps_at = [[] for _ in range(N)]
    for idx in range(num_aps):
        for pos in aps[idx]:
            aps_at[pos].append(idx)

    # Convert to numpy arrays for speed
    aps_at_arrays = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in aps_at]

    def count_mono(col):
        """Count monochromatic 4-APs given color array col (0/1)."""
        # For each AP, check if all 4 positions have the same color
        # Vectorized: col[aps] gives shape (num_aps, 4)
        c = col[aps]  # shape (num_aps, 4)
        # An AP is monochromatic if all 4 are 0 or all 4 are 1
        all_zero = np.all(c == 0, axis=1)
        all_one = np.all(c == 1, axis=1)
        return int(np.sum(all_zero) + np.sum(all_one))

    def fast_delta(col, i, new_val):
        """Compute change in mono count if we flip position i to new_val.
        Returns (delta, old_mono_count_affected) where delta = new_total - old_total.
        Actually, we compute the change in the number of monochromatic APs
        among the APs that contain position i.
        """
        affected = aps_at_arrays[i]
        if len(affected) == 0:
            return 0

        # Get the colors of the 4 positions in each affected AP
        # aps[affected] has shape (len(affected), 4)
        sub = col[aps[affected]]  # shape (len_affected, 4)

        # Before flip: check which are monochromatic
        old_mono = np.all(sub == sub[:, 0:1], axis=1)  # all equal to first element
        old_count = int(np.sum(old_mono))

        # After flip: set position i to new_val in the sub array
        # We need to find where i is in each AP's 4 positions
        # For each affected AP, find the column index where the position is i
        # aps[affected] has the actual positions; we need to find which column is i
        # This is tricky with numpy. Let's do it differently.

        # Actually, let's just recompute for affected APs
        # Create a copy of col with the flip
        col_new = col.copy()
        col_new[i] = new_val

        sub_new = col_new[aps[affected]]
        new_mono = np.all(sub_new == sub_new[:, 0:1], axis=1)
        new_count = int(np.sum(new_mono))

        return new_count - old_count

    def sa_run(seed, max_steps, start_temp, end_temp):
        """Run simulated annealing. Returns best color array and its score."""
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        # Start with random coloring
        col = np_rng.randint(0, 2, size=N).astype(np.int32)
        current_score = count_mono(col)
        best_col = col.copy()
        best_score = current_score

        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break

            # Temperature schedule (geometric cooling)
            t = start_temp * (end_temp / start_temp) ** (step / max_steps)

            # Pick a random position to flip
            i = rng.randint(0, N - 1)
            new_val = 1 - col[i]

            # Compute delta
            delta = fast_delta(col, i, new_val)

            if delta <= 0:
                # Accept
                col[i] = new_val
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_col = col.copy()
            else:
                # Accept with probability exp(-delta / t)
                if t > 0 and rng.random() < np.exp(-delta / t):
                    col[i] = new_val
                    current_score += delta

        return best_col, best_score

    # Multiple restarts
    best_overall = None
    best_overall_score = float('inf')

    # First, try a few simple structured colorings
    # Alternating blocks of 2
    col_alt2 = np.array([(i // 2) % 2 for i in range(N)], dtype=np.int32)
    s = count_mono(col_alt2)
    if s < best_overall_score:
        best_overall_score = s
        best_overall = col_alt2.copy()

    # Alternating blocks of 4
    col_alt4 = np.array([(i // 4) % 2 for i in range(N)], dtype=np.int32)
    s = count_mono(col_alt4)
    if s < best_overall_score:
        best_overall_score = s
        best_overall = col_alt4.copy()

    # Random starting points with SA
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        seed = 42 + r * 1000
        # Use more steps if we have time
        max_steps = 50000
        start_temp = 5.0
        end_temp = 0.01

        best_col, best_score = sa_run(seed, max_steps, start_temp, end_temp)

        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall = best_col.copy()

    # Local search refinement on best result
    if best_overall is not None and time.time() < deadline - 5.0:
        col = best_overall.copy()
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
        if current_score < best_overall_score:
            best_overall_score = current_score
            best_overall = col.copy()

    return [int(c) for c in best_overall]
# EVOLVE-BLOCK-END
