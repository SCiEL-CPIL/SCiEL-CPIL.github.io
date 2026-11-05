"""
Fast optimizer for two-colourings of [1,256] minimizing monochromatic 4-APs.
Uses vectorized NumPy delta computation and multi-restart simulated annealing.
"""
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit

    # Precompute all 4-term APs as 0-indexed tuples
    aps = []
    for d in range(1, 86):
        max_start = 256 - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))

    num_aps = len(aps)

    # For each position, store (ap_index, other_positions) for fast delta
    pos_to_aps = [[] for _ in range(256)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        for pos in (i0, i1, i2, i3):
            others = tuple(x for x in (i0, i1, i2, i3) if x != pos)
            pos_to_aps[pos].append((idx, others))

    # Precompute as numpy arrays for vectorized access
    # For each position, store the 3 "other" indices as a 2D array
    pos_others_arrays = []
    pos_ap_indices = []
    for pos in range(256):
        if pos_to_aps[pos]:
            others_list = [o for (_, o) in pos_to_aps[pos]]
            ap_idx_list = [idx for (idx, _) in pos_to_aps[pos]]
            pos_others_arrays.append(np.array(others_list, dtype=np.int32))
            pos_ap_indices.append(np.array(ap_idx_list, dtype=np.int32))
        else:
            pos_others_arrays.append(np.empty((0, 3), dtype=np.int32))
            pos_ap_indices.append(np.empty(0, dtype=np.int32))

    # Vectorized count
    aps_arr = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)

    def count_mono(colors):
        # colors is a numpy array of 0/1
        c = colors[aps_arr]  # shape (num_aps, 4)
        # Mono if all 4 are equal
        return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))

    def calc_delta_vec(colors, pos):
        """Compute delta for flipping position pos, using vectorized ops."""
        others = pos_others_arrays[pos]
        if len(others) == 0:
            return 0
        # others shape: (n, 3)
        c_others = colors[others]  # shape (n, 3)
        # Check if all 3 others are same color
        all_same = (c_others[:, 0] == c_others[:, 1]) & (c_others[:, 1] == c_others[:, 2])
        if not np.any(all_same):
            return 0
        # For those where all 3 are same:
        #   if colors[pos] == colors[other], delta -= 1 (was mono, becomes non-mono)
        #   if colors[pos] != colors[other], delta += 1 (was non-mono, becomes mono)
        pos_color = colors[pos]
        other_color = c_others[all_same, 0]  # the common color of the 3 others
        # delta = -1 where pos_color == other_color, +1 where pos_color != other_color
        # delta = 1 - 2*(pos_color == other_color)
        delta = int(np.sum(1 - 2 * (pos_color == other_color)))
        return delta

    def greedy_local_search(colors, current_count, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try all positions, flip if it helps
            for pos in range(256):
                delta = calc_delta_vec(colors, pos)
                if delta < 0:
                    colors[pos] = 1 - colors[pos]
                    current_count += delta
                    improved = True
        return colors, current_count

    def simulated_annealing(colors, current_count, T_start, T_end, cooling, time_limit, rng):
        T = T_start
        while time.time() < time_limit and T > T_end:
            pos = rng.randint(0, 255)
            delta = calc_delta_vec(colors, pos)

            if delta <= 0:
                colors[pos] = 1 - colors[pos]
                current_count += delta
            else:
                # Accept with probability exp(-delta/T)
                if rng.random() < np.exp(-delta / T):
                    colors[pos] = 1 - colors[pos]
                    current_count += delta

            T *= cooling

        return colors, current_count

    best_count = float('inf')
    best_colors = None

    # Multiple restarts with diverse initializations
    num_restarts = 12
    time_per_restart = (deadline - time.time()) / num_restarts

    rng = random.Random(12345)

    for restart in range(num_restarts):
        remaining_time = deadline - time.time()
        if remaining_time < 3:
            break

        # Allocate time: 60% SA, 25% greedy, 15% second greedy pass
        sa_time = remaining_time * 0.55
        greedy_time1 = remaining_time * 0.25
        greedy_time2 = remaining_time * 0.15

        # Different initialization strategies
        if restart == 0:
            colors = np.random.randint(0, 2, size=256).astype(np.int8)
        elif restart == 1:
            colors = np.array([(i % 2) for i in range(256)], dtype=np.int8)
        elif restart == 2:
            colors = np.array([(i // 2) % 2 for i in range(256)], dtype=np.int8)
        elif restart == 3:
            colors = np.array([(i // 4) % 2 for i in range(256)], dtype=np.int8)
        elif restart == 4:
            colors = np.array([(i // 8) % 2 for i in range(256)], dtype=np.int8)
        elif restart == 5:
            colors = np.zeros(256, dtype=np.int8)
        elif restart == 6:
            colors = np.ones(256, dtype=np.int8)
        elif restart == 7:
            # Random with structure: blocks of random sizes
            colors = np.zeros(256, dtype=np.int8)
            i = 0
            while i < 256:
                block_size = rng.randint(1, 8)
                val = rng.randint(0, 1)
                end = min(i + block_size, 256)
                colors[i:end] = val
                i = end
        else:
            # Perturb best solution found so far
            if best_colors is not None:
                colors = best_colors.copy()
                # Flip a random number of positions (20-60)
                n_flips = rng.randint(20, 60)
                for _ in range(n_flips):
                    colors[rng.randint(0, 255)] = 1 - colors[rng.randint(0, 255)]
            else:
                colors = np.random.randint(0, 2, size=256).astype(np.int8)

        current_count = count_mono(colors)

        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        # Run simulated annealing
        T_start = 20.0
        T_end = 0.3
        cooling = 0.9995
        sa_deadline = time.time() + sa_time
        colors, current_count = simulated_annealing(
            colors, current_count, T_start, T_end, cooling, sa_deadline, rng
        )

        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        # Run greedy local search
        greedy_deadline1 = time.time() + greedy_time1
        colors, current_count = greedy_local_search(colors, current_count, greedy_deadline1)

        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        # Second greedy pass (may find more improvements after first pass)
        greedy_deadline2 = time.time() + greedy_time2
        colors, current_count = greedy_local_search(colors, current_count, greedy_deadline2)

        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

    # Final polish with all remaining time
    if best_colors is not None and time.time() < deadline - 2:
        colors = best_colors.copy()
        current_count = best_count
        final_deadline = deadline - 1
        colors, current_count = greedy_local_search(colors, current_count, final_deadline)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors

    return [int(c) for c in best_colors]
