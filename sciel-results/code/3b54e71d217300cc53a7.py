import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Precompute all 4-term arithmetic progressions
    aps = []
    ap_indices = [[] for _ in range(N)]  # ap_indices[i] = list of AP IDs containing position i

    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx1 = a - 1
            idx2 = a + d - 1
            idx3 = a + 2 * d - 1
            idx4 = a + 3 * d - 1
            ap_id = len(aps)
            aps.append((idx1, idx2, idx3, idx4))
            for idx in (idx1, idx2, idx3, idx4):
                ap_indices[idx].append(ap_id)

    total_aps = len(aps)

    # Precompute for each position p and each AP containing p:
    # - the other 3 positions in that AP
    # - the index of p within the AP (0,1,2,3)
    # This allows fast delta computation without creating lists each time
    pos_ap_data = [[] for _ in range(N)]
    for p in range(N):
        for ap_id in ap_indices[p]:
            ap = aps[ap_id]
            others = []
            for j in range(4):
                if ap[j] != p:
                    others.append(ap[j])
            pos_in_ap = ap.index(p)
            pos_ap_data[p].append((ap_id, tuple(others), pos_in_ap))

    def count_mono(colors):
        """Count monochromatic 4-APs using numpy."""
        colors_arr = np.array(colors, dtype=np.int8)
        count = 0
        for ap in aps:
            if colors_arr[ap[0]] == colors_arr[ap[1]] == colors_arr[ap[2]] == colors_arr[ap[3]]:
                count += 1
        return count

    def fast_count_mono(colors):
        """Fast count using precomputed AP array."""
        colors_arr = np.array(colors, dtype=np.int8)
        # Vectorized check
        ap_arr = np.array(aps, dtype=np.int32)  # shape (total_aps, 4)
        c0 = colors_arr[ap_arr[:, 0]]
        c1 = colors_arr[ap_arr[:, 1]]
        c2 = colors_arr[ap_arr[:, 2]]
        c3 = colors_arr[ap_arr[:, 3]]
        mono_mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(mono_mask))

    def compute_delta_fast(p, colors):
        """Compute delta in mono count if we flip position p."""
        delta = 0
        for ap_id, others, pos_in_ap in pos_ap_data[p]:
            # Get colors of the 4 positions
            c0 = colors[aps[ap_id][0]]
            c1 = colors[aps[ap_id][1]]
            c2 = colors[aps[ap_id][2]]
            c3 = colors[aps[ap_id][3]]

            # Before flip: check if all 4 same
            mono_before = (c0 == c1 == c2 == c3)

            # After flip: position p gets flipped
            new_c = 1 - colors[p]
            # Check if all 4 same after flip
            if pos_in_ap == 0:
                mono_after = (new_c == c1 == c2 == c3)
            elif pos_in_ap == 1:
                mono_after = (c0 == new_c == c2 == c3)
            elif pos_in_ap == 2:
                mono_after = (c0 == c1 == new_c == c3)
            else:
                mono_after = (c0 == c1 == c2 == new_c)

            if mono_before and not mono_after:
                delta -= 1
            elif not mono_before and mono_after:
                delta += 1
        return delta

    # Simulated annealing
    best_colors = None
    best_count = float('inf')

    # Multiple restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Initialize
        if restart == 0:
            # Start with alternating pairs
            colors = [(i // 2) % 2 for i in range(N)]
        elif restart == 1:
            # Start with random
            colors = [random.randint(0, 1) for _ in range(N)]
        elif restart == 2:
            # Start with blocks of 4
            colors = [(i // 4) % 2 for i in range(N)]
        elif restart == 3:
            # Start with blocks of 8
            colors = [(i // 8) % 2 for i in range(N)]
        else:
            # Start with best found so far, perturbed
            if best_colors is not None:
                colors = best_colors[:]
                # Flip ~8% of bits
                for i in range(N):
                    if random.random() < 0.08:
                        colors[i] ^= 1
            else:
                colors = [random.randint(0, 1) for _ in range(N)]

        current_count = fast_count_mono(colors)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors[:]

        # Simulated annealing parameters
        T = 3.0
        T_min = 0.01
        cooling = 0.9997
        max_steps = 80000

        for step in range(max_steps):
            if time.time() > deadline:
                break

            # Pick a random position to flip
            p = random.randint(0, N - 1)
            old_color = colors[p]

            # Compute delta using fast method
            delta = compute_delta_fast(p, colors)

            # Accept or reject
            if delta <= 0:
                colors[p] = 1 - old_color
                current_count += delta
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors[:]
            else:
                if random.random() < math.exp(-delta / T):
                    colors[p] = 1 - old_color
                    current_count += delta

            T = max(T_min, T * cooling)

        # Local search (hill climbing) from current state
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for p in range(N):
                if time.time() > deadline:
                    break
                delta = compute_delta_fast(p, colors)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_colors = colors[:]

    # Final local search from best
    if best_colors is not None:
        colors = best_colors[:]
        current_count = fast_count_mono(colors)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for p in range(N):
                if time.time() > deadline:
                    break
                delta = compute_delta_fast(p, colors)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_colors = colors[:]

    if best_colors is not None:
        return best_colors
    else:
        return [(i // 2) % 2 for i in range(N)]
