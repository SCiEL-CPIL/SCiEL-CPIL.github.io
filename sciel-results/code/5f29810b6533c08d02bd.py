"""
Search-based optimizer for two-colourings of [1,256] minimizing monochromatic 4-APs.
Uses simulated annealing with efficient delta computation and a time budget.
"""
import time
import random
import numpy as np


def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Precompute all 4-term APs as index tuples (0-indexed)
    # A 4-AP is (x, x+d, x+2d, x+3d) with 1 <= x and x+3d <= 256, d >= 1
    # 0-indexed: (x-1, x-1+d, x-1+2d, x-1+3d)
    aps = []
    for d in range(1, 86):  # d from 1 to 85 (since 3*d <= 255)
        max_start = 256 - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))

    aps = np.array(aps, dtype=np.int32)
    n_aps = len(aps)

    # Build for each position, the list of AP indices it participates in
    pos_aps = [[] for _ in range(256)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append(idx)
        pos_aps[i1].append(idx)
        pos_aps[i2].append(idx)
        pos_aps[i3].append(idx)

    def count_mono(colors):
        """Count monochromatic 4-APs given a color array."""
        # colors is a numpy array of 0s and 1s
        # For each AP, check if all four positions have the same color
        # Vectorized approach
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        # Monochromatic if all equal
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(mono.sum())

    def calc_delta(colors, pos, new_color):
        """
        Calculate the change in monochromatic AP count when flipping position 'pos'
        to 'new_color'. Returns the delta (new_count - old_count).
        """
        old_color = colors[pos]
        if old_color == new_color:
            return 0

        delta = 0
        for ap_idx in pos_aps[pos]:
            i0, i1, i2, i3 = aps[ap_idx]
            # Get colors of the four positions
            c0 = colors[i0]
            c1 = colors[i1]
            c2 = colors[i2]
            c3 = colors[i3]

            # Check if this AP is currently monochromatic
            was_mono = (c0 == c1) and (c1 == c2) and (c2 == c3)

            # After flip, the color at 'pos' becomes new_color
            # We need to determine the new colors
            new_c0 = new_color if i0 == pos else c0
            new_c1 = new_color if i1 == pos else c1
            new_c2 = new_color if i2 == pos else c2
            new_c3 = new_color if i3 == pos else c3

            is_mono_now = (new_c0 == new_c1) and (new_c1 == new_c2) and (new_c2 == new_c3)

            if was_mono and not is_mono_now:
                delta -= 1
            elif not was_mono and is_mono_now:
                delta += 1

        return delta

    # Initial coloring: random
    random.seed(42)
    colors = np.array([random.randint(0, 1) for _ in range(256)], dtype=np.int8)
    current_count = count_mono(colors)
    best_colors = colors.copy()
    best_count = current_count

    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995

    step = 0
    while time.time() < deadline:
        # Pick a random position to flip
        pos = random.randint(0, 255)
        new_color = 1 - colors[pos]

        delta = calc_delta(colors, pos, new_color)

        if delta <= 0 or random.random() < np.exp(-delta / T):
            colors[pos] = new_color
            current_count += delta

            if current_count < best_count:
                best_count = current_count
                best_colors = colors.copy()

        T *= cooling_rate
        if T < T_min:
            T = T_min

        step += 1
        # Periodic restart if stuck
        if step % 50000 == 0 and T <= T_min * 1.01:
            # Try a few random restarts with local improvement
            for _ in range(5):
                trial = np.array([random.randint(0, 1) for _ in range(256)], dtype=np.int8)
                trial_count = count_mono(trial)
                if trial_count < best_count:
                    best_count = trial_count
                    best_colors = trial.copy()
                    colors = trial.copy()
                    current_count = trial_count
            T = 5.0

    return [int(c) for c in best_colors]
