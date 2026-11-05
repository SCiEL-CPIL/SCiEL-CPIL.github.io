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
    # Also store which index (0,1,2,3) this position occupies in each AP
    pos_aps = [[] for _ in range(256)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append((idx, 0))
        pos_aps[i1].append((idx, 1))
        pos_aps[i2].append((idx, 2))
        pos_aps[i3].append((idx, 3))

    def count_mono(colors):
        """Count monochromatic 4-APs given a color array."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(mono.sum())

    def calc_delta_fast(colors, pos, new_color):
        """
        Fast delta calculation for flipping position 'pos' to 'new_color'.
        Uses precomputed position-in-AP info to avoid redundant lookups.
        """
        old_color = colors[pos]
        if old_color == new_color:
            return 0

        delta = 0
        for ap_idx, my_idx in pos_aps[pos]:
            # Get the other three colors
            row = aps[ap_idx]
            c0 = colors[row[0]]
            c1 = colors[row[1]]
            c2 = colors[row[2]]
            c3 = colors[row[3]]

            # Determine if currently monochromatic
            was_mono = (c0 == c1) and (c1 == c2) and (c2 == c3)

            # After flip, only the color at 'pos' changes
            # The other three remain the same
            # New color at pos is new_color
            # Check if new configuration is monochromatic:
            # All four must be equal, so the three others must all equal new_color
            others_same_as_new = (
                (c0 == new_color if my_idx != 0 else True) and
                (c1 == new_color if my_idx != 1 else True) and
                (c2 == new_color if my_idx != 2 else True) and
                (c3 == new_color if my_idx != 3 else True)
            )
            is_mono_now = others_same_as_new

            if was_mono and not is_mono_now:
                delta -= 1
            elif not was_mono and is_mono_now:
                delta += 1

        return delta

    def greedy_local_search(colors, current_count, max_iters=5000):
        """
        Greedy local search: repeatedly flip the position that gives the best improvement.
        Returns improved colors and count.
        """
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            best_delta = 0
            best_pos = -1
            best_new_color = -1

            # Try all positions
            for pos in range(256):
                new_color = 1 - colors[pos]
                d = calc_delta_fast(colors, pos, new_color)
                if d < best_delta:
                    best_delta = d
                    best_pos = pos
                    best_new_color = new_color

            if best_pos >= 0:
                colors[best_pos] = best_new_color
                current_count += best_delta
                improved = True
                iters += 1

        return colors, current_count

    # Initial coloring: random
    random.seed(42)
    colors = np.array([random.randint(0, 1) for _ in range(256)], dtype=np.int8)
    current_count = count_mono(colors)
    best_colors = colors.copy()
    best_count = current_count

    # First, do a greedy local search from the random start
    colors, current_count = greedy_local_search(colors, current_count)
    if current_count < best_count:
        best_count = current_count
        best_colors = colors.copy()

    # Simulated annealing with multiple phases
    T_start = 5.0
    T_min = 0.01
    cooling_rate = 0.9998
    steps_per_phase = 20000

    phase = 0
    while time.time() < deadline - 10:
        T = T_start
        phase_steps = 0

        while time.time() < deadline - 10 and phase_steps < steps_per_phase:
            pos = random.randint(0, 255)
            new_color = 1 - colors[pos]

            delta = calc_delta_fast(colors, pos, new_color)

            if delta <= 0 or random.random() < np.exp(-delta / T):
                colors[pos] = new_color
                current_count += delta

                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()

            T *= cooling_rate
            if T < T_min:
                T = T_min
            phase_steps += 1

        # After each SA phase, run greedy local search
        colors, current_count = greedy_local_search(colors, current_count)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        phase += 1

        # Occasionally try a fresh random restart
        if phase % 3 == 0:
            trial = np.array([random.randint(0, 1) for _ in range(256)], dtype=np.int8)
            trial_count = count_mono(trial)
            trial, trial_count = greedy_local_search(trial, trial_count)
            if trial_count < best_count:
                best_count = trial_count
                best_colors = trial.copy()
                colors = trial.copy()
                current_count = trial_count

    return [int(c) for c in best_colors]
