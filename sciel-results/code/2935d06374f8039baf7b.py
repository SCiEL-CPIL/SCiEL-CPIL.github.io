"""Rewritten program: Penalty-annealing on QUBO formulation for 4-term AP minimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = 112.0  # Safety margin before 120s/150s kill

    n = 256

    # Generate all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # 1-indexed, stored as 0-indexed
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))

    total_aps = len(aps)

    # Build adjacency: for each position, list of (ap_index, position_in_ap)
    ap_indices_by_pos = [[] for _ in range(n)]
    for ap_idx, ap in enumerate(aps):
        for pos in range(4):
            ap_indices_by_pos[ap[pos]].append((ap_idx, pos))

    # Precompute for each AP the other 3 positions given each position index
    # ap_others[ap_idx][pos] = tuple of the other 3 positions
    ap_others = [[None]*4 for _ in range(total_aps)]
    for ap_idx, (i0, i1, i2, i3) in enumerate(aps):
        ap_others[ap_idx][0] = (i1, i2, i3)
        ap_others[ap_idx][1] = (i0, i2, i3)
        ap_others[ap_idx][2] = (i0, i1, i3)
        ap_others[ap_idx][3] = (i0, i1, i2)

    def count_mono_vec(colors):
        """Count monochromatic 4-APs using vectorized numpy."""
        # colors is a numpy array of 0/1
        count = 0
        for ap in aps:
            i0, i1, i2, i3 = ap
            c = colors[i0]
            if colors[i1] == c and colors[i2] == c and colors[i3] == c:
                count += 1
        return count

    def count_mono_numpy(colors):
        """Fully vectorized count using numpy."""
        # colors: shape (n,) of 0/1
        # For each AP, check if all 4 are equal
        # Build index array
        idx = np.array(aps)  # shape (total_aps, 4)
        # Get colors at all positions
        c = colors[idx]  # shape (total_aps, 4)
        # Check if all equal: c[:,0] == c[:,1] == c[:,2] == c[:,3]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.sum(mono))

    def get_delta(k, colors):
        """Compute change in mono count if we flip colors[k]."""
        v = colors[k]
        v_new = 1 - v
        delta = 0
        for (ap_idx, pos) in ap_indices_by_pos[k]:
            others = ap_others[ap_idx][pos]
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            
            # Currently mono?
            curr_mono = (c1 == v) and (c2 == v) and (c3 == v)
            # New mono?
            new_mono = (c1 == v_new) and (c2 == v_new) and (c3 == v_new)
            
            if curr_mono and not new_mono:
                delta -= 1
            elif not curr_mono and new_mono:
                delta += 1
        return delta

    # Thue-Morse initialization
    def thue_morse_init(n):
        colors = np.zeros(n, dtype=np.int8)
        for i in range(n):
            # Count bits in binary representation
            colors[i] = bin(i).count('1') % 2
        return colors

    # Initialize with Thue-Morse
    current_colors = thue_morse_init(n)
    current_score = count_mono_numpy(current_colors)
    best_colors = current_colors.copy()
    best_score = current_score

    # Penalty-annealing: we treat this as minimizing the number of monochromatic 4-APs
    # The key insight: instead of pure local search, we use a penalty-based approach
    # where we track "pressure" on each position and use it to guide flips

    # Track how many times each position is in a monochromatic AP
    def compute_pressure(colors):
        """For each position, count how many monochromatic APs it's in."""
        pressure = np.zeros(n, dtype=np.int32)
        for ap in aps:
            i0, i1, i2, i3 = ap
            c = colors[i0]
            if colors[i1] == c and colors[i2] == c and colors[i3] == c:
                pressure[i0] += 1
                pressure[i1] += 1
                pressure[i2] += 1
                pressure[i3] += 1
        return pressure

    pressure = compute_pressure(current_colors)

    # Simulated annealing with pressure-guided selection
    T = 2.0
    cooling = 0.9997
    min_T = 0.02
    rng = np.random.default_rng(42)

    steps = 0
    no_improve = 0
    max_no_improve = 20000

    while time.time() < deadline:
        # Select position: weighted by pressure (positions in more mono-APs are more important)
        # Use softmax-like selection
        if T > 0.1:
            # Mix uniform and pressure-based selection
            mix = 0.3  # 30% pressure-guided, 70% uniform
            if rng.random() < mix:
                # Select proportional to pressure (with minimum 1 to avoid zero)
                weights = pressure + 1
                weights = weights / weights.sum()
                k = int(rng.choice(n, p=weights))
            else:
                k = int(rng.integers(0, n))
        else:
            # At low temperature, focus on high-pressure positions
            if rng.random() < 0.7:
                weights = pressure + 1
                weights = weights / weights.sum()
                k = int(rng.choice(n, p=weights))
            else:
                k = int(rng.integers(0, n))

        delta = get_delta(k, current_colors)

        # Acceptance
        accept = False
        if delta < 0:
            accept = True
            no_improve = 0
        elif delta == 0:
            accept = True
            no_improve += 1
        else:
            # Metropolis
            if T > 0.05:
                # Use log-space to avoid underflow
                log_prob = -delta / T
                if log_prob > -50:
                    prob = math.exp(log_prob)
                else:
                    prob = 0.0
                if rng.random() < prob:
                    accept = True
                    no_improve += 1

        if accept:
            current_colors[k] = 1 - current_colors[k]
            current_score += delta

            # Update pressure incrementally
            # Recompute pressure for affected APs
            # For efficiency, only update if delta changed things
            for (ap_idx, pos) in ap_indices_by_pos[k]:
                i0, i1, i2, i3 = aps[ap_idx]
                c = current_colors[i0]
                is_mono = (current_colors[i1] == c and current_colors[i2] == c and current_colors[i3] == c)
                # We need to know if this AP was mono before the flip
                # This is complex to track incrementally, so we periodically recompute
                pass

            if current_score < best_score:
                best_score = current_score
                best_colors = current_colors.copy()
                no_improve = 0
        else:
            no_improve += 1

        # Cooling
        T *= cooling
        if T < min_T:
            # Restart with perturbation
            T = 1.5
            no_improve = 0
            # Perturb: flip several random positions
            for _ in range(8):
                k2 = int(rng.integers(0, n))
                d2 = get_delta(k2, current_colors)
                if d2 <= 0 or rng.random() < 0.1:
                    current_colors[k2] = 1 - current_colors[k2]
                    current_score += d2
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = current_colors.copy()

        steps += 1

        # Periodic full pressure recomputation
        if steps % 5000 == 0:
            pressure = compute_pressure(current_colors)

        # Stagnation detection: if no improvement for a long time, do a larger perturbation
        if no_improve > max_no_improve:
            # Large perturbation: flip 15-20 random positions
            num_flips = int(rng.integers(15, 21))
            for _ in range(num_flips):
                k2 = int(rng.integers(0, n))
                d2 = get_delta(k2, current_colors)
                current_colors[k2] = 1 - current_colors[k2]
                current_score += d2
                if current_score < best_score:
                    best_score = current_score
                    best_colors = current_colors.copy()
            T = 2.0
            no_improve = 0
            pressure = compute_pressure(current_colors)

    return best_colors.tolist()
# EVOLVE-BLOCK-END
