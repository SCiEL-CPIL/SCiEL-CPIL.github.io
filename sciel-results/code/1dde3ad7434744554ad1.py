import time
import random
import numpy as np

def construct():
    n = 256
    # Keep a safety margin: evaluator kills at 150s, we target finishing by 110s.
    deadline = time.time() + 110.0
    max_d = (n - 3) // 3

    # Precompute all (a, d) pairs for 4-APs to enable vectorized counting
    # Represent each AP by its four indices; we'll store them as a (M,4) int array
    _ap_list = []
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            _ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    AP = np.array(_ap_list, dtype=np.int32)  # shape (M, 4)
    M = AP.shape[0]

    def count_mono(colors):
        # colors: 1-D array of 0/1 of length n
        # For each AP, check if all four entries are equal.
        # Equal iff (c0==c1) & (c1==c2) & (c2==c3)
        c = colors[AP]  # (M,4)
        eq = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.count_nonzero(eq))

    def delta_flip(colors, x):
        """Compute change in mono-AP count if we flip color at position x."""
        # For each AP that contains x, determine if it was mono and whether it becomes mono.
        # An AP (a, a+d, a+2d, a+3d) contains x if x == a + k*d for k in {0,1,2,3}.
        # We iterate over d and k, same as before, but use vectorized-style logic.
        # Given the small n=256, a Python loop is acceptable; we keep it but tighten.
        delta = 0
        c = colors  # alias
        for d in range(1, max_d + 1):
            # x must be a + k*d for some k in 0..3, with 0 <= a and a+3d < n
            # => a = x - k*d, require 0 <= a and a+3d < n
            for k in range(4):
                a = x - k * d
                if a < 0 or a + 3 * d >= n:
                    continue
                v0 = c[a]
                v1 = c[a + d]
                v2 = c[a + 2 * d]
                v3 = c[a + 3 * d]
                old_mono = (v0 == v1) and (v1 == v2) and (v2 == v3)
                # After flipping position x (which is a+k*d):
                if k == 0:
                    nv0 = 1 - v0
                    new_mono = (nv0 == v1) and (v1 == v2) and (v2 == v3)
                elif k == 1:
                    nv1 = 1 - v1
                    new_mono = (v0 == nv1) and (nv1 == v2) and (v2 == v3)
                elif k == 2:
                    nv2 = 1 - v2
                    new_mono = (v0 == v1) and (v1 == nv2) and (nv2 == v3)
                else:
                    nv3 = 1 - v3
                    new_mono = (v0 == v1) and (v1 == v2) and (v2 == nv3)
                delta += (1 if new_mono else 0) - (1 if old_mono else 0)
        return delta

    # Build a few structured initial colorings (multi-start within the same design family)
    def thue_morse():
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def block2():
        # Alternate colours in blocks of two (from inspiration)
        return np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)

    def block4():
        # Alternate colours in blocks of four
        return np.array([(i // 4) % 2 for i in range(n)], dtype=np.int8)

    def random_init():
        return np.random.randint(0, 2, size=n).astype(np.int8)

    seeds = [thue_morse(), block2(), block4(), random_init()]

    best_cost = float('inf')
    best_colors = None

    # Allocate time budget: use ~70% for SA multi-start, ~30% for final polish
    total_budget = 110.0
    sa_budget = total_budget * 0.7
    sa_deadline = time.time() + sa_budget

    for colors in seeds:
        if time.time() >= sa_deadline:
            break
        current_cost = count_mono(colors)
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = colors.copy()

        # Run SA from this seed
        T = 1.0
        cool_rate = 0.995
        min_T = 0.01
        local_best_cost = current_cost
        local_best_colors = colors.copy()

        while time.time() < sa_deadline:
            x = random.randint(0, n - 1)
            d_cost = delta_flip(colors, x)

            if d_cost < 0:
                colors[x] = 1 - colors[x]
                current_cost += d_cost
                if current_cost < local_best_cost:
                    local_best_cost = current_cost
                    local_best_colors = colors.copy()
            else:
                if T > min_T:
                    p = np.exp(-d_cost / T)
                    if random.random() < p:
                        colors[x] = 1 - colors[x]
                        current_cost += d_cost
                        if current_cost < local_best_cost:
                            local_best_cost = current_cost
                            local_best_colors = colors.copy()

            T *= cool_rate

        if local_best_cost < best_cost:
            best_cost = local_best_cost
            best_colors = local_best_colors.copy()

    # Final polish: coordinate descent (only accept improving flips) until convergence
    # or until deadline. This is an exact local solve on the original objective.
    polish_deadline = time.time() + (total_budget - sa_budget)
    if best_colors is not None and time.time() < polish_deadline:
        colors = best_colors.copy()
        current_cost = best_cost
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            for x in range(n):
                if time.time() >= polish_deadline:
                    break
                d_cost = delta_flip(colors, x)
                if d_cost < 0:
                    colors[x] = 1 - colors[x]
                    current_cost += d_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()

    return best_colors.tolist()
