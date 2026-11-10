"""Optimized coloring for minimizing monochromatic 4-term APs in [1,256]."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # Store as 0-indexed tuples
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))

    total_aps = len(aps)

    # For fast delta computation, precompute for each position which APs it belongs to
    # and what the other 3 positions are
    # ap_list[i] = list of (ap_index, other1, other2, other3)
    ap_list = [[] for _ in range(N)]
    for idx, (i1, i2, i3, i4) in enumerate(aps):
        ap_list[i1].append((idx, i2, i3, i4))
        ap_list[i2].append((idx, i1, i3, i4))
        ap_list[i3].append((idx, i1, i2, i4))
        ap_list[i4].append((idx, i1, i2, i3))

    def count_mono(coloring):
        """Count monochromatic 4-term APs."""
        count = 0
        for i1, i2, i3, i4 in aps:
            c1 = coloring[i1]
            if c1 == coloring[i2] == coloring[i3] == coloring[i4]:
                count += 1
        return count

    def get_delta(coloring, pos):
        """
        Compute the change in monochromatic AP count if we flip position 'pos'.
        Returns (delta, new_count) where delta = new_count - old_count.
        """
        old_color = coloring[pos]
        new_color = 1 - old_color
        delta = 0
        for ap_idx, o1, o2, o3 in ap_list[pos]:
            c1 = coloring[o1]
            c2 = coloring[o2]
            c3 = coloring[o3]
            # Before flip: AP is monochromatic if all four match old_color
            if c1 == old_color and c2 == old_color and c3 == old_color:
                delta += 1  # was mono, now not
            # After flip: AP is monochromatic if all four match new_color
            if c1 == new_color and c2 == new_color and c3 == new_color:
                delta -= 1  # was not mono (or was mono of other color), now mono
        return delta

    def local_search(coloring, max_iters=2000):
        """Greedy local search: flip positions that reduce count."""
        coloring = coloring.copy()
        current_count = count_mono(coloring)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for pos in range(N):
                delta = get_delta(coloring, pos)
                if delta < 0:
                    coloring[pos] = 1 - coloring[pos]
                    current_count += delta
                    improved = True
        return coloring, current_count

    def simulated_annealing(coloring, max_iters=5000, t_start=1.0, t_end=0.01):
        """Simulated annealing with single-bit flips."""
        coloring = coloring.copy()
        current_count = count_mono(coloring)
        best_coloring = coloring.copy()
        best_count = current_count

        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Compute cooling schedule
            progress = it / max_iters
            temperature = t_start * (t_end / t_start) ** progress

            pos = random.randint(0, N - 1)
            delta = get_delta(coloring, pos)

            if delta < 0 or random.random() < np.exp(-delta / temperature):
                coloring[pos] = 1 - coloring[pos]
                current_count += delta
                if current_count < best_count:
                    best_count = current_count
                    best_coloring = coloring.copy()

        return best_coloring, best_count

    # Initialize with several diverse starting points
    def make_pattern(pattern_type):
        """Generate a base coloring pattern."""
        coloring = np.zeros(N, dtype=np.int8)
        if pattern_type == 0:
            # Alternating blocks of 2
            for i in range(N):
                coloring[i] = (i // 2) % 2
        elif pattern_type == 1:
            # Alternating blocks of 4
            for i in range(N):
                coloring[i] = (i // 4) % 2
        elif pattern_type == 2:
            # Random
            coloring = np.random.randint(0, 2, N).astype(np.int8)
        elif pattern_type == 3:
            # Alternating blocks of 8
            for i in range(N):
                coloring[i] = (i // 8) % 2
        elif pattern_type == 4:
            # Alternating blocks of 1
            for i in range(N):
                coloring[i] = i % 2
        return coloring

    # Try multiple patterns and keep the best
    best_overall = None
    best_overall_count = float('inf')

    random.seed(42)
    np.random.seed(42)

    patterns = [0, 1, 2, 3, 4]
    for pt in patterns:
        if time.time() > deadline:
            break

        coloring = make_pattern(pt)
        coloring_list = coloring.tolist()

        # Quick local search first
        ls_coloring, ls_count = local_search(coloring_list, max_iters=500)

        # Then simulated annealing
        sa_coloring, sa_count = simulated_annealing(ls_coloring, max_iters=3000, t_start=1.0, t_end=0.01)

        # Final local search polish
        final_coloring, final_count = local_search(sa_coloring, max_iters=1000)

        if final_count < best_overall_count:
            best_overall_count = final_count
            best_overall = final_coloring

    # If we have time, try a few more random restarts
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 10:
        attempts += 1
        coloring = np.random.randint(0, 2, N).astype(np.int8).tolist()
        ls_coloring, ls_count = local_search(coloring, max_iters=500)
        sa_coloring, sa_count = simulated_annealing(ls_coloring, max_iters=3000, t_start=1.0, t_end=0.01)
        final_coloring, final_count = local_search(sa_coloring, max_iters=1000)
        if final_count < best_overall_count:
            best_overall_count = final_count
            best_overall = final_coloring

    # Final verification
    if best_overall is None:
        best_overall = [(i // 2) % 2 for i in range(N)]

    return best_overall
# EVOLVE-BLOCK-END
