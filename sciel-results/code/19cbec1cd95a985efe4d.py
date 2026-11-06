"""Rewritten program: minimize monochromatic 4-AP count via SA + polish."""
import time
import random
import numpy as np


def _count_mono_4ap(colors):
    """Count monochromatic 4-term APs in a 2-coloring of 1..256."""
    n = 256
    count = 0
    c = np.asarray(colors, dtype=np.int8)
    for a in range(n):
        for d in range(1, n):
            a3 = a + 3 * d
            if a3 >= n:
                break
            if c[a] == c[a + d] == c[a + 2 * d] == c[a3]:
                count += 1
    return count


def construct():
    """
    Minimize the number of monochromatic 4-term arithmetic progressions
    in a 2-coloring of [1, 256].
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    N = 256
    rng = random.Random(42)

    # Precompute all 4-APs as (a, a+d, a+2d, a+3d) zero-indexed
    aps = []
    for a in range(N):
        for d in range(1, N):
            a3 = a + 3 * d
            if a3 >= N:
                break
            aps.append((a, a + d, a + 2 * d, a3))

    num_aps = len(aps)

    # For each position, precompute which APs it participates in
    # aps_by_pos[i] = list of (ap_index, role) where role in {0,1,2,3}
    aps_by_pos = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        aps_by_pos[a].append((idx, 0))
        aps_by_pos[b].append((idx, 1))
        aps_by_pos[c].append((idx, 2))
        aps_by_pos[d].append((idx, 3))

    # Precompute for each AP the tuple of positions for fast access
    ap_positions = [(a, b, c, d) for (a, b, c, d) in aps]

    def fast_count(colors):
        """Fast count using precomputed AP list."""
        count = 0
        for (a, b, c, d) in ap_positions:
            if colors[a] == colors[b] == colors[c] == colors[d]:
                count += 1
        return count

    def delta_flip(colors, pos):
        """
        Compute change in mono-4AP count if we flip colors[pos].
        """
        delta = 0
        for (ap_idx, role) in aps_by_pos[pos]:
            a, b, c, d = ap_positions[ap_idx]
            # Get the other three positions' colors
            if role == 0:
                others = (colors[b], colors[c], colors[d])
            elif role == 1:
                others = (colors[a], colors[c], colors[d])
            elif role == 2:
                others = (colors[a], colors[b], colors[d])
            else:
                others = (colors[a], colors[b], colors[c])
            
            cur_color = colors[pos]
            new_color = 1 - cur_color
            
            # Check if current state is mono
            cur_mono = (cur_color == others[0] == others[1] == others[2])
            # Check if new state is mono
            new_mono = (new_color == others[0] == others[1] == others[2])
            
            if new_mono and not cur_mono:
                delta += 1
            elif cur_mono and not new_mono:
                delta -= 1
        return delta

    def polish(colors, count, max_passes=100):
        """Greedy polish: flip any position that reduces count."""
        for _ in range(max_passes):
            improved = False
            for pos in range(N):
                delta = delta_flip(colors, pos)
                if delta < 0:
                    colors[pos] = 1 - colors[pos]
                    count += delta
                    improved = True
            if not improved:
                break
        return count

    def simulated_annealing(colors, count, steps, t_start, t_end):
        """Run simulated annealing for given steps with geometric cooling."""
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            # Geometric cooling
            t = t_start * (t_end / t_start) ** (step / max(steps - 1, 1))
            
            pos = rng.randint(0, N - 1)
            delta = delta_flip(colors, pos)
            
            if delta < 0:
                colors[pos] = 1 - colors[pos]
                count += delta
            else:
                # Accept with probability exp(-delta / t)
                if rng.random() < np.exp(-delta / t):
                    colors[pos] = 1 - colors[pos]
                    count += delta
        
        return count

    # Diverse starting points
    def make_random():
        return np.array([rng.randint(0, 1) for _ in range(N)], dtype=np.int8)

    def make_thue_morse():
        # Thue-Morse sequence: t(n) = popcount(n) % 2
        return np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int8)

    def make_periodic(p):
        return np.array([i % p % 2 for i in range(N)], dtype=np.int8)

    starts = [
        make_random(),
        make_thue_morse(),
        make_periodic(2),
        make_periodic(3),
    ]

    best_colors = None
    best_count = float('inf')

    for start_idx, init in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        current = init.copy()
        current_count = fast_count(current)
        
        # Run SA with geometric cooling
        # Multiple SA runs with different parameters
        sa_configs = [
            (50000, 5.0, 0.05),
            (30000, 3.0, 0.03),
            (20000, 2.0, 0.02),
        ]
        
        for (steps, t_start, t_end) in sa_configs:
            if time.time() > deadline - 10:
                break
            
            current = current.copy()
            current_count = simulated_annealing(current, current_count, steps, t_start, t_end)
            
            # Polish
            current_count = polish(current, current_count, max_passes=50)
            
            if current_count < best_count:
                best_count = current_count
                best_colors = current.copy()
        
        # Try a few random restarts from this start
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            
            current = best_colors.copy() if best_colors is not None else init.copy()
            current_count = fast_count(current)
            
            # Perturb
            num_flips = rng.randint(2, 8)
            for _ in range(num_flips):
                pos = rng.randint(0, N - 1)
                current[pos] = 1 - current[pos]
            current_count = fast_count(current)
            
            # SA + polish
            current_count = simulated_annealing(current, current_count, 20000, 2.0, 0.05)
            current_count = polish(current, current_count, max_passes=50)
            
            if current_count < best_count:
                best_count = current_count
                best_colors = current.copy()

    # Final polish on best
    if best_colors is not None:
        best_count = polish(best_colors, best_count, max_passes=200)

    return [int(x) for x in best_colors]
