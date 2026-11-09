"""Rewritten program: Multi-start SA with greedy polish for 4-term AP minimization."""
import time
import numpy as np


def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s kill

    n = 256

    # Precompute all 4-term APs: a, a+d, a+2d, a+3d (1-indexed), d>=1, a+3d<=n
    # Store as 0-indexed tuples
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))

    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)

    # Build per-position list of AP indices
    pos_aps = [[] for _ in range(n)]
    for idx in range(M):
        for pos in aps_arr[idx]:
            pos_aps[pos].append(idx)

    pos_aps_arr = [np.array(p, dtype=np.int32) if p else np.array([], dtype=np.int32) for p in pos_aps]

    def count_mono_vec(cols):
        c = cols[aps_arr]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.sum(mono))

    def compute_mono_flags(cols):
        c = cols[aps_arr]
        return (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])

    def greedy_polish(colors, mono_flag, max_iters=20000):
        """Iteratively flip positions that reduce mono count, until no improvement or budget exhausted."""
        current_mono = int(np.sum(mono_flag))
        improved = True
        iters = 0
        while improved and iters < max_iters and (time.time() - start_time) < deadline - 5.0:
            improved = False
            iters += 1
            # Try flipping each position; accept if delta < 0
            for p in range(n):
                ap_indices = pos_aps_arr[p]
                if len(ap_indices) == 0:
                    continue
                old_mono = mono_flag[ap_indices]
                colors[p] = 1 - colors[p]
                new_colors = colors[aps_arr[ap_indices]]
                new_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                           (new_colors[:, 1] == new_colors[:, 2]) & \
                           (new_colors[:, 2] == new_colors[:, 3])
                delta = int(np.sum(new_mono) - np.sum(old_mono))
                if delta < 0:
                    mono_flag[ap_indices] = new_mono
                    current_mono += delta
                    improved = True
                else:
                    colors[p] = 1 - colors[p]
        return current_mono, mono_flag

    def sa_search(colors, mono_flag, T_start, T_end, cooling_rate, time_budget):
        """Run SA from given state. Returns best colors and mono count."""
        rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
        current_mono = int(np.sum(mono_flag))
        best_mono = current_mono
        best_colors = colors.copy()
        best_flag = mono_flag.copy()
        T = T_start
        end_time = start_time + time_budget

        while (time.time() - start_time) < end_time:
            p = rng.randint(n)
            ap_indices = pos_aps_arr[p]
            if len(ap_indices) == 0:
                T *= cooling_rate
                continue

            old_mono = mono_flag[ap_indices]
            colors[p] = 1 - colors[p]
            new_colors = colors[aps_arr[ap_indices]]
            new_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                       (new_colors[:, 1] == new_colors[:, 2]) & \
                       (new_colors[:, 2] == new_colors[:, 3])
            delta = int(np.sum(new_mono) - np.sum(old_mono))

            if delta <= 0 or rng.rand() < np.exp(-delta / T):
                mono_flag[ap_indices] = new_mono
                current_mono += delta
                if current_mono < best_mono:
                    best_mono = current_mono
                    best_colors = colors.copy()
                    best_flag = mono_flag.copy()
            else:
                colors[p] = 1 - colors[p]

            T *= cooling_rate
            if T < T_end:
                T = T_end

        return best_colors, best_mono, best_flag

    def thue_morse(i):
        c = 0
        while i:
            c += i & 1
            i >>= 1
        return c % 2

    # Multi-start: Thue-Morse + several random starts
    best_overall_mono = float('inf')
    best_overall_colors = None

    # Start 1: Thue-Morse
    tm_colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    tm_flag = compute_mono_flags(tm_colors)
    tm_mono = int(np.sum(tm_flag))

    # Greedy polish first
    tm_mono, tm_flag = greedy_polish(tm_colors.copy(), tm_flag.copy())
    if tm_mono < best_overall_mono:
        best_overall_mono = tm_mono
        best_overall_colors = tm_colors.copy()

    # SA from Thue-Morse
    sa_colors, sa_mono, sa_flag = sa_search(
        tm_colors.copy(), tm_flag.copy(),
        T_start=3.0, T_end=0.05, cooling_rate=0.9998,
        time_budget=25.0
    )
    if sa_mono < best_overall_mono:
        best_overall_mono = sa_mono
        best_overall_colors = sa_colors.copy()

    # Further polish
    sa_mono, sa_flag = greedy_polish(sa_colors.copy(), sa_flag.copy())
    if sa_mono < best_overall_mono:
        best_overall_mono = sa_mono
        best_overall_colors = sa_colors.copy()

    # Additional random starts if time permits
    rng_master = np.random.RandomState(12345)
    num_random_starts = 3
    time_per_start = (deadline - 5.0 - (time.time() - start_time)) / (num_random_starts + 1)
    if time_per_start > 5.0:
        for rs in range(num_random_starts):
            if (time.time() - start_time) > deadline - 10.0:
                break
            rand_colors = rng_master.randint(0, 2, size=n).astype(np.int8)
            rand_flag = compute_mono_flags(rand_colors)
            rand_mono = int(np.sum(rand_flag))

            # Quick greedy polish
            rand_mono, rand_flag = greedy_polish(rand_colors.copy(), rand_flag.copy(), max_iters=5000)
            if rand_mono < best_overall_mono:
                best_overall_mono = rand_mono
                best_overall_colors = rand_colors.copy()

            # SA from this start
            remaining = deadline - 10.0 - (time.time() - start_time)
            if remaining > 5.0:
                sa_c, sa_m, sa_f = sa_search(
                    rand_colors.copy(), rand_flag.copy(),
                    T_start=2.0, T_end=0.03, cooling_rate=0.9999,
                    time_budget=min(remaining * 0.7, 20.0)
                )
                if sa_m < best_overall_mono:
                    best_overall_mono = sa_m
                    best_overall_colors = sa_c.copy()

                # Final polish
                sa_m, sa_f = greedy_polish(sa_c.copy(), sa_f.copy())
                if sa_m < best_overall_mono:
                    best_overall_mono = sa_m
                    best_overall_colors = sa_c.copy()

    # Final comprehensive greedy polish on the best solution
    final_colors = best_overall_colors.copy()
    final_flag = compute_mono_flags(final_colors)
    final_mono, final_flag = greedy_polish(final_colors, final_flag, max_iters=50000)
    if final_mono < best_overall_mono:
        best_overall_mono = final_mono
        best_overall_colors = final_colors.copy()

    return [int(c) for c in best_overall_colors]
