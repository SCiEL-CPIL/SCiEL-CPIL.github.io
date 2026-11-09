"""Rewritten program: Simulated Annealing on 4-term AP minimization."""
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
    # pos_aps[i] = list of AP indices where position i appears
    pos_aps = [[] for _ in range(n)]
    for idx in range(M):
        for pos in aps_arr[idx]:
            pos_aps[pos].append(idx)

    # Convert to arrays for vectorized operations
    pos_aps_arr = [np.array(p, dtype=np.int32) if p else np.array([], dtype=np.int32) for p in pos_aps]

    # Thue-Morse initialization: t(i) = popcount(i) % 2
    # This gives a balanced 2-coloring with known low AP density
    def thue_morse(i):
        # Number of 1-bits in i
        c = 0
        while i:
            c += i & 1
            i >>= 1
        return c % 2

    colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)

    # Count initial monochromatic APs
    def count_mono_vec(cols):
        # For each AP, check if all 4 positions have the same color
        c = cols[aps_arr]  # shape (M, 4)
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.sum(mono))

    current_mono = count_mono_vec(colors)
    best_colors = colors.copy()
    best_mono = current_mono

    # Simulated Annealing
    # Temperature schedule: geometric cooling
    T_start = 5.0
    T_end = 0.01
    # Number of iterations - we'll run as many as time allows
    # Each iteration: pick a random position, compute delta, accept/reject

    rng = np.random.RandomState(42)

    # Precompute: for each position, the AP indices it belongs to
    # Delta computation: when flipping position p, for each AP containing p,
    # check if the AP becomes mono or stops being mono

    # For speed, precompute for each AP the colors at other 3 positions
    # But colors change, so we need dynamic computation

    # Faster approach: maintain a mono_count array (M,) indicating which APs are mono
    # When flipping position p:
    #   For each AP idx in pos_aps[p]:
    #     old_state = mono_count[idx]
    #     new_state = check if all 4 positions now same color
    #     delta += (new_state - old_state)
    #   Update mono_count for affected APs

    # Let's use a vectorized approach per flip
    # mono_flag: boolean array of length M
    def compute_mono_flags(cols):
        c = cols[aps_arr]
        return (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])

    mono_flag = compute_mono_flags(colors)

    # SA loop
    # We'll do multiple passes. Each pass tries to flip each position once.
    # Use geometric cooling per pass.

    num_passes = 50  # will adjust based on time
    # Actually, let's just iterate until deadline

    T = T_start
    cooling_rate = 0.9995  # per iteration

    while time.time() - start_time < deadline:
        # Pick random position
        p = rng.randint(n)
        p_arr = np.array([p], dtype=np.int32)

        # Get AP indices containing p
        ap_indices = pos_aps_arr[p]
        if len(ap_indices) == 0:
            T *= cooling_rate
            continue

        # Current mono status of these APs
        old_mono = mono_flag[ap_indices]

        # Flip color at p
        colors[p] = 1 - colors[p]

        # New mono status: check if all 4 positions now same
        # For each AP in ap_indices, get the 4 colors
        new_colors = colors[aps_arr[ap_indices]]
        new_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                   (new_colors[:, 1] == new_colors[:, 2]) & \
                   (new_colors[:, 2] == new_colors[:, 3])

        delta = int(np.sum(new_mono) - np.sum(old_mono))

        # Accept or reject
        if delta <= 0 or rng.rand() < np.exp(-delta / T):
            # Accept: update mono_flag
            mono_flag[ap_indices] = new_mono
            current_mono += delta
            if current_mono < best_mono:
                best_mono = current_mono
                best_colors = colors.copy()
        else:
            # Reject: flip back
            colors[p] = 1 - colors[p]
            # mono_flag unchanged since we didn't accept

        T *= cooling_rate
        if T < T_end:
            T = T_end

    # Return best found
    return [int(c) for c in best_colors]
