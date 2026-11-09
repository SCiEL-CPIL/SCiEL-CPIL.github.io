import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill
    n = 256

    # Precompute all 4-term APs as index tuples (0-indexed)
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
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
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append(idx)
        pos_aps[i1].append(idx)
        pos_aps[i2].append(idx)
        pos_aps[i3].append(idx)

    # Convert to numpy arrays for faster access
    pos_aps_arr = [np.array(x, dtype=np.int32) if x else np.array([], dtype=np.int32) for x in pos_aps]

    def count_mono(colours):
        """Count monochromatic 4-term APs."""
        cols = np.array(colours, dtype=np.int8)
        # For each AP, check if all 4 positions have same colour
        c0 = cols[aps_arr[:, 0]]
        c1 = cols[aps_arr[:, 1]]
        c2 = cols[aps_arr[:, 2]]
        c3 = cols[aps_arr[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(mono))

    def get_delta(colours, pos):
        """Compute change in mono count if we flip position pos."""
        cols = np.array(colours, dtype=np.int8)
        current = cols[pos]
        new_val = 1 - current

        # Get all APs containing pos
        ap_indices = pos_aps_arr[pos]
        if len(ap_indices) == 0:
            return 0

        # For each AP, determine if it's currently mono and if it would be mono after flip
        c0 = cols[aps_arr[ap_indices, 0]]
        c1 = cols[aps_arr[ap_indices, 1]]
        c2 = cols[aps_arr[ap_indices, 2]]
        c3 = cols[aps_arr[ap_indices, 3]]

        # Currently mono?
        is_mono_now = (c0 == c1) & (c1 == c2) & (c2 == c3)

        # After flip: replace the value at pos with new_val
        # We need to check which position in the AP is pos
        # Build arrays of the four positions for these APs
        a0 = aps_arr[ap_indices, 0]
        a1 = aps_arr[ap_indices, 1]
        a2 = aps_arr[ap_indices, 2]
        a3 = aps_arr[ap_indices, 3]

        # Create copies of colour vectors with pos flipped
        # For efficiency, compute new colours for each of the 4 slots
        # Slot 0
        new_c0 = c0.copy()
        mask0 = (a0 == pos)
        new_c0[mask0] = new_val
        mono0 = (new_c0 == c1) & (c1 == c2) & (c2 == c3)

        # Slot 1
        new_c1 = c1.copy()
        mask1 = (a1 == pos)
        new_c1[mask1] = new_val
        mono1 = (c0 == new_c1) & (new_c1 == c2) & (c2 == c3)

        # Slot 2
        new_c2 = c2.copy()
        mask2 = (a2 == pos)
        new_c2[mask2] = new_val
        mono2 = (c0 == c1) & (c1 == new_c2) & (new_c2 == c3)

        # Slot 3
        new_c3 = c3.copy()
        mask3 = (a3 == pos)
        new_c3[mask3] = new_val
        mono3 = (c0 == c1) & (c1 == c2) & (c2 == new_c3)

        # Count mono after flip
        is_mono_after = mono0 | mono1 | mono2 | mono3

        delta = int(np.sum(is_mono_after)) - int(np.sum(is_mono_now))
        return delta

    def thue_morse(n):
        """Generate Thue-Morse sequence."""
        tm = [0]
        for i in range(n - 1):
            tm.append(1 - tm[i // 2])
        return tm

    # Generate initial candidates
    candidates = []

    # Thue-Morse
    tm = thue_morse(n)
    candidates.append(tm)

    # Thue-Morse flipped
    candidates.append([1 - x for x in tm])

    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(8):
        candidates.append(list(rng.randint(0, 2, n)))

    # Evaluate all candidates
    best_colours = None
    best_count = float('inf')

    for cand in candidates:
        cnt = count_mono(cand)
        if cnt < best_count:
            best_count = cnt
            best_colours = cand[:]

    # Local search with simulated annealing on best candidates
    # Sort candidates by count and take top 3
    scored = [(count_mono(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0])
    top_candidates = [c for _, c in scored[:3]]

    for init_colours in top_candidates:
        if time.time() - start_time > deadline - 10:
            break

        current = init_colours[:]
        current_count = count_mono(current)
        best_local = current[:]
        best_local_count = current_count

        # Simulated annealing
        T = 5.0
        T_min = 0.1
        cooling = 0.97
        max_iters = 5000

        for it in range(max_iters):
            if time.time() - start_time > deadline - 5:
                break

            # Pick random position to flip
            pos = rng.randint(0, n)
            delta = get_delta(current, pos)

            if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
                current[pos] = 1 - current[pos]
                current_count += delta

                if current_count < best_local_count:
                    best_local_count = current_count
                    best_local = current[:]

            T *= cooling

        if best_local_count < best_count:
            best_count = best_local_count
            best_colours = best_local[:]

    # Final polish: greedy local search on best
    if time.time() - start_time < deadline - 5:
        current = best_colours[:]
        current_count = count_mono(current)
        improved = True
        while improved and time.time() - start_time < deadline - 3:
            improved = False
            for pos in range(n):
                delta = get_delta(current, pos)
                if delta < 0:
                    current[pos] = 1 - current[pos]
                    current_count += delta
                    improved = True
        if current_count < best_count:
            best_count = current_count
            best_colours = current

    return best_colours
