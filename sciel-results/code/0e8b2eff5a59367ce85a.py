import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill
    n = 256

    # Precompute all 4-term APs as index tuples (0-indexed)
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
    aps = []
    for d in range(1, 86):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))

    aps_arr = np.array(aps, dtype=np.int32)  # shape (M, 4)
    M = len(aps_arr)

    # Build per-position list of AP indices for delta computation
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)
    pos_aps_arr = [np.array(lst, dtype=np.int32) for lst in pos_aps]

    # Thue-Morse initialization: c[i] = popcount(i) % 2
    c = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def count_mono(col):
        # For each AP, check if all 4 positions have same color
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        mono = (a0 == a1) & (a1 == a2) & (a2 == a3)
        return int(np.sum(mono))

    best_count = count_mono(c)
    best_c = c.copy()

    def get_delta(pos, col):
        # Compute change in monochromatic count if we flip position `pos`
        # An AP is mono iff all 4 positions have same color.
        # Flipping pos affects only APs containing pos.
        # For each such AP, check before/after mono status.
        ap_indices = pos_aps_arr[pos]
        if len(ap_indices) == 0:
            return 0
        # Get the 4 positions of each affected AP
        affected = aps_arr[ap_indices]  # shape (K, 4)
        # Current colors at all 4 positions
        colors = col[affected]  # shape (K, 4)
        # Check which are currently mono
        cur_mono = (colors[:, 0] == colors[:, 1]) & (colors[:, 1] == colors[:, 2]) & (colors[:, 2] == colors[:, 3])
        cur_mono_count = int(np.sum(cur_mono))
        # After flip: the position `pos` in each AP gets flipped
        # Find which column in each AP corresponds to `pos`
        # We need to flip the specific element
        # Build a flipped copy
        flipped = colors.copy()
        for row_idx in range(len(ap_indices)):
            for col_idx in range(4):
                if affected[row_idx, col_idx] == pos:
                    flipped[row_idx, col_idx] = 1 - flipped[row_idx, col_idx]
                    break
        new_mono = (flipped[:, 0] == flipped[:, 1]) & (flipped[:, 1] == flipped[:, 2]) & (flipped[:, 2] == flipped[:, 3])
        new_mono_count = int(np.sum(new_mono))
        return new_mono_count - cur_mono_count

    # Local search: iterate through positions, flip if beneficial
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for pos in range(n):
            if (time.time() - start_time) > deadline - 5.0:
                break
            d = get_delta(pos, c)
            if d < 0:
                c[pos] = 1 - c[pos]
                best_count += d
                best_c = c.copy()
                improved = True
            elif d == 0:
                # Try flip anyway to escape local minima (with rollback)
                c[pos] = 1 - c[pos]
                # Check if this leads to further improvement in a quick pass
                # For simplicity, just accept neutral moves occasionally
                # Actually, let's not accept neutral moves to keep it simple
                c[pos] = 1 - c[pos]  # revert

    # Final check: if best_c is better, use it
    if count_mono(best_c) < best_count:
        best_count = count_mono(best_c)

    return best_c.tolist()
