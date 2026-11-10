# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy expansion of the excluded set.
    """
    p = 7
    start_time = time.time()
    deadline = 600.0  # seconds

    # Smallest quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break

    # Precompute all points as indices 0..2400
    # Point index = ((xa * p + xb) * p + ya) * p + yb
    def point_index(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Precompute point coordinates
    points_coords = []
    for idx in range(p**4):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        points_coords.append((xa, xb, ya, yb))

    # Precompute inv for all (c, d) != (0, 0)
    inv = {}
    for c in range(p):
        for d in range(p):
            if c == 0 and d == 0:
                continue
            det = (c * c - d * d * k) % p
            det_inv = pow(det, p - 2, p)  # Fermat's little theorem
            inv[(c, d)] = ((c * det_inv) % p, (-d * det_inv) % p)

    # Precompute dir_of_pair as a 2D array
    dir_of_pair = np.full((p**4, p**4), -1, dtype=np.int8)

    for i in range(p**4):
        xa1, xb1, ya1, yb1 = points_coords[i]
        for j in range(p**4):
            if i == j:
                continue
            xa2, xb2, ya2, yb2 = points_coords[j]
            dx_a = (xa2 - xa1) % p
            dx_b = (xb2 - xb1) % p
            dy_a = (ya2 - ya1) % p
            dy_b = (yb2 - yb1) % p

            if dx_a == 0 and dx_b == 0:
                dir_of_pair[i, j] = 0  # vertical direction
            else:
                inv_a, inv_b = inv[(dx_a, dx_b)]
                ma = (dy_a * inv_a + dy_b * inv_b * k) % p
                mb = (dy_a * inv_b + dy_b * inv_a) % p
                dir_of_pair[i, j] = 1 + ma * p + mb

    # Initialize S = base_excluded
    y_special = set()
    for u in range(1, p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        idx = point_index(xa, xb, ya, yb)
                        S.add(idx)

    # Initialize count[p][d]
    count = np.zeros((p**4, 50), dtype=np.int8)

    # For each q in S, for each p != q, d = dir_of_pair[p][q], count[p][d] += 1
    S_list = list(S)
    for q in S_list:
        for p_idx in range(p**4):
            if p_idx == q:
                continue
            d = dir_of_pair[p_idx, q]
            if d >= 0:
                count[p_idx, d] += 1

    # Compute good_dirs
    good_dirs = np.sum(count == 0, axis=1)

    # Greedily add points
    candidates = [idx for idx in range(p**4) if idx not in S]

    for q in candidates:
        if time.time() - start_time > deadline - 30:
            break

        d_q = dir_of_pair[:, q].copy()
        d_q[q] = -1  # mark self

        # Get count[p][d_q[p]] for all p
        valid_mask = d_q >= 0
        count_at_d = np.zeros(p**4, dtype=np.int8)
        count_at_d[valid_mask] = count[np.where(valid_mask)[0], d_q[valid_mask]]

        # Check if any p has good_dirs[p] == 1 and count[p][d_q[p]] == 0
        # If so, adding q would make good_dirs[p] == 0, which is invalid
        bad_mask = (good_dirs == 1) & (count_at_d == 0) & valid_mask
        if np.any(bad_mask):
            continue  # Cannot add q

        # Safe to add q
        S.add(q)
        # Update count and good_dirs
        for p_idx in np.where(valid_mask)[0]:
            d = d_q[p_idx]
            if count[p_idx, d] == 0:
                good_dirs[p_idx] -= 1
            count[p_idx, d] += 1

    # Build the Nikodym set (complement of S)
    points = []
    for idx in range(p**4):
        if idx not in S:
            xa, xb, ya, yb = points_coords[idx]
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
