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

    best_S = set()
    best_size = 0
    
    # Run multiple greedy restarts
    num_restarts = 200
    
    for _ in range(num_restarts):
        if time.time() - start_time > deadline - 10:
            break
            
        S = set()
        count = np.zeros((p**4, 50), dtype=np.int8)
        good_dirs = np.full(p**4, 50, dtype=np.int8)
        
        candidates = list(range(p**4))
        np.random.shuffle(candidates)
        
        for q in candidates:
            d_q = dir_of_pair[:, q].copy()
            d_q[q] = -1  # mark self
            
            valid_mask = d_q >= 0
            count_at_d = np.zeros(p**4, dtype=np.int8)
            count_at_d[valid_mask] = count[np.where(valid_mask)[0], d_q[valid_mask]]
            
            bad_mask = (good_dirs == 1) & (count_at_d == 0) & valid_mask
            if np.any(bad_mask):
                continue
            
            S.add(q)
            for p_idx in np.where(valid_mask)[0]:
                d = d_q[p_idx]
                if count[p_idx, d] == 0:
                    good_dirs[p_idx] -= 1
                count[p_idx, d] += 1
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    S = best_S

    # Build the Nikodym set (complement of S)
    points = []
    for idx in range(p**4):
        if idx not in S:
            xa, xb, ya, yb = points_coords[idx]
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
