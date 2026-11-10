import numpy as np

def ring_routing() -> np.ndarray:
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]

    # Precompute edge usage matrices
    # cw_mat[i, e] = 1 if edge e is used by demand i in CW direction
    # ccw_mat[i, e] = 1 if edge e is used by demand i in CCW direction
    cw_mat = np.zeros((D, n), dtype=np.float64)
    ccw_mat = np.zeros((D, n), dtype=np.float64)
    weights = np.zeros(D, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        d_cw = (t - s) % n
        if d_cw == 0:
            # Source and target are the same? Not possible in this problem set.
            # But handle just in case.
            pass
        else:
            # CW edges: s, s+1, ..., t-1
            for k in range(d_cw):
                e = (s + k) % n
                cw_mat[i, e] = 1.0
            # CCW edges: all other edges
            for e in range(n):
                if cw_mat[i, e] == 0.0:
                    ccw_mat[i, e] = 1.0

    # Base load if all demands go CW
    base_load = weights @ cw_mat # shape (n,)
    
    # Delta vector for flipping demand i from CW to CCW
    # delta[i] = w_i * (ccw_mat[i] - cw_mat[i])
    delta_mat = (ccw_mat - cw_mat) * weights[:, None]
    
    # Vectorized exhaustive search
    total_masks = 1 << D
    # Generate all masks
    masks = np.arange(total_masks, dtype=np.int32)
    # Generate bit matrix: shape (total_masks, D)
    # bit i of mask m is (m >> i) & 1
    # We want (total_masks, D) where column j is bit j of the mask
    # np.arange(D) gives [0, 1, ..., D-1]
    # masks[:, None] has shape (total_masks, 1)
    # (masks[:, None] >> np.arange(D)) has shape (total_masks, D)
    mask_bits = ((masks[:, None] >> np.arange(D)) & 1).astype(np.float64)
    
    # Compute loads for all masks
    # loads[m, e] = base_load[e] + sum_i (mask_bits[m, i] * delta_mat[i, e])
    # loads = base_load[None, :] + mask_bits @ delta_mat.T
    loads = base_load[None, :] + mask_bits @ delta_mat.T
    
    # Find the mask with the minimum maximum load
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx]
    
    # Reconstruct the routing vector from the best_mask
    out = np.zeros(D, dtype=np.float64)
    for i in range(D):
        if (best_mask >> i) & 1:
            out[i] = 1.0
        else:
            out[i] = 0.0
            
    return out
