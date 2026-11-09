# EVOLVE-BLOCK-START
import time
import numpy as np


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        value >= 0.5 -> route the demand counter-clockwise
        value <  0.5 -> route the demand clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]
    D = len(demands)

    # Precompute, for each demand, the clockwise and counter-clockwise edge masks.
    cw_masks = []
    ccw_masks = []
    for (s, t, w) in demands:
        cw = np.zeros(n)
        ccw = np.zeros(n)
        if s == t:
            # degenerate: both paths empty
            pass
        else:
            # clockwise: edges s, s+1, ..., t-1 (mod n)
            i = s
            while i != t:
                cw[i] = 1.0
                i = (i + 1) % n
            # counter-clockwise: all other edges
            ccw = 1.0 - cw
        cw_masks.append(cw)
        ccw_masks.append(ccw)

    cw_masks = np.array(cw_masks)      # (D, n)
    ccw_masks = np.array(ccw_masks)    # (D, n)
    weights = np.array([w for (_, _, w) in demands], dtype=float)  # (D,)

    # For a binary direction vector x (0=cw, 1=ccw), edge load:
    #   load_e = sum_d w_d * ( (1 - x_d) * cw_mask[d,e] + x_d * ccw_mask[d,e] )
    # We can express load = cw_load + x @ (ccw - cw) * weights
    # where cw_load_e = sum_d w_d * cw_mask[d,e]
    cw_load = (cw_masks * weights[:, None]).sum(axis=0)  # (n,)
    diff = (ccw_masks - cw_masks) * weights[:, None]     # (D, n)

    def binary_loads(x):
        # x is binary (0/1) of shape (D,)
        return cw_load + x @ diff

    def max_load(x):
        return float(np.max(binary_loads(x)))

    # Soft surrogate using log-sum-exp to smooth the max
    def soft_max_load(x, beta=20.0):
        loads = binary_loads(x)
        m = np.max(loads)
        # log-sum-exp for stability
        z = loads - m
        return m + np.log(np.sum(np.exp(beta * z))) / beta

    # Objective to minimize: soft max load
    def objective(x):
        return soft_max_load(x, beta=25.0)

    # Start from all-clockwise
    x0 = np.zeros(D)
    best_x = x0.copy()
    best_val = objective(x0)

    # Deadline guard
    t_start = time.time()
    deadline = 540.0  # safety margin under 630s

    # Try differential evolution if available and time permits
    try:
        from scipy.optimize import differential_evolution
        bounds = [(0.0, 1.0)] * D
        # Moderate population to stay well under deadline
        result = differential_evolution(
            objective,
            bounds,
            maxiter=200,
            popsize=12,
            tol=1e-4,
            seed=42,
            init="latinhypercube",
            polish=True,
            callback=lambda xk, fk: (
                setattr(ring_routing, "_time", time.time())
                or (time.time() - t_start > deadline)
            ),
        )
        if result.x is not None:
            # Round to binary
            x_round = np.where(result.x >= 0.5, 1.0, 0.0)
            v = max_load(x_round)
            if v < best_val:
                best_val = v
                best_x = x_round.copy()
    except Exception:
        pass

    # Local search polish: flip single bits and accept improvements
    def improve_local(x):
        improved = True
        cur = max_load(x)
        while improved and (time.time() - t_start) < deadline:
            improved = False
            for d in range(D):
                x2 = x.copy()
                x2[d] = 1.0 - x2[d]
                v2 = max_load(x2)
                if v2 < cur - 1e-12:
                    x = x2
                    cur = v2
                    improved = True
        return x, cur

    # Run local search from the best continuous solution rounded, and from all-clockwise
    for start in [best_x, np.zeros(D), np.ones(D)]:
        if (time.time() - t_start) >= deadline:
            break
        x_ls, v_ls = improve_local(start.copy())
        if v_ls < best_val:
            best_val = v_ls
            best_x = x_ls.copy()

    # Pairwise swap local search for stronger moves
    def improve_local_pairs(x):
        improved = True
        cur = max_load(x)
        while improved and (time.time() - t_start) < deadline:
            improved = False
            for d1 in range(D):
                for d2 in range(d1 + 1, D):
                    x2 = x.copy()
                    x2[d1] = 1.0 - x2[d1]
                    x2[d2] = 1.0 - x2[d2]
                    v2 = max_load(x2)
                    if v2 < cur - 1e-12:
                        x = x2
                        cur = v2
                        improved = True
        return x, cur

    x_ls2, v_ls2 = improve_local_pairs(best_x.copy())
    if v_ls2 < best_val:
        best_val = v_ls2
        best_x = x_ls2.copy()

    # Final binary result
    routing = np.where(best_x >= 0.5, 1.0, 0.0)
    return routing


# EVOLVE-BLOCK-END
