# EVOLVE-BLOCK-START
import time
import numpy as np


def ring_routing() -> np.ndarray:
    """
    Route 15 demands on a 10-node ring to minimize the maximum edge load.
    Returns an array of shape (15,) where entry >= 0.5 means CCW, else CW.
    """
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]

    # Precompute the set of edges used by each demand in each direction.
    # CW from s to t uses edges s, s+1, ..., t-1 (mod n). Edge e joins e and (e+1)%n.
    cw_edges = []
    ccw_edges = []
    for (s, t, w) in demands:
        d_cw = (t - s) % n
        if d_cw == 0:
            cw_edges.append(frozenset())
            ccw_edges.append(frozenset())
        else:
            cw_set = frozenset(((s + k) % n) for k in range(d_cw))
            ccw_set = frozenset(((s + k) % n) for k in range(n - d_cw))
            cw_edges.append(cw_set)
            ccw_edges.append(ccw_set)

    weights = np.array([w for (_, _, w) in demands], dtype=np.float64)
    # For each demand and each direction, a vector of length n: 1 if edge on path else 0.
    cw_mat = np.zeros((D, n), dtype=np.float64)
    ccw_mat = np.zeros((D, n), dtype=np.float64)
    for i in range(D):
        for e in cw_edges[i]:
            cw_mat[i, e] = 1.0
        for e in ccw_edges[i]:
            ccw_mat[i, e] = 1.0

    # A routing is a binary vector r of length D: 1 = CCW, 0 = CW.
    # Edge load = sum_i w_i * M_i[e] where M_i = ccw_mat[i] if r_i else cw_mat[i].
    # Build base = sum w_i * cw_mat[i] (all CW). Then flipping i to CCW adds
    # w_i * (ccw_mat[i] - cw_mat[i]).
    base_load = weights @ cw_mat  # shape (n,)
    delta = (ccw_mat - cw_mat) * weights[:, None]  # shape (D, n)

    def compute_load(r):
        """
        Compute the load vector for a given routing r.
        r: 0/1 array length D
        """
        if np.any(r):
            load = base_load + delta[r == 1].sum(axis=0)
        else:
            load = base_load
        return load

    def max_load(r):
        """
        Return the maximum edge load for a given routing r.
        """
        load = compute_load(r)
        return float(np.max(load))

    start = time.time()
    deadline = start + 590.0  # Safety margin

    # 1. Initial Solution: Greedy Hill Climbing
    # Start with all CW
    r = np.zeros(D, dtype=np.int8)
    best_r = r.copy()
    best_val = max_load(best_r)

    # Greedy local improvement (single flips)
    improved = True
    while improved:
        improved = False
        for i in range(D):
            r[i] = 1 - r[i]
            v = max_load(r)
            if v < best_val - 1e-12:
                best_val = v
                best_r = r.copy()
                improved = True
            else:
                r[i] = 1 - r[i]  # revert

    # Also try starting from all-CCW
    r_ccw = np.ones(D, dtype=np.int8)
    v_ccw = max_load(r_ccw)
    if v_ccw < best_val - 1e-12:
        best_val = v_ccw
        best_r = r_ccw.copy()
        r = r_ccw.copy()
        improved = True
        while improved:
            improved = False
            for i in range(D):
                r[i] = 1 - r[i]
                v = max_load(r)
                if v < best_val - 1e-12:
                    best_val = v
                    best_r = r.copy()
                    improved = True
                else:
                    r[i] = 1 - r[i]

    # 2. Simulated Annealing
    # We use SA to escape local optima.
    # State: current routing cur_r, current load cur_val
    # Global best: best_r, best_val
    
    cur_r = best_r.copy()
    cur_val = best_val
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    # We want to run as many iterations as possible within the time limit.
    # Estimate iterations per second to set a budget, or just loop until deadline.
    
    # We will alternate between single flips and double flips (2-flips) to explore larger neighborhoods.
    # Double flips allow escaping from states where no single flip is beneficial but a pair is.
    
    # Cooling schedule: Geometric
    # We don't know the exact number of iterations in advance, so we use a time-based cooling or fixed steps.
    # Let's use a fixed number of steps that we estimate fits in the time, or just loop until time runs out.
    # Looping until time runs out is safer.
    
    # To make SA effective, we need to adjust T. 
    # Since we don't know N_iters, we can use a simple linear decrease based on elapsed time? 
    # Or just pick a reasonable T decay per iteration.
    # Let's assume we can do ~100k-1M iterations in 600s? 
    # Each iteration is very fast (numpy ops on small arrays).
    # Let's try to do as many as possible.
    
    # However, SA needs a cooling schedule. 
    # If we just loop until deadline, T needs to decrease.
    # Let's estimate max iterations: 500,000.
    # If we finish early, we can restart or continue.
    
    # Actually, a simpler robust approach:
    # Run SA for a fixed number of iterations (e.g., 100,000) with a geometric cooling schedule.
    # If time remains, run another round with a lower T_start or more iterations.
    
    # Let's define a function to run a block of SA.
    def run_sa_block(r_init, val_init, T_init, T_final, n_iters, deadline):
        cur_r = r_init.copy()
        cur_val = val_init
        best_r_local = r_init.copy()
        best_val_local = val_init
        
        # Precompute alpha for geometric cooling
        if n_iters > 0:
            alpha = (T_final / T_init) ** (1.0 / n_iters)
        else:
            alpha = 1.0
            
        T = T_init
        rng = np.random.default_rng(42)
        
        for k in range(n_iters):
            if time.time() > deadline:
                break
                
            # Choose move: 80% single flip, 20% double flip
            if rng.random() < 0.8:
                i = int(rng.integers(D))
                cur_r[i] = 1 - cur_r[i]
                new_val = max_load(cur_r)
                # Revert for now to check acceptance
                # Actually, computing max_load is the expensive part. 
                # We can calculate delta more efficiently?
                # No, max_load is O(N) which is 10. It's very fast.
                
                if new_val < cur_val:
                    cur_val = new_val
                    if new_val < best_val_local - 1e-12:
                        best_val_local = new_val
                        best_r_local = cur_r.copy()
                else:
                    # Accept with probability exp(-(new_val - cur_val) / T)
                    delta_val = new_val - cur_val
                    if T > 0:
                        accept_prob = np.exp(-delta_val / T)
                    else:
                        accept_prob = 0.0
                    
                    if rng.random() < accept_prob:
                        cur_val = new_val
                    else:
                        cur_r[i] = 1 - cur_r[i] # Revert
            else:
                i = int(rng.integers(D))
                j = int(rng.integers(D))
                if i == j:
                    # Degenerate to single flip
                    cur_r[i] = 1 - cur_r[i]
                    new_val = max_load(cur_r)
                    if new_val < cur_val:
                        cur_val = new_val
                        if new_val < best_val_local - 1e-12:
                            best_val_local = new_val
                            best_r_local = cur_r.copy()
                    else:
                        delta_val = new_val - cur_val
                        if T > 0:
                            accept_prob = np.exp(-delta_val / T)
                        else:
                            accept_prob = 0.0
                        if rng.random() < accept_prob:
                            cur_val = new_val
                        else:
                            cur_r[i] = 1 - cur_r[i]
                else:
                    cur_r[i] = 1 - cur_r[i]
                    cur_r[j] = 1 - cur_r[j]
                    new_val = max_load(cur_r)
                    if new_val < cur_val:
                        cur_val = new_val
                        if new_val < best_val_local - 1e-12:
                            best_val_local = new_val
                            best_r_local = cur_r.copy()
                    else:
                        delta_val = new_val - cur_val
                        if T > 0:
                            accept_prob = np.exp(-delta_val / T)
                        else:
                            accept_prob = 0.0
                        if rng.random() < accept_prob:
                            cur_val = new_val
                        else:
                            cur_r[i] = 1 - cur_r[i]
                            cur_r[j] = 1 - cur_r[j]
                            
            T *= alpha
            
        return best_r_local, best_val_local, cur_r, cur_val

    # Run SA blocks
    # Block 1: High temp, explore
    T_start = 5.0
    T_end = 0.1
    n_iters = 50000
    
    best_r, best_val, cur_r, cur_val = run_sa_block(best_r, best_val, T_start, T_end, n_iters, deadline)
    
    # Block 2: Lower temp, refine
    if time.time() < deadline - 10:
        T_start = 0.1
        T_end = 0.001
        n_iters = 50000
        best_r, best_val, cur_r, cur_val = run_sa_block(best_r, best_val, T_start, T_end, n_iters, deadline)

    # Block 3: Even lower temp, fine tune
    if time.time() < deadline - 5:
        T_start = 0.001
        T_end = 0.0001
        n_iters = 50000
        best_r, best_val, cur_r, cur_val = run_sa_block(best_r, best_val, T_start, T_end, n_iters, deadline)
        
    # Final greedy polish on the best solution found
    r = best_r.copy()
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for i in range(D):
            r[i] = 1 - r[i]
            v = max_load(r)
            if v < best_val - 1e-12:
                best_val = v
                best_r = r.copy()
                improved = True
            else:
                r[i] = 1 - r[i]

    # Final: ensure best_r is a 0/1 int array; map to output (>=0.5 CCW else CW).
    out = best_r.astype(np.float64)
    # Per contract: entry >= 0.5 routes CCW. Our 1 = CCW, 0 = CW, so out in {0,1} works.
    return out


# EVOLVE-BLOCK-END
