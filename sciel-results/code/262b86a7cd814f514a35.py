# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        value >= 0.5 -> route the demand counter-clockwise
        value < 0.5  -> route the demand clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute for each demand:
    #   cw_mask[d, e] = 1 if edge e is used when routing demand d clockwise
    #   ccw_mask[d, e] = 1 if edge e is used when routing demand d counter-clockwise
    cw_mask = np.zeros((D, n), dtype=np.int32)
    ccw_mask = np.zeros((D, n), dtype=np.int32)
    weights = np.zeros(D, dtype=np.float64)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        if s == t:
            continue
        e = s
        while e != t:
            cw_mask[d, e] = 1
            e = (e + 1) % n
        # Counter-clockwise uses all other edges
        for e in range(n):
            if cw_mask[d, e] == 0:
                ccw_mask[d, e] = 1

    # We model the routing direction x_d in [0, 1].
    # x_d = 0 means CW, x_d = 1 means CCW.
    # Load on edge e: L_e(x) = sum_d w_d * ( (1-x_d)*cw_mask[d,e] + x_d*ccw_mask[d,e] )
    #                 = sum_d w_d * ( cw_mask[d,e] + x_d*(ccw_mask[d,e] - cw_mask[d,e]) )
    
    # Let A[d, e] = cw_mask[d, e]
    # Let B[d, e] = ccw_mask[d, e] - cw_mask[d, e]
    # L_e(x) = sum_d w_d * A[d, e] + sum_d w_d * B[d, e] * x_d
    # Let base_load[e] = sum_d w_d * A[d, e]
    # Let Delta[e] = sum_d w_d * B[d, e] * x_d

    base_load = np.zeros(n, dtype=np.float64)
    Delta_matrix = np.zeros((n, D), dtype=np.float64) # Delta_matrix[e, d] = w_d * B[d, e]

    for d in range(D):
        for e in range(n):
            base_load[e] += weights[d] * cw_mask[d, e]
            Delta_matrix[e, d] = weights[d] * (ccw_mask[d, e] - cw_mask[d, e])

    start_time = time.time()
    deadline = 600 # Safety margin

    # Function to compute loads and max load for a binary vector
    def compute_loads(x_bin):
        # x_bin is 0 or 1
        # Load = base_load + Delta_matrix @ x_bin
        loads = base_load + Delta_matrix @ x_bin.astype(np.float64)
        return loads, np.max(loads)

    # Local Search Polish
    # Flip each bit and see if max load improves
    def local_search(x_bin, max_time=None):
        def get_max_load(x_bin):
            # x_bin is 0 or 1
            # Load = base_load + Delta_matrix @ x_bin
            loads = base_load + Delta_matrix @ x_bin.astype(np.float64)
            return np.max(loads), loads

        current_max, _ = get_max_load(x_bin)
        
        improved = True
        while improved and (max_time is None or (time.time() - start_time) < max_time):
            improved = False
            for d in range(D):
                # Try flipping d
                new_x = x_bin.copy()
                new_x[d] = 1 - new_x[d]
                new_max, _ = get_max_load(new_x)
                if new_max < current_max - 1e-9:
                    x_bin = new_x
                    current_max = new_max
                    improved = True
        return x_bin, current_max

    # Multiple random restarts with local search
    best_x_bin = None
    best_max_load = float('inf')

    # First, try the all-0.5 initialization (which rounds to all 0 or 1 depending on tie-break)
    # Actually, let's just do random restarts
    np.random.seed(42)
    
    num_restarts = 20
    for i in range(num_restarts):
        if (time.time() - start_time) > deadline - 30:
            break
        
        # Random initial binary vector
        x0_bin = np.random.randint(0, 2, D).astype(np.int8)
        
        # Local search
        x_opt, max_load = local_search(x0_bin, max_time=deadline - 30)
        
        if max_load < best_max_load - 1e-9:
            best_max_load = max_load
            best_x_bin = x_opt.copy()

    # Also try the result from the smooth optimization as a starting point
    def soft_max_load(x, alpha=10.0):
        """
        Compute the smooth approximation of the max load.
        x: array of shape (D,)
        """
        # L = base_load + Delta_matrix @ x
        loads = base_load + Delta_matrix @ x
        # Log-sum-exp trick for numerical stability
        # max(loads) + log(sum(exp(loads - max(loads))))
        m = np.max(loads)
        if m > 500: # prevent overflow
            return m
        return m + np.log(np.sum(np.exp(alpha * (loads - m)))) / alpha

    def soft_max_load_grad(x, alpha=10.0):
        """
        Gradient of the smooth approximation.
        """
        loads = base_load + Delta_matrix @ x
        m = np.max(loads)
        if m > 500:
            idx = np.argmax(loads)
            return Delta_matrix[idx, :]
        
        exp_loads = np.exp(alpha * (loads - m))
        probs = exp_loads / np.sum(exp_loads)
        # Gradient is probs @ Delta_matrix
        return probs @ Delta_matrix

    # Initial guess: all 0.5 (balanced)
    x0 = np.full(D, 0.5)
    
    # Bounds: x in [0, 1]
    bounds = [(0, 1)] * D

    # Use L-BFGS-B
    try:
        res = minimize(
            soft_max_load, 
            x0, 
            method='L-BFGS-B', 
            jac=soft_max_load_grad, 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x_opt = res.x
        x_bin = (x_opt >= 0.5).astype(np.int8)
        
        # Local search from this point
        x_opt2, max_load2 = local_search(x_bin, max_time=deadline - 30)
        
        if max_load2 < best_max_load - 1e-9:
            best_max_load = max_load2
            best_x_bin = x_opt2.copy()
    except Exception:
        pass

    # If we still don't have a result (shouldn't happen), use a simple heuristic
    if best_x_bin is None:
        best_x_bin = np.zeros(D, dtype=np.int8)

    # Convert to float array where >= 0.5 means CCW
    # x_bin is 0 (CW) or 1 (CCW)
    # Return x_bin as float
    return best_x_bin.astype(float)

# EVOLVE-BLOCK-END
