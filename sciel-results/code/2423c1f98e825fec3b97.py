import numpy as np
import time
from scipy.optimize import linprog

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
            # Same node, no edges? Or full ring? 
            # "Clockwise from s to t uses edges s, ..., t-1". If s==t, range is empty.
            # But usually in ring routing, if s==t, it's 0 or full ring. 
            # Given weights > 0, and s!=t in all demands here, this case doesn't occur.
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

    # LP Formulation:
    # Minimize Z
    # Variables: x_0, ..., x_{D-1}, Z
    # x_i = 1 if CCW, 0 if CW.
    # Load on edge e: 
    # L_e = sum_i w_i * (cw_mat[i,e] * (1-x_i) + ccw_mat[i,e] * x_i)
    # L_e = sum_i w_i * cw_mat[i,e] + sum_i w_i * (ccw_mat[i,e] - cw_mat[i,e]) * x_i
    # Constraint: L_e <= Z for all e
    # Z - sum_i w_i * (ccw_mat[i,e] - cw_mat[i,e]) * x_i >= sum_i w_i * cw_mat[i,e]
    # -Z + sum_i w_i * (ccw_mat[i,e] - cw_mat[i,e]) * x_i <= - sum_i w_i * cw_mat[i,e]
    
    # Let's use the standard linprog form: c^T x <= b
    # Variables vector v = [x_0, ..., x_{D-1}, Z]
    # Objective: minimize Z -> c = [0, ..., 0, 1]
    
    c = np.zeros(D + 1)
    c[D] = 1.0
    
    # Constraints for each edge e:
    # -Z + sum_i w_i * (ccw_mat[i,e] - cw_mat[i,e]) * x_i <= - base_load_e
    # where base_load_e = sum_i w_i * cw_mat[i,e]
    
    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    
    base_load = weights @ cw_mat # shape (n,)
    
    for e in range(n):
        # Coefficients for x_i
        diff = (ccw_mat[:, e] - cw_mat[:, e]) * weights
        A_ub[e, :D] = diff
        A_ub[e, D] = -1.0
        b_ub[e] = -base_load[e]
    
    # Bounds: 0 <= x_i <= 1, Z >= 0
    bounds = [(0, 1)] * D + [(0, None)]
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        x_lp = res.x[:D]
        # Round to get integer solution
        x_int = np.round(x_lp).astype(np.int8)
        # Ensure 0/1
        x_int[x_int < 0.5] = 0
        x_int[x_int >= 0.5] = 1
    else:
        # Fallback to random
        x_int = np.random.randint(0, 2, D).astype(np.int8)
        
    # Evaluate max load for a given routing
    def get_max_load(x):
        # x is 0/1 array
        # Load = base_load + sum_{i: x_i=1} w_i * (ccw_mat[i] - cw_mat[i])
        if np.any(x == 1):
            delta_sum = (ccw_mat[x == 1] * weights[x == 1, None]).sum(axis=0)
        else:
            delta_sum = np.zeros(n)
        load = base_load + delta_sum
        return np.max(load)
    
    best_x = x_int.copy()
    best_val = get_max_load(best_x)
    
    # Local Search to refine
    # Since D=15, we can also just check all 2^15 combinations if needed, 
    # but local search from LP solution should be very fast.
    
    improved = True
    while improved:
        improved = False
        for i in range(D):
            # Try flipping bit i
            trial_x = best_x.copy()
            trial_x[i] = 1 - trial_x[i]
            val = get_max_load(trial_x)
            if val < best_val - 1e-9:
                best_val = val
                best_x = trial_x
                improved = True
        
        # Try pairwise flips if single flips didn't help (escape local minima)
        if not improved:
            for i in range(D):
                for j in range(i + 1, D):
                    trial_x = best_x.copy()
                    trial_x[i] = 1 - trial_x[i]
                    trial_x[j] = 1 - trial_x[j]
                    val = get_max_load(trial_x)
                    if val < best_val - 1e-9:
                        best_val = val
                        best_x = trial_x
                        improved = True
                        break
                if improved:
                    break

    # Final check: Brute force all 2^15 if time permits? 
    # 32768 evaluations. Each evaluation is ~15 ops. 
    # This is ~500k ops. Very fast.
    # Let's do a full brute force to be absolutely sure we have the global optimum.
    # It's safer and fast enough.
    
    best_val_brute = np.inf
    best_x_brute = best_x.copy()
    
    # Iterate through all 2^15 masks
    # We can optimize this by updating loads incrementally or just vectorizing.
    # Given 32k iterations, a simple loop in Python might take ~0.1-0.5s.
    # Let's stick to the loop for clarity and correctness, as 32k is small.
    
    # To speed up the brute force, we can precompute the delta vectors.
    # delta_mat[i] = w_i * (ccw_mat[i] - cw_mat[i])
    delta_mat = (ccw_mat - cw_mat) * weights[:, None]
    
    for mask in range(1 << D):
        # Calculate load for this mask
        # We can use bit operations to find set bits, but summing is fine.
        # To make it faster, we can use a dynamic programming approach or just numpy.
        # Let's just sum the rows corresponding to set bits.
        # This is O(D) per mask.
        # Total O(2^D * D) = 32768 * 15 = ~500k operations.
        
        # Optimization: Use a lookup table or just simple accumulation?
        # Simple accumulation is fine.
        
        # Get indices of set bits
        # Instead of iterating bits, we can precompute loads for all masks?
        # No, memory is fine but time to precompute is same.
        
        # Let's just do the sum.
        # We can use:
        # bits = np.unpackbits(np.array([mask], dtype=np.uint16)[:1], axis=1, bitorder='little')[:D]
        # This is slow.
        
        # Better:
        # load = base_load.copy()
        # for i in range(D):
        #     if mask & (1 << i):
        #         load += delta_mat[i]
        # if max(load) < best_val_brute: ...
        
        # This loop is 32768 * 15 = 491k iterations. Python can do ~10-50M simple ops/sec.
        # So ~0.01 - 0.05 seconds. Totally acceptable.
        
        load = base_load.copy()
        for i in range(D):
            if mask & (1 << i):
                load += delta_mat[i]
        
        max_load = np.max(load)
        if max_load < best_val_brute - 1e-9:
            best_val_brute = max_load
            # Reconstruct x
            best_x_brute = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.int8)

    # The brute force guarantees the global optimum for this small problem size.
    # Return the best solution found.
    
    out = best_x_brute.astype(np.float64)
    return out

ring_routing()
