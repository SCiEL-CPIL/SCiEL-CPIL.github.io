# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _compute_1d_union_length(positions, N):
    """
    Compute the length of the union of N sheared intervals in 1D.
    positions: array of shape (N,) representing base positions x_k.
    Tube k (k=0..N-1) at height c in [0,1] is [x_k + c*k/N, x_k + c*k/N + 1/N].
    The union is over all k and c.
    Due to linearity, the union of these intervals at any fixed c is the union of
    intervals [x_k + c*k/N, x_k + c*k/N + 1/N].
    We integrate the length of this union over c from 0 to 1.
    """
    N = len(positions)
    # We can't just sum lengths because they overlap.
    # A robust way to compute the length of the union of a set of intervals is:
    # 1. Collect all endpoints (start and end) for each k.
    # 2. Sort them.
    # 3. Sweep line to find the length of the union.
    
    # However, the intervals depend on c.
    # The union length L(c) is a piecewise linear function of c.
    # The breakpoints occur when the order of the endpoints changes.
    # Endpoint k_start = x_k + c*k/N
    # Endpoint k_end = x_k + c*k/N + 1/N
    
    # Let's sample c at many points and integrate using Simpson's rule or trapezoid.
    # Since the function is piecewise linear, we can find the exact breakpoints.
    # But for N=8, there are 16 endpoints. The order changes when 
    # x_i + c*i/N = x_j + c*j/N  => c = (x_j - x_i) / ((i-j)/N) = N*(x_j-x_i)/(i-j).
    # There are O(N^2) such critical c values.
    
    # Let's collect all critical c values in [0, 1].
    critical_cs = [0.0, 1.0]
    for i in range(N):
        for j in range(N):
            if i != j:
                c_crit = N * (positions[j] - positions[i]) / (i - j)
                if 0.0 <= c_crit <= 1.0:
                    critical_cs.append(c_crit)
    
    # Also include endpoints where intervals start or end exactly at boundaries?
    # The sweep line logic works on the set of active intervals.
    # The length of the union changes slope only when an endpoint crosses another.
    # So sampling at critical_cs and integrating linearly between them is exact.
    
    critical_cs = np.unique(critical_cs)
    # Sort
    critical_cs.sort()
    
    total_length = 0.0
    for idx in range(len(critical_cs) - 1):
        c1 = critical_cs[idx]
        c2 = critical_cs[idx+1]
        dc = c2 - c1
        if dc <= 0:
            continue
        
        # Calculate union length at c1 and c2?
        # Since it's linear in between, we can just take the average of the lengths at c1 and c2?
        # No, the union of intervals is not linear in the same way if the topology changes.
        # But the topology (which intervals overlap) only changes at critical points.
        # So between c1 and c2, the set of active intervals and their relative order is constant.
        # Thus, the length of the union is linear in c.
        # So we can compute L(c1) and L(c2) and take the average.
        
        # Helper to compute union length at a specific c
        def union_len_at_c(c_val):
            intervals = []
            for k in range(N):
                start = positions[k] + c_val * k / N
                end = start + 1.0 / N
                intervals.append((start, end))
            
            # Sort by start
            intervals.sort(key=lambda x: x[0])
            
            # Merge
            merged = []
            for interval in intervals:
                if not merged:
                    merged.append(list(interval))
                else:
                    last = merged[-1]
                    if interval[0] <= last[1]:
                        last[1] = max(last[1], interval[1])
                    else:
                        merged.append(list(interval))
            
            length = 0.0
            for m in merged:
                length += m[1] - m[0]
            return length
        
        L1 = union_len_at_c(c1)
        L2 = union_len_at_c(c2)
        
        # Trapezoidal rule for linear function
        total_length += 0.5 * (L1 + L2) * dc
        
    return total_length

def _objective_1d(params, N=8):
    """
    Objective function for 1D optimization.
    params: array of N base positions.
    Returns the union length.
    """
    return _compute_1d_union_length(params, N)

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimizes the 1D Kakeya positions using differential evolution.
    The 3D problem decomposes into two identical 1D problems for x and y.
    We solve for the optimal 1D positions and use them for both dimensions.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, leave margin
    
    # Initial guess: Keich construction (from seed)
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Define bounds for differential evolution
    # The positions should be within a reasonable range.
    # The union width is at most 1 + max_offset.
    # Let's bound x in [-1, 2] to be safe.
    bounds = [(-1.0, 2.0)] * N
    
    # Run differential evolution
    # We need to be careful with the time budget.
    # DE is stochastic. We'll set maxiter and check time.
    
    best_pos = keich.copy()
    best_score = _objective_1d(best_pos, N)
    
    # Wrap objective to handle time
    def timed_obj(params):
        if time.time() > deadline - 5:
            return best_score # Return current best to stop early? No, DE doesn't support that directly.
        return _objective_1d(params, N)
    
    # We'll run DE with a reasonable number of iterations.
    # N=8 is small, so DE should be fast.
    # Let's allow up to 500 iterations or until time is up.
    
    try:
        result = differential_evolution(
            _objective_1d,
            bounds,
            args=(N,),
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True
        )
        
        if result.success or result.fun < best_score:
            best_pos = result.x
            best_score = result.fun
            
    except Exception as e:
        # If DE fails, fall back to Keich
        pass
    
    # Check time
    if time.time() > deadline - 10:
        # Use best found
        pass
    
    # Construct the 3D positions
    # i = k // N, j = k % N
    # base_x = best_pos[i], base_y = best_pos[j]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_pos[i_flat]
    base_y = best_pos[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
