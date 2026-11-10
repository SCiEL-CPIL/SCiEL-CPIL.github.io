# EVOLVE-BLOCK-START
import numpy as np

def _compute_area(positions: np.ndarray, num_heights=20000) -> float:
    """
    Compute the area of the union of the n triangles using the specified quadrature.
    """
    n = len(positions)
    y = (np.arange(num_heights) + 0.5) / num_heights  # shape (num_heights,)
    
    # a_i = (i+1)/n for i=0..n-1
    a = (np.arange(n) + 1.0) / n  # shape (n,)
    
    # For each height y, L_i = x_i + a_i * y, w = (1 - y) / n
    # L has shape (num_heights, n)
    L = positions[None, :] + a[None, :] * y[:, None]  # shape (num_heights, 64)
    w = (1.0 - y) / n  # shape (num_heights,)
    
    # Sort L along axis=1
    L_sorted = np.sort(L, axis=1)  # shape (num_heights, 64)
    
    # Union length at each height: w + sum_{j=0}^{n-2} min(L_{j+1} - L_j, w)
    diffs = L_sorted[:, 1:] - L_sorted[:, :-1]  # shape (num_heights, 63)
    min_diffs = np.minimum(diffs, w[:, None])  # shape (num_heights, 63)
    
    union_lengths = w + np.sum(min_diffs, axis=1)  # shape (num_heights,)
    
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Uses Coordinate Descent with Ternary Search using a reduced-height metric for speed,
    then refines with the full metric.
    """
    n = 64
    k = int(np.log2(n))
    
    # Initial guess: Keich construction
    j = np.arange(1, k + 1)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
        
    # Phase 1: Coarse optimization with reduced heights (2000)
    # This is ~10x faster than full metric.
    max_outer_iters = 10
    ternary_steps = 25
    
    for _ in range(max_outer_iters):
        for i in range(n):
            lo = -1.0
            hi = 2.0
            
            for _ in range(ternary_steps):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                
                positions[i] = m1
                f1 = _compute_area(positions, num_heights=2000)
                
                positions[i] = m2
                f2 = _compute_area(positions, num_heights=2000)
                
                if f1 < f2:
                    hi = m2
                else:
                    lo = m1
            
            positions[i] = (lo + hi) / 2.0
            
    # Phase 2: Fine optimization with full heights (20000)
    # Fewer iterations as we are close to optimum.
    max_outer_iters_fine = 2
    ternary_steps_fine = 20
    
    for _ in range(max_outer_iters_fine):
        for i in range(n):
            # Narrow the search range around current value
            # The step size in ternary search will determine precision.
            # We can use a window of +/- 0.05 around current pos to ensure we don't jump far.
            # Or just use the same wide range, it's only 2*64*20*2 = 5120 evals.
            # 5120 * 1.28M * (20000/2000) ... wait.
            # Full metric eval is 10x slower.
            # 5120 evals * 10x cost = 51200 "fast" equiv.
            # 51200 * 1.28M / 10 (wait, fast is 2000, full is 20000)
            # Fast eval cost: C. Full eval cost: 10C.
            # Phase 1: 10 * 64 * 25 * 2 = 32000 fast evals.
            # Phase 2: 2 * 64 * 20 * 2 = 5120 full evals = 51200 fast equiv.
            # Total ~83200 fast equiv.
            # 83200 * 1.28M ops / 10 (for fast) ... 
            # Let's just estimate time.
            # 1 fast eval ~ 0.1 ms? (1.28M ops is small).
            # 32000 * 0.1ms = 3.2s.
            # 5120 full evals. Full eval is ~1ms?
            # 5120 * 1ms = 5s.
            # Total < 10s. Safe.
            
            lo = -1.0
            hi = 2.0
            
            for _ in range(ternary_steps_fine):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                
                positions[i] = m1
                f1 = _compute_area(positions, num_heights=20000)
                
                positions[i] = m2
                f2 = _compute_area(positions, num_heights=20000)
                
                if f1 < f2:
                    hi = m2
                else:
                    lo = m1
            
            positions[i] = (lo + hi) / 2.0

    return positions.astype(np.float64)

# EVOLVE-BLOCK-END
