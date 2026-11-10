# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _union_length_and_grad(base, motion, width, M=400):
    """Compute integral of L(c)^2 over c in [0,1] and gradient w.r.t. base.
    
    Uses trapezoidal rule with M points. For each c, computes union length
    via sweep-line and contribution of each interval to the union.
    """
    K = len(base)
    c_vals = np.linspace(0.0, 1.0, M)
    dc = 1.0 / (M - 1)
    
    # Precompute: for each c, lo[k] = base[k] + c*motion[k], hi[k] = lo[k] + width
    # We'll compute L[c] and contrib[c,k] (length of union that interval k contributes)
    
    L = np.zeros(M)
    contrib = np.zeros((M, K))
    
    for m in range(M):
        lo = base + c_vals[m] * motion
        hi = lo + width
        # Sweep line: sort by lo, track current rightmost
        order = np.argsort(lo, kind='stable')
        s_lo = lo[order]
        s_hi = hi[order]
        
        cur_right = -np.inf
        # For each interval in sorted order, compute its contribution to union
        # An interval [lo_i, hi_i] contributes max(0, hi_i - max(cur_right, lo_i))
        for t in range(K):
            interval_lo = s_lo[t]
            interval_hi = s_hi[t]
            # Contribution is the part of this interval not covered by previous
            new_part = interval_hi - max(cur_right, interval_lo)
            if new_part > 0:
                contrib[m, order[t]] = new_part
                cur_right = interval_hi
            # Update cur_right
            if interval_hi > cur_right:
                cur_right = interval_hi
        L[m] = cur_right - s_lo[0] if K > 0 else 0.0
    
    # Integral of L^2
    val = dc * np.sum(L * L)
    
    # Gradient: d/dc integral of L^2 = integral of 2*L * dL/dbase_k
    # dL/dbase_k = sum over c of (contribution of interval k to union) * dc
    # Actually, L = sum_k contrib[k], so dL/dbase_k = contrib[k] (piecewise)
    # More carefully: the union length is a piecewise linear function, and the
    # derivative w.r.t. base_k is the indicator that interval k is "active" in the union
    # which is exactly contrib[k] / width... no, contrib[k] is the length contribution.
    # 
    # Let me reconsider: L(c) = measure of union of intervals at height c.
    # dL/dc_k = 1 if moving base_k right increases the union, which happens when
    # interval k is not fully covered. The contribution contrib[m,k] is the length
    # of interval k that is in the union but not covered by earlier intervals.
    # When we shift base_k by delta, the union changes by delta * (indicator that
    # interval k is "exposed"). The exposed part is exactly contrib[m,k] in length,
    # but the derivative is actually 1 if the interval is exposed at its right end, 0 otherwise.
    # 
    # Actually, for a single interval [a, a+w] in a union, shifting a by da changes the
    # union length by da * (1 - coverage_of_right_end). This is complex.
    # 
    # Simpler: use automatic differentiation via finite differences for gradient,
    # or use the fact that for non-overlapping intervals at c=0, the gradient is simpler.
    # 
    # Let's use a cleaner approach: the gradient of L^2 w.r.t. base_k is
    # 2 * L * dL/dbase_k, and dL/dbase_k can be computed as the measure of c where
    # interval k is "critical" (its right endpoint extends the union).
    # 
    # For now, use finite differences for gradient since M is small enough.
    
    g = np.zeros(K)
    eps = 1e-7
    for k in range(K):
        base_plus = base.copy()
        base_plus[k] += eps
        # Recompute L^2 integral for perturbed base
        L_plus = np.zeros(M)
        for m in range(M):
            lo = base_plus + c_vals[m] * motion
            hi = lo + width
            order = np.argsort(lo, kind='stable')
            s_lo = lo[order]
            s_hi = hi[order]
            cur_right = -np.inf
            for t in range(K):
                if s_hi[t] > cur_right:
                    cur_right = s_hi[t]
            L_plus[m] = cur_right - s_lo[0] if K > 0 else 0.0
        val_plus = dc * np.sum(L_plus * L_plus)
        g[k] = (val_plus - val) / eps
    
    return val, g


def _solve_axis_1d(motion, N, iters=200):
    """Solve 1D axis optimization: find base positions for N intervals.
    
    Minimize ∫₀¹ L(c)² dc subject to non-overlapping at c=0.
    Uses L-BFGS-B with barrier for non-overlap.
    """
    width = 1.0 / N
    K = N
    
    # Initial guess: evenly spaced
    base0 = np.linspace(0, (K - 1) * width, K)
    
    # Barrier parameter for non-overlap constraint
    lam = 1e8
    
    def objective(x):
        v, g = _union_length_and_grad(x, motion, width, M=200)
        # Add barrier for non-overlap at c=0
        # Constraint: x[k] + width <= x[k+1] for k=0..K-2
        # i.e., x[k+1] - x[k] - width >= 0
        for k in range(K - 1):
            d = x[k + 1] - x[k] - width
            if d < 0:
                v += lam * d * d
                g[k] -= 2 * lam * d
                g[k + 1] += 2 * lam * d
        return v, g
    
    res = minimize(objective, base0, jac=True, method='L-BFGS-B',
                   options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-10})
    
    if res.success:
        return res.x
    else:
        return base0


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    width = 1.0 / N
    
    # Motion for each axis: for tube (i,j), x-motion is i/N, y-motion is j/N
    motion = np.arange(N, dtype=np.float64) / N
    
    # Solve for x-base positions (depend only on i)
    base_x = _solve_axis_1d(motion, N, iters=200)
    
    # Solve for y-base positions (depend only on j)
    base_y = _solve_axis_1d(motion, N, iters=200)
    
    # Build 64 positions: row k has i=k//8, j=k%8
    positions = np.empty((64, 2), dtype=np.float64)
    for k in range(64):
        i = k // N
        j = k % N
        positions[k, 0] = base_x[i]
        positions[k, 1] = base_y[j]
    
    return positions


# EVOLVE-BLOCK-END
