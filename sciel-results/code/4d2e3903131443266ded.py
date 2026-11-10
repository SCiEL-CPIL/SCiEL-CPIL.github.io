# EVOLVE-BLOCK-START
import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 580  # safety margin before 630s kill
    
    # Start with Keich construction as initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # The construction is separable: x depends on i, y depends on j
    # We optimize the 1D base positions for x (and reuse for y)
    # Since the problem is separable, V = (1D union vol) * (1D union vol) * 1
    # Actually V = ∫ area(c) dc where area(c) = union_x(c) * union_y(c)
    # If x and y use same construction: V = (∫ union_x(c) dc) * (∫ union_y(c) dc) / 1? No.
    # V = ∫₀¹ [union_x(c) * union_y(c)] dc
    # If x and y are independent with same distribution: V = E[ux(c)*uy(c)] = E[ux(c)]*E[uy(c)] if independent
    # But they're deterministic, so V = ∫ ux(c) uy(c) dc
    
    # For separable: minimize ∫ ux(c) uy(c) dc. If ux = uy, minimize (∫ ux)²? No, ∫ ux² dc.
    # Actually if ux(c) = uy(c) for all c, V = ∫ ux(c)² dc.
    # To minimize this, we want ux(c) small for all c.
    
    # 1D problem: minimize ∫₀¹ U(c) dc where U(c) = length of union of {xi + c*i/N + [0, 1/N]}
    # Or if we square it: minimize ∫ U(c)² dc for the separable case.
    
    # Let's use a fast evaluation and scipy optimization
    from scipy.optimize import minimize
    
    def union_length_1d(c, x):
        """Compute union length of intervals [x_i + c*i/N, x_i + c*i/N + 1/N] for i=0..N-1"""
        starts = x + c * np.arange(N) / N
        ends = starts + 1.0 / N
        # Sort intervals by start
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        # Merge and compute union length
        if len(s) == 0:
            return 0.0
        total = 0.0
        cur_start = s[0]
        cur_end = e[0]
        for k in range(1, len(s)):
            if s[k] > cur_end:
                total += cur_end - cur_start
                cur_start = s[k]
                cur_end = e[k]
            else:
                cur_end = max(cur_end, e[k])
        total += cur_end - cur_start
        return total
    
    def objective_sep(x1d):
        """Objective for separable case: ∫ U(c)² dc (since ux=uy)"""
        # Sample c values
        cs = np.linspace(0.01, 0.99, 50)
        vals = np.array([union_length_1d(c, x1d) for c in cs])
        return np.mean(vals**2)
    
    def objective_full(x1d):
        """More accurate: ∫ U(c) dc for the 1D part, but we need the product"""
        cs = np.linspace(0.01, 0.99, 100)
        vals = np.array([union_length_1d(c, x1d) for c in cs])
        return np.mean(vals**2)
    
    # Optimize the 1D positions
    x0 = keich.copy()
    
    result = minimize(objective_sep, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
    
    if time.time() > deadline - 30:
        x_opt = x0
    else:
        x_opt = result.x if result.success or result.fun < objective_sep(x0) else x0
    
    # Try a few more local optimizations from different starts if time permits
    best_x = x_opt
    best_val = objective_sep(x_opt)
    
    rng_local = np.random.default_rng(42)
    while time.time() < deadline - 60:
        # Random perturbation
        x_try = best_x + rng_local.normal(0, 0.05, size=N)
        val_try = objective_sep(x_try)
        if val_try < best_val:
            best_val = val_try
            best_x = x_try
            # Local polish
            res = minimize(objective_sep, best_x, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        else:
            # Occasionally try a larger jump
            if rng_local.random() < 0.1:
                x_try = best_x + rng_local.normal(0, 0.2, size=N)
                val_try = objective_sep(x_try)
                if val_try < best_val:
                    best_val = val_try
                    best_x = x_try
    
    # Build the 64x2 positions array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_x[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
