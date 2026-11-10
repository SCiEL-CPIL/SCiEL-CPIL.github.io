# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _keich_init(N):
    """Binary Keich construction for 1-D positions."""
    k = int(np.log2(N))
    vals = np.empty(N, dtype=np.float64)
    for idx in range(N):
        bits = np.array([int(d) for d in bin(idx)[2:].zfill(k)])
        weights = np.array([(1 - j) * (2.0 ** -j) for j in range(1, k + 1)])
        vals[idx] = np.dot(bits, weights) / k
    return vals


def _get_breakpoints_1d(x, N, c):
    """Get all interval endpoints for 1D at height c."""
    w = 1.0 / N
    idx = np.arange(N, dtype=np.float64)
    starts = x + c * idx / N
    ends = starts + w
    pts = np.concatenate([starts, ends])
    pts.sort()
    return pts


def _union_length_1d(x, N, c):
    """Compute union length of 1D intervals at height c."""
    w = 1.0 / N
    idx = np.arange(N, dtype=np.float64)
    starts = x + c * idx / N
    order = np.argsort(starts)
    s = starts[order]
    e = s + w
    gaps = np.maximum(0.0, s[1:] - e[:-1])
    return (e[-1] - s[0]) - np.sum(gaps)


def _area_2d_at_c(x, y, N, c):
    """Compute the 2D union area at height c.
    
    Each tube k=(i,j) is [x_i + c*i/N, x_i + c*i/N + 1/N] x [y_j + c*j/N, y_j + c*j/N + 1/N].
    The union area is the area of the union of all N^2 rectangles.
    """
    w = 1.0 / N
    i_idx = np.repeat(np.arange(N, dtype=np.float64), N)
    j_idx = np.tile(np.arange(N, dtype=np.float64), N)
    
    # All rectangle corners
    rx0 = x[i_idx] + c * i_idx / N
    rx1 = rx0 + w
    ry0 = y[j_idx] + c * j_idx / N
    ry1 = ry0 + w
    
    # Use sweep line / coordinate compression
    xs = np.concatenate([rx0, rx1])
    ys = np.concatenate([ry0, ry1])
    xs = np.unique(xs)
    ys = np.unique(ys)
    
    # Build grid
    n_x = len(xs)
    n_y = len(ys)
    
    # For each cell, check if it's covered by any rectangle
    # Cell (ix, iy) is [xs[ix], xs[ix+1]] x [ys[iy], ys[iy+1]]
    # It's covered if there exists a rect with rx0 <= xs[ix] and rx1 >= xs[ix+1] and ry0 <= ys[iy] and ry1 >= ys[iy+1]
    
    # Vectorized approach: for each cell center, check coverage
    x_centers = 0.5 * (xs[:-1] + xs[1:])
    y_centers = 0.5 * (ys[:-1] + ys[1:])
    cell_wx = np.diff(xs)
    cell_wy = np.diff(ys)
    
    # For each cell, check if covered by any of the N^2 rectangles
    # A cell is covered if its center is inside any rectangle
    # rx0 <= xc <= rx1 and ry0 <= yc <= ry1
    
    # Shape: (n_x, n_y, N^2)
    xc = x_centers[:, np.newaxis, np.newaxis]
    yc = y_centers[np.newaxis, :, np.newaxis]
    
    in_x = (rx0[np.newaxis, np.newaxis, :] <= xc) & (xc <= rx1[np.newaxis, np.newaxis, :])
    in_y = (ry0[np.newaxis, np.newaxis, :] <= yc) & (yc <= ry1[np.newaxis, np.newaxis, :])
    
    covered = np.any(in_x & in_y, axis=2)  # (n_x, n_y)
    
    area = np.sum(covered * cell_wx[:, np.newaxis] * cell_wy[np.newaxis, :])
    return area


def _volume_2d(x, y, N, n_c=200):
    """Compute the 3D volume as integral of 2D area over c in [0,1]."""
    c_vals = np.linspace(0.0, 1.0, n_c)
    total = 0.0
    for c in c_vals:
        total += _area_2d_at_c(x, y, N, c)
    return total / n_c


def _volume_1d_sep(x, y, N, n_c=500):
    """Compute volume using separable 1D approach (product of union lengths)."""
    c_vals = np.linspace(0.0, 1.0, n_c)
    total = 0.0
    for c in c_vals:
        lx = _union_length_1d(x, N, c)
        ly = _union_length_1d(y, N, c)
        total += lx * ly
    return total / n_c


def _objective_2d(params, N, n_c=150):
    """Objective: 2D joint volume."""
    x = params[:N]
    y = params[N:2*N]
    return _volume_2d(x, y, N, n_c)


def _objective_1d_sep(params, N, n_c=500):
    """Objective: separable 1D volume."""
    x = params[:N]
    y = params[N:2*N]
    return _volume_1d_sep(x, y, N, n_c)


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    x0 = _keich_init(N)
    y0 = x0.copy()
    
    total_budget = 600.0
    safety = 30.0
    deadline = time.time() + total_budget - safety
    
    # Start with separable 1D optimization for quick baseline
    t0 = time.time()
    
    # Phase 1: Optimize x and y independently with 1D objective
    def _opt_1d_single(x_init, budget):
        best_x = x_init.copy()
        best_val = _volume_1d_sep(np.concatenate([x_init, x_init]), N, n_c=300)
        
        if time.time() < deadline - 10:
            try:
                bounds = [(-1.0, 2.0)] * N
                
                def obj_1d(x):
                    return _volume_1d_sep(np.concatenate([x, x]), N, n_c=200)
                
                res = differential_evolution(
                    obj_1d, bounds, seed=42, maxiter=100, tol=1e-8,
                    popsize=10, init='sobol', polish=False, updating='immediate'
                )
                if res.fun < best_val:
                    best_x = res.x.copy()
                    best_val = res.fun
            except Exception:
                pass
        
        if time.time() < deadline - 5:
            try:
                def obj_lbfgs(x):
                    return _volume_1d_sep(np.concatenate([x, x]), N, n_c=500)
                
                res2 = minimize(
                    obj_lbfgs, best_x, method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res2.fun < best_val:
                    best_x = res2.x.copy()
                    best_val = res2.fun
            except Exception:
                pass
        
        return best_x
    
    x_opt = _opt_1d_single(x0, (deadline - time.time()) * 0.4)
    y_opt = x_opt.copy()  # Symmetric
    
    # Phase 2: Joint 2D optimization
    params0 = np.concatenate([x_opt, y_opt])
    best_params = params0.copy()
    best_val = _objective_2d(params0, N, n_c=100)
    
    if time.time() < deadline - 15:
        try:
            bounds = [(-1.5, 2.5)] * (2 * N)
            
            def obj_de(params):
                return _objective_2d(params, N, n_c=100)
            
            res = differential_evolution(
                obj_de, bounds, seed=12345, maxiter=80, tol=1e-8,
                popsize=10, init='sobol', polish=False, updating='immediate'
            )
            if res.fun < best_val:
                best_params = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    # Phase 3: Local polish with L-BFGS-B on 2D objective
    if time.time() < deadline - 8:
        try:
            def obj_lbfgs(params):
                return _objective_2d(params, N, n_c=200)
            
            res2 = minimize(
                obj_lbfgs, best_params, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.fun < best_val:
                best_params = res2.x.copy()
                best_val = res2.fun
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead final polish
    if time.time() < deadline - 3:
        try:
            def obj_nm(params):
                return _objective_2d(params, N, n_c=300)
            
            res3 = minimize(
                obj_nm, best_params, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res3.fun < best_val:
                best_params = res3.x.copy()
                best_val = res3.fun
        except Exception:
            pass
    
    x_final = best_params[:N]
    y_final = best_params[N:2*N]
    
    # Build positions
    i_idx = np.repeat(np.arange(N, dtype=np.int64), N)
    j_idx = np.tile(np.arange(N, dtype=np.int64), N)
    
    base_x = x_final[i_idx].astype(np.float64)
    base_y = y_final[j_idx].astype(np.float64)
    
    positions = np.column_stack([base_x, base_y])
    
    # Safety check: ensure finite and valid
    if not np.all(np.isfinite(positions)):
        # Fallback to initial
        base_x = x0[i_idx].astype(np.float64)
        base_y = x0[j_idx].astype(np.float64)
        positions = np.column_stack([base_x, base_y])
    
    return positions


# EVOLVE-BLOCK-END
