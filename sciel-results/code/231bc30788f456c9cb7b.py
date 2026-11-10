import numpy as np
import time
from scipy.optimize import minimize

def _merge_intervals(intervals):
    """Merge overlapping intervals. intervals: list of (start, end). Returns merged list."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]

def _union_length(intervals):
    """Compute total length of union of intervals."""
    merged = _merge_intervals(intervals)
    return sum(e - s for s, e in merged)

def _compute_1d_union_length(params_1d, c, N):
    """
    Compute union length of 1D projections for given c.
    params_1d: array of shape (2*N,) = [p(8), q(8)] where position for index k is p[k] + q[k]
    For x-projection: x_ij = A_i + B_j, so at height c, interval is [x_ij + c*i/N, x_ij + c*i/N + 1/N]
    We need to compute union of all 64 intervals.
    """
    p = params_1d[:N]
    q = params_1d[N:]
    
    # x_ij = p_i + q_j for i in range(N), j in range(N)
    # But here p and q are the 1D parameters. For x: p=A, q=B. For y: p=C, q=D.
    # Interval for (i,j): [p_i + q_j + c * i / N, p_i + q_j + c * i / N + 1/N]
    # Note: the shear depends on i (the row index), not j.
    
    intervals = []
    for i in range(N):
        for j in range(N):
            x0 = p[i] + q[j] + c * i / N
            x1 = x0 + 1.0 / N
            intervals.append((x0, x1))
    
    return _union_length(intervals)

def _compute_volume(params):
    """
    Compute the union volume using exact 1D integration.
    params: shape (32,) = [A(8), B(8), C(8), D(8)]
    x_ij = A_i + B_j, y_ij = C_i + D_j
    
    The volume is integral_0^1 Lx(c) * Ly(c) dc, where Lx(c) is the union length
    of x-projections and Ly(c) is the union length of y-projections at height c.
    Wait, this is NOT correct. The union of rectangles does not factorize in general.
    However, if we assume that the x and y projections are independent (which they're not
    for the union), we can't just multiply.
    
    Actually, the union volume of rectangles [x_min, x_max] x [y_min, y_max] is NOT
    equal to Lx * Ly in general. That would be the volume of the bounding box of the union.
    
    We need to compute the actual union area at each c, then integrate.
    For 64 rectangles, we can use coordinate compression at each c.
    """
    if time.time() > _deadline:
        return 1e10
    
    N = 8
    A = params[0:N]
    B = params[N:2*N]
    C = params[2*N:3*N]
    D = params[3*N:4*N]
    
    # For a fixed c, the rectangle (i,j) is:
    # x in [A_i + B_j + c*i/N, A_i + B_j + c*i/N + 1/N]
    # y in [C_i + D_j + c*j/N, C_i + D_j + c*j/N + 1/N]
    
    # We'll use trapezoidal rule with 50 c-values for speed during optimization
    c_vals = np.linspace(0, 1, 51)
    
    total_area = 0.0
    for c in c_vals:
        x_mins = np.zeros((N, N))
        x_maxs = np.zeros((N, N))
        y_mins = np.zeros((N, N))
        y_maxs = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                x_mins[i, j] = A[i] + B[j] + c * i / N
                x_maxs[i, j] = x_mins[i, j] + 1.0 / N
                y_mins[i, j] = C[i] + D[j] + c * j / N
                y_maxs[i, j] = y_mins[i, j] + 1.0 / N
        
        # Coordinate compression
        x_coords = np.unique(np.concatenate([x_mins.flatten(), x_maxs.flatten()]))
        y_coords = np.unique(np.concatenate([y_mins.flatten(), y_maxs.flatten()]))
        
        if len(x_coords) < 2 or len(y_coords) < 2:
            total_area += 0.0
            continue
        
        # For each cell in the compressed grid, check if it's covered by any rectangle
        # Cell (a, b) is [x_coords[a], x_coords[a+1]] x [y_coords[b], y_coords[b+1]]
        # It's covered if there exists (i,j) such that:
        #   x_mins[i,j] <= x_coords[a] and x_maxs[i,j] >= x_coords[a+1]
        #   y_mins[i,j] <= y_coords[b] and y_maxs[i,j] >= y_coords[b+1]
        
        area = 0.0
        for a_idx in range(len(x_coords) - 1):
            x_lo = x_coords[a_idx]
            x_hi = x_coords[a_idx + 1]
            if x_hi - x_lo < 1e-15:
                continue
            # Check which rectangles cover this x-range
            x_covered = (x_mins <= x_lo + 1e-12) & (x_maxs >= x_hi - 1e-12)
            if not np.any(x_covered):
                continue
            for b_idx in range(len(y_coords) - 1):
                y_lo = y_coords[b_idx]
                y_hi = y_coords[b_idx + 1]
                if y_hi - y_lo < 1e-15:
                    continue
                y_covered = (y_mins <= y_lo + 1e-12) & (y_maxs >= y_hi - 1e-12)
                if np.any(x_covered & y_covered):
                    area += (x_hi - x_lo) * (y_hi - y_lo)
        
        total_area += area
    
    # Trapezoidal integration
    volume = total_area * (1.0 / 50)  # spacing is 1/50
    
    return volume

def kakeya_needle_3d8() -> np.ndarray:
    global _deadline
    start_time = time.time()
    _deadline = start_time + 580
    
    N = 8
    
    def objective(params):
        return _compute_volume(params)
    
    # Initial guess: spread out the positions
    A_init = -np.arange(N) * 0.1
    B_init = np.arange(N) * 0.1
    C_init = -np.arange(N) * 0.1
    D_init = np.arange(N) * 0.1
    
    params_init = np.concatenate([A_init, B_init, C_init, D_init])
    
    best_params = params_init.copy()
    best_volume = objective(best_params)
    
    # Use L-BFGS-B with multiple restarts
    inits = [
        np.concatenate([A_init, B_init, C_init, D_init]),
        np.concatenate([-np.arange(N) / N, np.zeros(N), -np.arange(N) / N, np.zeros(N)]),
        np.concatenate([np.zeros(N), -np.arange(N) / N, np.zeros(N), -np.arange(N) / N]),
        np.concatenate([np.arange(N) / N, -np.arange(N) / N, np.arange(N) / N, -np.arange(N) / N]),
        np.concatenate([-np.arange(N) / N, np.arange(N) / N, -np.arange(N) / N, np.arange(N) / N]),
        np.concatenate([np.random.default_rng(42).uniform(-1, 1, N),
                        np.random.default_rng(43).uniform(-1, 1, N),
                        np.random.default_rng(44).uniform(-1, 1, N),
                        np.random.default_rng(45).uniform(-1, 1, N)]),
    ]
    
    for init in inits:
        if time.time() > _deadline - 60:
            break
        try:
            res = minimize(objective, init, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
            if res.fun < best_volume:
                best_params = res.x
                best_volume = res.fun
        except Exception:
            pass
    
    # Construct final positions
    A = best_params[0:N]
    B = best_params[N:2*N]
    C = best_params[2*N:3*N]
    D = best_params[3*N:4*N]
    
    x_coords = A[:, None] + B[None, :]
    y_coords = C[:, None] + D[None, :]
    
    positions = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
    
    return positions
