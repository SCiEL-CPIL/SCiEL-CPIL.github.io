"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _hilbert_curve_points(order=4, n_pts=1000):
    """Generate points along a Hilbert curve in [0,1]^2."""
    # Standard Hilbert curve generation
    def d2xy(order, d):
        """Convert Hilbert index d to (x, y) coordinates."""
        x = 0
        y = 0
        rx = 0
        ry = 0
        t = d
        s = 1
        while s < (1 << order):
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            t //= 4
            s *= 2
        return x + s, y + s

    total = 1 << (2 * order)
    pts = np.zeros((n_pts, 2))
    for i in range(n_pts):
        d = int(i * (total - 1) / (n_pts - 1))
        x, y = d2xy(order, d)
        pts[i, 0] = (x + 0.5) / total
        pts[i, 1] = (y + 0.5) / total
    return pts


def _smooth_hilbert_position(t, curve_pts):
    """Interpolate position along Hilbert curve given parameter t in [0,1]."""
    n = len(curve_pts)
    idx = t * (n - 1)
    i0 = int(np.floor(idx))
    i1 = min(i0 + 1, n - 1)
    frac = idx - i0
    return curve_pts[i0] * (1 - frac) + curve_pts[i1] * frac


def _objective(params, curve_pts, n_circles):
    """
    params: array of shape (n_circles, 2): [t_i, r_i]
    Returns negative sum of radii (to minimize), with penalties for violations.
    """
    t = params[:, 0]
    r = np.clip(params[:, 1], 0, 0.5)
    
    # Compute positions
    centers = np.array([_smooth_hilbert_position(ti, curve_pts) for ti in t])
    
    # Boundary violations
    x = centers[:, 0]
    y = centers[:, 1]
    # Penalty for being outside [r, 1-r]
    pen_x = np.sum(np.maximum(0, r - x) ** 2 + np.maximum(0, r - (1 - x)) ** 2)
    pen_y = np.sum(np.maximum(0, r - y) ** 2 + np.maximum(0, r - (1 - y)) ** 2)
    boundary_pen = 100 * (pen_x + pen_y)
    
    # Overlap violations
    overlap_pen = 0.0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_pen += 100 * (min_dist - dist) ** 2
    
    return -(np.sum(r)) + boundary_pen + overlap_pen


def _grad_objective(params, curve_pts, n_circles):
    """Numerical gradient."""
    eps = 1e-7
    grad = np.zeros_like(params)
    f0 = _objective(params, curve_pts, n_circles)
    for i in range(len(params)):
        old = params[i]
        params[i] = old + eps
        f1 = _objective(params, curve_pts, n_circles)
        params[i] = old
        grad[i] = (f1 - f0) / eps
    return grad


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Generate Hilbert curve
    curve_pts = _hilbert_curve_points(order=5, n_pts=2000)
    
    best_params = None
    best_obj = np.inf
    best_circles = None
    
    # Initialize with equal spacing along curve
    n = N
    t_init = np.linspace(0.05, 0.95, n)
    r_init = np.full(n, 0.08)
    params = np.column_stack([t_init, r_init])
    
    # Multiple restarts with different initializations
    rng = np.random.RandomState(42)
    
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            params = np.column_stack([t_init, r_init])
        else:
            # Perturbed initialization
            t_pert = t_init + rng.uniform(-0.05, 0.05, n)
            t_pert = np.clip(t_pert, 0.02, 0.98)
            r_pert = r_init * rng.uniform(0.8, 1.2, n)
            params = np.column_stack([t_pert, np.clip(r_pert, 0.01, 0.2)])
        
        # L-BFGS optimization
        try:
            from scipy.optimize import minimize
            result = minimize(
                _objective, params,
                args=(curve_pts, n),
                method='L-BFGS-B',
                bounds=[(0.01, 0.99), (0.01, 0.25)] * n,
                options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-10}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract best circles
    if best_params is not None:
        t_best = best_params[:, 0]
        r_best = np.clip(best_params[:, 1], 0.01, 0.5)
        centers = np.array([_smooth_hilbert_position(ti, curve_pts) for ti in t_best])
        
        # Clip to valid region
        x = centers[:, 0]
        y = centers[:, 1]
        x = np.clip(x, r_best, 1 - r_best)
        y = np.clip(y, r_best, 1 - r_best)
        centers[:, 0] = x
        centers[:, 1] = y
        
        circles = np.column_stack([x, y, r_best])
        
        # Final validation and fix
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist - 1e-9:
                    # Reduce radii to fix overlap
                    excess = min_dist - dist
                    circles[i, 2] = max(0.01, circles[i, 2] - excess / 2)
                    circles[j, 2] = max(0.01, circles[j, 2] - excess / 2)
        
        # Ensure within bounds
        for i in range(n):
            r = circles[i, 2]
            circles[i, 0] = np.clip(circles[i, 0], r, 1 - r)
            circles[i, 1] = np.clip(circles[i, 1], r, 1 - r)
        
        return circles
    else:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
