"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


def _score_from_points(points):
    """Compute (max pairwise dist / min pairwise dist)^2. Lower is better."""
    n = len(points)
    if n < 2:
        return np.inf
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Exclude self-distances
    iu = np.triu_indices(n, k=1)
    pd = dists[iu]
    if len(pd) == 0:
        return np.inf
    dmin = pd.min()
    dmax = pd.max()
    if dmin <= 1e-12:
        return np.inf
    return (dmax / dmin) ** 2


def _deformed_grid(params):
    """
    Build 16 points from a 4x4 base grid with sinusoidal deformation.
    params:
      p0: scale factor
      p1: amp sine y (x-dir)
      p2: freq sine y (x-dir)
      p3: amp cos x (x-dir)
      p4: freq cos x (x-dir)
      p5: amp sine x (y-dir)
      p6: freq sine x (y-dir)
      p7: amp cos y (y-dir)
      p8: freq cos y (y-dir)
      p9: rotation angle
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Base 4x4 grid centered at origin
    # Coordinates: -1.5, -0.5, 0.5, 1.5
    xs = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float)
    ys = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float)
    GX, GY = np.meshgrid(xs, ys, indexing='ij')
    x = GX.ravel()
    y = GY.ravel()
    
    # Sinusoidal deformation
    # x' = x + p1*sin(p2*y) + p3*cos(p4*x)
    # y' = y + p5*sin(p6*x) + p7*cos(p8*y)
    dx = p1 * np.sin(p2 * y) + p3 * np.cos(p4 * x)
    dy = p5 * np.sin(p6 * x) + p7 * np.cos(p8 * y)
    
    x_new = x + dx
    y_new = y + dy
    
    # Apply global scale
    x_new = x_new * p0
    y_new = y_new * p0
    
    # Apply rotation
    c, s = np.cos(p9), np.sin(p9)
    xr = c * x_new - s * y_new
    yr = s * x_new + c * y_new
    
    return np.column_stack([xr, yr])


def _objective(params):
    """Objective function for DE: minimize the ratio squared."""
    points = _deformed_grid(params)
    # Check distinctness
    d = _score_from_points(points)
    if not np.isfinite(d):
        return 1e10
    return d


def construct():
    # Define bounds for DE
    # p0: scale, p1-p4: x-deform, p5-p8: y-deform, p9: rotation
    bounds = [
        (0.5, 2.0),            # p0: scale
        (-0.5, 0.5),           # p1: amp sine y
        (0.0, 2.0),            # p2: freq sine y
        (-0.5, 0.5),           # p3: amp cos x
        (0.0, 2.0),            # p4: freq cos x
        (-0.5, 0.5),           # p5: amp sine x
        (0.0, 2.0),            # p6: freq sine x
        (-0.5, 0.5),           # p7: amp cos y
        (0.0, 2.0),            # p8: freq cos y
        (-np.pi/4, np.pi/4),   # p9: rotation
    ]
    
    # Run differential evolution
    result = differential_evolution(
        _objective,
        bounds,
        maxiter=400,
        popsize=40,
        tol=1e-10,
        seed=42,
        polish=False,
        init='sobol',
        mutation=(0.3, 1.0),
        recombination=0.7,
    )
    
    best_params = result.x
    
    # Local refinement using L-BFGS-B
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    
    try:
        local_res = minimize(
            _objective,
            best_params,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        if local_res.success and local_res.fun < _objective(best_params):
            best_params = local_res.x
    except Exception:
        pass
    
    points = _deformed_grid(best_params)
    
    # Verify distinctness
    score = _score_from_points(points)
    if not np.isfinite(score):
        # Fallback: regular grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
        
    return points
# EVOLVE-BLOCK-END
