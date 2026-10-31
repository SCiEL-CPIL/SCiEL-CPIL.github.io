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


def _sinusoidal_grid(params):
    """
    Build 16 points from a 4x4 base grid with sinusoidal deformation.
    
    params (8 parameters):
      p0: scale factor
      p1: amplitude of sin mode in x (affects x-coord)
      p2: amplitude of cos mode in x (affects x-coord)
      p3: amplitude of sin mode in y (affects y-coord)
      p4: amplitude of cos mode in y (affects y-coord)
      p5: frequency multiplier for x-mode
      p6: frequency multiplier for y-mode
      p7: rotation angle
    """
    p0, p1, p2, p3, p4, p5, p6, p7 = params
    
    # Base 4x4 grid centered at origin
    xs = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float)
    ys = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float)
    GX, GY = np.meshgrid(xs, ys, indexing='ij')
    x = GX.ravel()
    y = GY.ravel()
    
    # Sinusoidal deformation modes
    # Mode 1: sin(kx * x) affects x, sin(ky * y) affects y
    # Mode 2: cos(kx * x) affects x, cos(ky * y) affects y
    
    kx = p5  # frequency in x direction
    ky = p6  # frequency in y direction
    
    # Deformation in x: p1*sin(kx*x) + p2*cos(kx*x)
    dx = p1 * np.sin(kx * x) + p2 * np.cos(kx * x)
    
    # Deformation in y: p3*sin(ky*y) + p4*cos(ky*y)
    dy = p3 * np.sin(ky * y) + p4 * np.cos(ky * y)
    
    x_new = x + dx
    y_new = y + dy
    
    # Apply global scale
    x_new = x_new * p0
    y_new = y_new * p0
    
    # Apply rotation
    c, s = np.cos(p7), np.sin(p7)
    xr = c * x_new - s * y_new
    yr = s * x_new + c * y_new
    
    return np.column_stack([xr, yr])


def _objective(params):
    """Objective function for DE: minimize the ratio squared."""
    points = _sinusoidal_grid(params)
    # Check distinctness
    d = _score_from_points(points)
    if not np.isfinite(d):
        return 1e10
    return d


def construct():
    # Define bounds for DE
    # p0: scale, p1-p4: sinusoidal amplitudes, p5-p6: frequencies, p7: rotation
    bounds = [
        (0.5, 2.0),         # p0: scale
        (-0.3, 0.3),        # p1: sin amplitude x
        (-0.3, 0.3),        # p2: cos amplitude x
        (-0.3, 0.3),        # p3: sin amplitude y
        (-0.3, 0.3),        # p4: cos amplitude y
        (0.5, 3.0),         # p5: frequency x
        (0.5, 3.0),         # p6: frequency y
        (-np.pi/4, np.pi/4),# p7: rotation
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
        mutation=(0.4, 1.0),
        recombination=0.8,
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
            options={'maxiter': 2000, 'ftol': 1e-14}
        )
        if local_res.fun < _objective(best_params):
            best_params = local_res.x
    except Exception:
        pass
    
    points = _sinusoidal_grid(best_params)
    
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
