"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _score_from_points(points):
    """Compute (max pairwise dist / min pairwise dist)^2. Lower is better."""
    n = len(points)
    if n < 2:
        return np.inf
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
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
    Build 16 points from a 4x4 base grid with multi-mode sinusoidal deformation.
    Uses sum of two orthogonal sinusoidal modes in each direction for richer warping.
    params (10 total):
      p0, p1: amplitudes of mode 1 and mode 2 for x-displacement
      p2, p3: frequencies of mode 1 and mode 2 for x-displacement
      p4, p5: amplitudes of mode 1 and mode 2 for y-displacement
      p6, p7: frequencies of mode 1 and mode 2 for y-displacement
      p8: scale factor
      p9: rotation angle
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Base 4x4 grid centered at origin
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.array([0.0, 1.0, 2.0, 3.0])
    GX, GY = np.meshgrid(xs, ys, indexing='ij')
    x = GX.ravel()
    y = GY.ravel()
    
    # Center
    x = x - 1.5
    y = y - 1.5
    
    # Multi-mode sinusoidal deformation
    # x-displacement: sum of two sinusoidal modes based on y
    dx = p0 * np.sin(p2 * y) + p1 * np.sin(p3 * y + np.pi / 4)
    # y-displacement: sum of two sinusoidal modes based on x
    dy = p4 * np.sin(p6 * x) + p5 * np.sin(p7 * x + np.pi / 4)
    
    x = x + dx
    y = y + dy
    
    # Scale
    x = x * p8
    y = y * p8
    
    # Rotate
    c, s = np.cos(p9), np.sin(p9)
    xr = c * x - s * y
    yr = s * x + c * y
    
    return np.column_stack([xr, yr])


def _objective(params):
    """Objective function for DE: minimize the ratio squared."""
    points = _deformed_grid(params)
    d = _score_from_points(points)
    if not np.isfinite(d):
        return 1e10
    return d


def construct():
    # Define bounds for DE
    bounds = [
        (-0.3, 0.3),   # p0: x-disp mode 1 amplitude
        (-0.3, 0.3),   # p1: x-disp mode 2 amplitude
        (0.5, 3.0),    # p2: x-disp mode 1 frequency
        (0.5, 3.0),    # p3: x-disp mode 2 frequency
        (-0.3, 0.3),   # p4: y-disp mode 1 amplitude
        (-0.3, 0.3),   # p5: y-disp mode 2 amplitude
        (0.5, 3.0),    # p6: y-disp mode 1 frequency
        (0.5, 3.0),    # p7: y-disp mode 2 frequency
        (0.5, 1.5),    # p8: scale
        (-np.pi / 4, np.pi / 4),  # p9: rotation (limited to avoid symmetry issues)
    ]
    
    # Run differential evolution with more iterations for better convergence
    result = differential_evolution(
        _objective,
        bounds,
        maxiter=300,
        popsize=25,
        tol=1e-9,
        seed=42,
        polish=True,
        mutation=(0.3, 0.9),
        recombination=0.7,
    )
    
    best_params = result.x
    points = _deformed_grid(best_params)
    
    # Verify distinctness
    score = _score_from_points(points)
    if not np.isfinite(score):
        # Fallback: regular grid
        import itertools
        grid = list(itertools.product(range(4), repeat=2))
        return np.array(grid, dtype=float)
    
    return points
# EVOLVE-BLOCK-END
