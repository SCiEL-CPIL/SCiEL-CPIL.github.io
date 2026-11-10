# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import powell
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    timeout = 590  # Safety margin before 600s limit

    def objective(pts):
        # pts is shape (32,)
        p = pts.reshape(n, d)
        dists = pdist(p)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return -(ratio ** 2)

    best_score = -1
    best_points = None

    # Generate diverse initializations
    inits = []

    # 1. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    poly = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(poly.flatten())

    # 2. 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    grid_x, grid_y = np.meshgrid(x, y)
    grid = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    inits.append(grid.flatten())

    # 3. Jittered Grids
    np.random.seed(42)
    for i in range(20):
        jitter = np.random.randn(n, d) * 0.2
        inits.append((grid + jitter).flatten())

    # 4. Random Points
    for i in range(20):
        inits.append(np.random.randn(n, d).flatten())

    # 5. Two Rings (8 inner, 8 outer)
    # Inner ring radius 0.5, Outer ring radius 1.5
    angles_in = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_out = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    inner = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    outer = np.column_stack([1.5 * np.cos(angles_out), 1.5 * np.sin(angles_out)])
    rings = np.vstack([inner, outer])
    inits.append(rings.flatten())
    
    # 6. Jittered Rings
    for i in range(10):
        jitter = np.random.randn(n, d) * 0.1
        inits.append((rings + jitter).flatten())

    # 7. Distorted Circle (varying radii)
    angles_c = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = 1.0 + 0.2 * np.sin(2 * angles_c)
    circ = np.column_stack([radii * np.cos(angles_c), radii * np.sin(angles_c)])
    inits.append(circ.flatten())
    
    # 8. Jittered Distorted Circle
    for i in range(10):
        jitter = np.random.randn(n, d) * 0.1
        inits.append((circ + jitter).flatten())

    # Run Powell optimization for each initialization
    for init in inits:
        if time.time() - start_time > timeout:
            break

        try:
            # Use a reasonable number of iterations and tolerance
            res = powell(objective, init, maxiter=1000, tol=1e-8)
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(n, d)
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon if all optimizations failed
        best_points = poly

    return best_points


# EVOLVE-BLOCK-END
