"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def _pairwise_distances_sq(points):
    """Compute squared pairwise distances."""
    pts = np.asarray(points, dtype=float)
    # Use formula: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    sq_norms = np.sum(pts * pts, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (pts @ pts.T)
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return dist_sq


def _score(points):
    """Return (max_dist / min_dist)^2 for the given points."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.inf
    d2 = _pairwise_distances_sq(pts)
    # Mask diagonal
    mask = ~np.eye(len(pts), dtype=bool)
    off_diag = d2[mask]
    if len(off_diag) == 0:
        return np.inf
    min_d2 = np.min(off_diag)
    max_d2 = np.max(off_diag)
    if min_d2 < 1e-15:
        return np.inf
    return max_d2 / min_d2


def _soft_score(points, beta=50.0):
    """Smooth surrogate using log-sum-exp for max and min."""
    pts = np.asarray(points, dtype=float)
    d2 = _pairwise_distances_sq(pts)
    mask = ~np.eye(len(pts), dtype=bool)
    off_diag = d2[mask]
    if len(off_diag) == 0:
        return np.inf
    # Soft max
    max_val = np.max(off_diag)
    log_sum_exp_max = np.log(np.sum(np.exp(off_diag - max_val))) + max_val
    # Soft min: -soft_max(-x)
    min_val = np.min(off_diag)
    log_sum_exp_min = np.log(np.sum(np.exp(-(off_diag) + min_val))) - min_val
    # soft_max = log_sum_exp_max / beta, soft_min = -log_sum_exp_min / beta... 
    # Actually for ratio we want: soft_max / soft_min
    # Let's use: soft_max = (1/beta) * log(sum(exp(beta*x)))
    # soft_min = -(1/beta) * log(sum(exp(-beta*x)))
    soft_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * off_diag - beta * max_val))) + max_val
    soft_min = -(1.0 / beta) * np.log(np.sum(np.exp(-beta * off_diag + beta * min_val))) + min_val
    if soft_min < 1e-15:
        return np.inf
    return soft_max / soft_min


def _objective(x, beta=50.0):
    """Objective function for optimizer. x is flattened (N*D,) array."""
    pts = x.reshape(N, D)
    return _soft_score(pts, beta)


def _halton_sequence(n, dims):
    """Generate Halton sequence points in [0,1]^dims."""
    points = []
    for i in range(1, n + 1):
        point = []
        for base in range(2, 2 + dims):
            f = 1.0
            r = 0.0
            i_val = i
            while i_val > 0:
                i_val, rem = divmod(i_val, base)
                f /= base
                r += f * rem
            point.append(r)
        points.append(point)
    return np.array(points)


def _fibonacci_sphere(n):
    """Generate n points on a sphere using Fibonacci lattice."""
    points = []
    for i in range(n):
        z = 1.0 - (2.0 * i + 1.0) / n
        r = np.sqrt(1.0 - z * z)
        phi = 2.0 * np.pi * i / n
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x, y, z])
    return np.array(points)


def _optimize_from_start(start_pts, max_iters=5000, tol=1e-12, beta=50.0):
    """Run L-BFGS-B optimization from a starting configuration."""
    x0 = start_pts.reshape(-1)
    result = minimize(_objective, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iters, 'ftol': tol, 'gtol': 1e-8, 'maxfun': 100000},
                      args=(beta,))
    return result.x.reshape(N, D), result.fun


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate multiple starting configurations
    starts = []
    
    # 1. Fibonacci sphere scaled
    fib = _fibonacci_sphere(N)
    starts.append(fib * 1.0)
    
    # 2. Halton sequence mapped to sphere-like distribution
    halton = _halton_sequence(N, 3)
    # Map to unit ball
    halton_scaled = halton * 2.0 - 1.0
    norms = np.linalg.norm(halton_scaled, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    halton_scaled = halton_scaled / norms * 0.8
    starts.append(halton_scaled)
    
    # 3. Regular simplex-like (first N vertices of high-dim simplex projected)
    # For 14 points in 3D, use a structured grid perturbed
    grid_pts = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if len(grid_pts) < N:
                    grid_pts.append([i * 0.5, j * 0.5, k * 0.5])
    if len(grid_pts) < N:
        # Pad with random
        while len(grid_pts) < N:
            grid_pts.append(np.random.uniform(-1, 1, 3))
    starts.append(np.array(grid_pts[:N]))
    
    # 4. Random spherical
    random_pts = np.random.randn(N, 3)
    random_pts = random_pts / np.linalg.norm(random_pts, axis=1, keepdims=True) * 0.9
    starts.append(random_pts)
    
    # 5. Icosahedron + center + extra points
    # Icosahedron has 12 vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Add 2 more points (center and one more)
    ico_ext = np.vstack([ico, [[0, 0, 0]], [[0, 0, 1.2]]])[:N]
    starts.append(ico_ext)
    
    # Optimize from each start
    best_pts = None
    best_score = np.inf
    
    for idx, start in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        try:
            opt_pts, opt_soft_score = _optimize_from_start(start, max_iters=3000, tol=1e-14, beta=100.0)
            actual_score = _score(opt_pts)
            
            if actual_score < best_score:
                best_score = actual_score
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Final polish with higher beta
    if best_pts is not None and time.time() < deadline - 20.0:
        try:
            polished_pts, _ = _optimize_from_start(best_pts, max_iters=5000, tol=1e-16, beta=200.0)
            polished_score = _score(polished_pts)
            if polished_score < best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
    
    return best_pts
# EVOLVE-BLOCK-END
