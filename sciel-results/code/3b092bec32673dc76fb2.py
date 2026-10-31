"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _pairwise_distances(pts):
    """Compute all pairwise distances for an (N, 2) array."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    return dists

def _score(pts):
    """Compute (max_dist / min_dist)^2."""
    dists = _pairwise_distances(pts)
    # Exclude zero distances (self-pairs)
    np.fill_diagonal(dists, np.inf)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_min <= 0:
        return np.inf
    return (d_max / d_min)**2

def _objective(x):
    """Objective function for optimization: (max/min)^2"""
    pts = x.reshape(N, D)
    return _score(pts)

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    best_pts = None
    best_score = np.inf
    
    # Strategy 1: Differential Evolution (global optimization)
    # Use a reasonable range for coordinates
    bounds = [(-5, 5)] * (N * D)
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=200,
            tol=1e-10,
            polish=True,
            init='sobol',
            updating='immediate',
            popsize=15
        )
        if result.success or result.fun < best_score:
            best_score = result.fun
            best_pts = result.x.reshape(N, D)
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 10:
        if best_pts is not None:
            return best_pts
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        grid = list(__import__('itertools').product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
    
    # Strategy 2: Local refinement from multiple random starts
    rng = np.random.default_rng(42)
    
    # Try several random initializations
    for i in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random initialization within a unit circle
        theta = rng.uniform(0, 2*np.pi, N)
        r = rng.uniform(0, 1, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        
        # Local optimization
        try:
            res = minimize(
                _objective,
                pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_pts = res.x.reshape(N, D)
        except Exception:
            pass
    
    # Strategy 3: Try hexagonal lattice initialization
    # For 16 points, a hexagonal arrangement might be good
    # Hexagonal lattice with spacing 1
    hex_pts = []
    # Center point
    hex_pts.append([0, 0])
    # First ring: 6 points
    for k in range(6):
        angle = k * np.pi / 3
        hex_pts.append([np.cos(angle), np.sin(angle)])
    # Second ring: 10 points (we need 9 more for total 16)
    for k in range(6):
        angle = k * np.pi / 3
        hex_pts.append([2*np.cos(angle), 2*np.sin(angle)])
    for k in range(3):
        angle = (2*k + 1) * np.pi / 6
        hex_pts.append([2*np.cos(angle), 2*np.sin(angle)])
    
    hex_pts = np.array(hex_pts[:16], dtype=float)
    
    try:
        res = minimize(
            _objective,
            hex_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_pts = res.x.reshape(N, D)
    except Exception:
        pass
    
    # Strategy 4: Square lattice initialization
    side = 4  # 4x4 grid
    square_pts = []
    for i in range(side):
        for j in range(side):
            square_pts.append([i, j])
    square_pts = np.array(square_pts[:16], dtype=float)
    
    try:
        res = minimize(
            _objective,
            square_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_pts = res.x.reshape(N, D)
    except Exception:
        pass
    
    # Final check
    if best_pts is not None:
        return best_pts
    
    # Fallback to grid
    import itertools
    side = int(np.ceil(N ** (1 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    return np.array(grid, dtype=float)

# EVOLVE-BLOCK-END
