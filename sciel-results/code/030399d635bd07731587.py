"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(pts):
    """Compute the packing radius for a set of points."""
    # pts: (N, 2) array
    # Radius is min of (min pairwise distance / 2, min (1 - |c_i|))
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    
    dist_to_origin = np.linalg.norm(pts, axis=1)
    min_boundary_dist = np.min(1.0 - dist_to_origin)
    
    # Check validity
    if not np.all(np.isfinite(pts)):
        return -1.0
    if np.any(dist_to_origin > 1.0):
        return -1.0
    if min_pair_dist <= 0:
        return -1.0
        
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _negative_score(pts):
    """Negative score for minimization."""
    return -_score(pts)

def _init_ring(n):
    """Initialize points in rings."""
    pts = np.zeros((n, 2))
    # Center
    # pts[0] = [0, 0]
    
    # We need to distribute n points.
    # Let's try a few ring configurations.
    # For N=50, maybe 1 + 6 + 12 + 18 + 13? Or similar.
    # A common heuristic is to place points in rings with increasing radii.
    
    # Simple initialization: random points in the unit disk.
    # But ring init is often better for starting local search.
    
    # Let's create rings with counts roughly proportional to radius.
    # Area of ring i ~ r_i^2.
    
    # Let's just use a random initialization with a few restarts,
    # or a structured one.
    
    # Structured:
    # Ring 0: 1 point at center? No, for equal circles, center is often not optimal if N is large enough to fill.
    # Actually, for N=50, the optimal packing likely has a hole in the middle or not.
    # Let's try a hexagonal lattice based init which is dense.
    
    # Or, let's try a "ring" init where we place points at specific radii and angles.
    
    # Let's try:
    # 1 point at center (maybe)
    # 6 points at r=1
    # 12 points at r=2
    # 18 points at r=3
    # 13 points at r=4
    # Total: 1+6+12+18+13 = 50.
    
    # Scale radii to fit in unit circle.
    # Max radius in hex lattice for these rings...
    # If spacing is s, radii are s, 2s, 3s, 4s.
    # We need 4s + s/2 <= 1 => s <= 2/9 ~ 0.22.
    # This is a decent start.
    
    s = 0.19 # slightly smaller to be safe
    pts = []
    
    # Ring 0
    pts.append([0.0, 0.0])
    
    # Ring 1
    for i in range(6):
        theta = 2 * np.pi * i / 6
        pts.append([s * np.cos(theta), s * np.sin(theta)])
        
    # Ring 2
    for i in range(12):
        theta = 2 * np.pi * i / 12
        pts.append([2 * s * np.cos(theta), 2 * s * np.sin(theta)])
        
    # Ring 3
    for i in range(18):
        theta = 2 * np.pi * i / 18
        pts.append([3 * s * np.cos(theta), 3 * s * np.sin(theta)])
        
    # Ring 4
    for i in range(13):
        theta = 2 * np.pi * i / 13
        pts.append([4 * s * np.cos(theta), 4 * s * np.sin(theta)])
        
    return np.array(pts[:n])

def construct():
    start_time = time.time()
    deadline = 110.0 # Leave 10s margin for 120s limit (or 150s kill)
    
    best_score = -1.0
    best_pts = None
    
    # Try a few different initializations
    inits = []
    inits.append(_init_ring(N))
    
    # Random inits
    for _ in range(5):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        inits.append(pts)
        
    # Hexagonal lattice init
    # Generate hex lattice points
    spacing = 0.2
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    if len(points) >= N:
        # Sort by radius to pick central ones? Or random?
        # Picking central ones might be good.
        idx = np.argsort(radius, kind="stable")[:N]
        inits.append(points[idx])
        
    for init_pts in inits:
        if time.time() > deadline:
            break
            
        # Local search using L-BFGS-B or Nelder-Mead
        # L-BFGS-B is good for smooth problems, but this one has non-differentiable mins.
        # Nelder-Mead is derivative-free and robust.
        
        # We'll use a simple local search: perturb points and keep if score improves.
        # Or use scipy.optimize.minimize with Nelder-Mead.
        
        # Let's use Nelder-Mead.
        # We need to flatten the points.
        x0 = init_pts.flatten()
        
        # To speed up, we can limit the number of iterations.
        # Also, we can use a custom callback to stop if time is up.
        
        result = minimize(
            _negative_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        pts = result.x.reshape(N, 2)
        score = _score(pts)
        
        if score > best_score:
            best_score = score
            best_pts = pts
            
    return best_pts
# EVOLVE-BLOCK-END
