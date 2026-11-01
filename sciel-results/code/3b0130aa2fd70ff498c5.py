"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


def _score(pts):
    """Compute the packing radius for a set of points."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to container boundary
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    if np.any(boundary < -1e-9):
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    r = min(np.min(boundary), min_pair)
    if r < 0:
        return 0.0
    return r


def _init_ring_structure():
    """
    Initialize 50 circles in a ring-like structure that resembles known optimal packings.
    For N=50, a good structure is multiple concentric rings.
    
    Known best packing for 50 circles in a unit circle has radius ~0.1258.
    Structure: 1 center, then rings of 6, 12, 18, 13 (total = 1+6+12+18+13 = 50)
    or similar arrangements.
    
    Let's use: 1 center, ring of 6, ring of 12, ring of 18, ring of 13
    Actually, let me think about this more carefully.
    
    For optimal packing of 50 equal circles:
    - 1 at center
    - 6 in first ring (hexagonal)
    - 12 in second ring
    - 18 in third ring  
    - 13 in fourth ring
    Total: 1+6+12+18+13 = 50
    
    Let's initialize with approximate positions.
    """
    pts = []
    
    # Center
    pts.append([0.0, 0.0])
    
    # Ring 1: 6 circles
    r1 = 0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 12 circles
    r2 = 0.45
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12  # offset
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Ring 3: 18 circles
    r3 = 0.65
    for i in range(18):
        angle = 2 * np.pi * i / 18
        pts.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # Ring 4: 13 circles
    r4 = 0.82
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi / 13
        pts.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    return np.array(pts[:N])


def _objective(x, target_r):
    """
    Objective function: minimize violation of constraints.
    x: flattened array of 2N coordinates
    target_r: the radius we want to achieve
    
    We want all pairwise distances >= 2*target_r and all points within (1-target_r) of origin.
    """
    pts = x.reshape(N, 2)
    
    # Boundary violations: want |pts[i]| <= 1 - target_r
    boundary_dist = np.linalg.norm(pts, axis=1)
    boundary_viol = np.maximum(0, boundary_dist - (1.0 - target_r))
    
    # Pairwise distance violations: want |pts[i] - pts[j]| >= 2*target_r
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_viol = np.maximum(0, 2 * target_r - dists)
    
    # Sum of squares of violations
    obj = np.sum(boundary_viol ** 2) + np.sum(pair_viol ** 2)
    return obj


def _try_optimize_from_init(init_pts, time_limit):
    """
    Try to optimize from an initial configuration using binary search on radius
    and L-BFGS-B to find feasible positions.
    """
    best_pts = init_pts.copy()
    best_r = _score(best_pts)
    
    if best_r <= 0:
        return best_pts, best_r
    
    # Binary search on radius
    # Lower bound: current score
    # Upper bound: theoretical max (for N=50 in unit circle, max r < 0.13)
    r_low = best_r
    r_high = 0.13  # slightly above known best
    
    max_iters = 20
    for it in range(max_iters):
        if time.time() > time_limit:
            break
        
        r_mid = (r_low + r_high) / 2
        
        # Try to find feasible configuration for radius r_mid
        x0 = best_pts.flatten()
        
        # Use L-BFGS-B to minimize violations
        result = minimize(
            _objective,
            x0,
            args=(r_mid,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.success or result.fun < 1e-8:
            # Check if this is actually feasible
            candidate_pts = result.x.reshape(N, 2)
            candidate_r = _score(candidate_pts)
            if candidate_r >= r_mid - 1e-9:
                # Feasible! Update
                best_pts = candidate_pts
                best_r = max(best_r, candidate_r)
                r_low = r_mid
            else:
                r_high = r_mid
        else:
            r_high = r_mid
        
        # Also try gradient-based local improvement
        if time.time() > time_limit:
            break
        
        # Local improvement via direct optimization of the score
        def _neg_score(x):
            pts = x.reshape(N, 2)
            return -_score(pts)
        
        # Try a few random perturbations
        for _ in range(3):
            if time.time() > time_limit:
                break
            
            x_pert = best_pts.flatten() + 0.01 * np.random.randn(2 * N)
            res = minimize(
                _neg_score,
                x_pert,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res.fun < -best_r:
                candidate_pts = res.x.reshape(N, 2)
                candidate_r = _score(candidate_pts)
                if candidate_r > best_r:
                    best_pts = candidate_pts
                    best_r = candidate_r
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Multiple initial configurations to try
    inits = []
    
    # Ring structure
    inits.append(_init_ring_structure())
    
    # Hexagonal lattice (from seed)
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            inits.append(points[np.argsort(radius, kind="stable")[:N]])
            break
        spacing *= 0.98
    
    # Random initializations
    np.random.seed(42)
    for _ in range(5):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.random.uniform(0, 0.95, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)
    
    best_pts = None
    best_r = 0.0
    
    for init in inits:
        if time.time() > deadline:
            break
        
        pts, r = _try_optimize_from_init(init, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    if best_pts is None:
        # Fallback: return the first init
        best_pts = inits[0]
        best_r = _score(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
