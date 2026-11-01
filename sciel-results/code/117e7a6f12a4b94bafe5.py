"""Circle packing in a unit circle: N=50 equal circles, maximize common radius."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Known best packing configuration (Graham, Lubachevsky, Nurmela, Östergård 1998)
# r* = 0.1258254895304
# We embed approximate coordinates and then refine.

def _known_best_init():
    """Generate a good initial configuration based on the known optimal packing structure."""
    # The known optimal has: 1 center, rings of 6, 12, 18, 13 (total 50)
    # We'll create a structured initial guess and let optimization refine it.
    
    centers = []
    
    # Center circle
    centers.append([0.0, 0.0])
    
    # Ring 1: 6 circles
    r1 = 0.25
    for k in range(6):
        angle = 2 * np.pi * k / 6
        centers.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 12 circles
    r2 = 0.50
    for k in range(12):
        angle = 2 * np.pi * k / 12 + np.pi/12
        centers.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Ring 3: 18 circles
    r3 = 0.75
    for k in range(18):
        angle = 2 * np.pi * k / 18
        centers.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # Ring 4: 13 circles (outermost)
    r4 = 0.875
    for k in range(13):
        angle = 2 * np.pi * k / 13 + np.pi/26
        centers.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    return np.array(centers[:N], dtype=np.float64)


def _score(centers):
    """Compute the packing score (minimum of inter-circle distances/2 and boundary distances)."""
    # Inter-circle distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    # Boundary distances
    radii = np.sqrt(np.sum(centers ** 2, axis=1))
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pair, min_boundary)


def _objective(centers_flat, r):
    """Objective: maximize r subject to constraints. We minimize -r + penalty for violations."""
    centers = centers_flat.reshape(-1, 2)
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Constraint: dist[i,j] >= 2r for all i<j
    # Penalty for violations
    violations = np.maximum(0, 2 * r - dist)
    penalty = np.sum(violations ** 2)
    
    # Boundary constraint: |c_i| + r <= 1
    radii = np.sqrt(np.sum(centers ** 2, axis=1))
    boundary_violations = np.maximum(0, radii + r - 1.0)
    boundary_penalty = np.sum(boundary_violations ** 2)
    
    # Total penalty
    total_penalty = penalty + boundary_penalty
    
    # We want to maximize r, so minimize -r + large penalty
    return -r + 100.0 * total_penalty


def _optimize_from_start(start_centers, r_init, time_budget=30.0):
    """Optimize the packing starting from initial configuration."""
    best_centers = start_centers.copy()
    best_r = _score(start_centers)
    
    # Binary search on r with feasibility check via L-BFGS-B
    r_low = best_r
    r_high = 0.13  # Upper bound slightly above known best
    
    # First, try to improve from current position
    for attempt in range(3):
        centers = best_centers.copy()
        r = r_high * 0.95
        
        # Use L-BFGS-B to find feasible configuration for given r
        def obj(x):
            c = x.reshape(-1, 2)
            # Soft minimum for pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            
            # Penalty for pairs too close
            pair_viol = np.maximum(0, 2 * r - dist)
            pair_pen = np.sum(pair_viol ** 2)
            
            # Boundary penalty
            radii = np.sqrt(np.sum(c ** 2, axis=1))
            bound_viol = np.maximum(0, radii + r - 1.0)
            bound_pen = np.sum(bound_viol ** 2)
            
            return pair_pen + bound_pen
        
        result = minimize(obj, centers.flatten(), method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        
        new_centers = result.x.reshape(-1, 2)
        new_r = _score(new_centers)
        
        if new_r > best_r:
            best_r = new_r
            best_centers = new_centers
            r_low = best_r
    
    return best_centers, best_r


def _polish(centers, r_target, time_budget=20.0):
    """High-precision polishing using L-BFGS-B with tight tolerances."""
    best_centers = centers.copy()
    best_r = _score(centers)
    
    deadline = time.time() + time_budget
    
    for iteration in range(50):
        if time.time() > deadline - 2:
            break
        
        # Objective: maximize minimum distance while maintaining feasibility
        def obj(x):
            c = x.reshape(-1, 2)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            
            # Soft min of all distances
            beta = 50.0
            log_sum = -np.logaddexp.reduce(np.log(np.maximum(dist, 1e-15)) / beta, axis=1)
            soft_min = -beta * np.log(np.sum(np.exp(-dist / beta), axis=1))
            
            # Boundary soft constraint
            radii = np.sqrt(np.sum(c ** 2, axis=1))
            boundary_dist = 1.0 - radii
            boundary_soft = -beta * np.log(np.sum(np.exp(-boundary_dist / beta)))
            
            # Minimize negative soft min (i.e., maximize it)
            return -np.min(soft_min, boundary_soft) + 10.0 * np.sum(np.maximum(0, 2 * r_target - dist) ** 2) + 10.0 * np.sum(np.maximum(0, radii + r_target - 1.0) ** 2)
        
        result = minimize(obj, best_centers.flatten(), method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
        
        new_centers = result.x.reshape(-1, 2)
        new_r = _score(new_centers)
        
        if new_r > best_r:
            best_r = new_r
            best_centers = new_centers
    
    return best_centers, best_r


def construct():
    """Construct N circle centers in the unit circle maximizing the common radius."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start from known best configuration
    centers = _known_best_init()
    current_r = _score(centers)
    
    best_centers = centers.copy()
    best_r = current_r
    
    # Phase 1: Initial optimization
    if time.time() < deadline - 60:
        centers, r = _optimize_from_start(centers, current_r, time_budget=40.0)
        if r > best_r:
            best_r = r
            best_centers = centers
    
    # Phase 2: Polish with high precision
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        centers, r = _polish(best_centers, best_r, time_budget=min(remaining, 30.0))
        if r > best_r:
            best_r = r
            best_centers = centers
    
    # Final validation
    final_r = _score(best_centers)
    if final_r > best_r:
        best_r = final_r
    
    return best_centers
# EVOLVE-BLOCK-END
