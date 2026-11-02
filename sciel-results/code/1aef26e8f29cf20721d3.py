"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import cdist
import time

def _compute_radius(pts):
    """Compute the achievable radius for a set of points."""
    if len(pts) == 0:
        return 0.0
    # Boundary constraint
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    # Pairwise constraints
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    pair_min = d.min() / 2.0
    return min(boundary.min(), pair_min)

def _hexagonal_initial(n_points):
    """Generate a hexagonal lattice initial configuration."""
    # Estimate spacing
    spacing = 2.0 / (np.sqrt(n_points) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        points = np.array(rows)
        if len(points) >= n_points:
            # Sort by radius and take first n
            radius = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radius, kind='stable')[:n_points]
            return points[idx]
        spacing *= 0.98

def _refine_with_optimization(pts, max_time=100.0):
    """
    Refine point positions using sequential quadratic programming-like approach.
    We maximize the minimum radius by iteratively moving points that are closest to boundary or each other.
    """
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    # Precompute initial radius
    current_r = _compute_radius(pts)
    
    # We'll do iterative refinement
    # Strategy: repeatedly find the pair of points that are closest (or point closest to boundary)
    # and move them apart
    
    for iteration in range(5000):
        if time.time() - start_time > max_time:
            break
        
        # Compute current radius
        boundary = 1.0 - np.linalg.norm(pts, axis=1)
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        
        # Find the minimum pairwise distance
        min_pair_dist = d.min()
        min_pair_idx = np.unravel_index(d.argmin(), d.shape)
        i, j = min_pair_idx
        
        # Find the point closest to boundary
        min_boundary_idx = boundary.argmin()
        min_boundary_val = boundary[min_boundary_idx]
        
        # Decide which constraint is binding
        if min_pair_dist / 2.0 < min_boundary_val:
            # Pairwise constraint is binding
            # Move points i and j apart
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                # Random perturbation
                diff = np.random.randn(2)
                dist = np.linalg.norm(diff)
            direction = diff / dist
            # Move each point half the needed distance
            needed = max(0.0, 2.0 * current_r - dist)
            if needed > 0:
                pts[i] += 0.5 * needed * direction
                pts[j] -= 0.5 * needed * direction
        else:
            # Boundary constraint is binding
            # Move point min_boundary_idx inward
            p = pts[min_boundary_idx]
            norm = np.linalg.norm(p)
            if norm < 1e-12:
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
            else:
                direction = -p / norm  # Move toward center
            needed = max(0.0, 1.0 - current_r - norm)
            if needed > 0:
                pts[min_boundary_idx] += needed * direction
        
        # Recompute radius
        new_r = _compute_radius(pts)
        if new_r > current_r:
            current_r = new_r
        else:
            # Revert if it got worse
            # Actually, let's just continue; the moves are small
            pass
    
    return pts, current_r

def _final_polish(pts, max_time=20.0):
    """
    Final high-precision polish using scipy.optimize.minimize on a smooth surrogate.
    We use a soft-min approximation to make the objective differentiable.
    """
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    # Use a log-sum-exp based soft-min for the radius
    # r = min(min_i (1 - |p_i|), min_{i<j} |p_i - p_j| / 2)
    # We'll use a parameter beta for the soft-min
    
    beta = 50.0  # Larger beta = closer to true min
    
    def soft_min(values, beta):
        """Compute soft minimum using log-sum-exp."""
        # soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
        # For numerical stability
        x = np.array(values)
        min_x = x.min()
        exp_terms = np.exp(-beta * (x - min_x))
        return min_x - np.log(exp_terms.sum()) / beta
    
    def objective_and_grad(x):
        """
        Objective: negative of soft-min radius (we minimize this)
        x is flattened array of shape (2*n,)
        """
        if time.time() - start_time > max_time:
            return 0.0, np.zeros(2 * n)
        
        p = x.reshape(n, 2)
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        boundary = 1.0 - norms
        
        # Pairwise distances
        d = cdist(p, p)
        np.fill_diagonal(d, np.inf)
        pair_dists = d / 2.0
        
        # Collect all constraints
        all_vals = np.concatenate([boundary, pair_dists.ravel()])
        
        # Soft min
        r_soft = soft_min(all_vals, beta)
        
        # Gradient
        # d(r_soft)/d(x) = -1/beta * sum(exp(-beta*(v_i - min_v)) * (-beta * dv_i/dx)) / sum(exp(...))
        # = sum(exp(-beta*(v_i - min_v)) * dv_i/dx) / sum(exp(...))
        
        min_v = all_vals.min()
        exp_terms = np.exp(-beta * (all_vals - min_v))
        weights = exp_terms / exp_terms.sum()
        
        grad = np.zeros(2 * n)
        
        # Boundary gradients
        for i in range(n):
            norm_i = norms[i]
            if norm_i > 1e-12:
                # d(1 - |p_i|)/d(p_i) = -p_i / |p_i|
                grad[2*i:2*i+2] += weights[i] * (-p[i] / norm_i)
        
        # Pairwise gradients
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                w = weights[n + idx]
                diff = p[i] - p[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-12:
                    # d(|p_i - p_j|/2)/d(p_i) = (p_i - p_j) / (2 * |p_i - p_j|)
                    grad[2*i:2*i+2] += w * (diff / (2.0 * dist))
                    # d(|p_i - p_j|/2)/d(p_j) = -(p_i - p_j) / (2 * |p_i - p_j|)
                    grad[2*j:2*j+2] += w * (-diff / (2.0 * dist))
                idx += 1
        
        return -r_soft, grad
    
    # Initial point
    x0 = pts.ravel()
    
    # Run optimization
    result = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                      jac=True, options={'maxiter': 100, 'gtol': 1e-8})
    
    if result.success or result.fun < objective_and_grad(x0)[0]:
        new_pts = result.x.reshape(n, 2)
        new_r = _compute_radius(new_pts)
        old_r = _compute_radius(pts)
        if new_r >= old_r - 1e-10:
            return new_pts
    return pts

def construct():
    """
    Construct 300 equal non-overlapping circles in the unit disk.
    Returns array of shape (300, 2) with circle centers.
    """
    total_deadline = 150.0  # Safety margin before 150s limit
    start_time = time.time()
    
    # Phase 1: Generate initial hexagonal configuration
    pts = _hexagonal_initial(N)
    
    # Phase 2: Iterative refinement
    pts, current_r = _refine_with_optimization(pts, max_time=100.0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < total_deadline - 30.0:
        # Phase 3: Final polish with smooth optimization
        pts = _final_polish(pts, max_time=25.0)
    
    # Ensure we have exactly N points
    if len(pts) != N:
        # This shouldn't happen, but handle gracefully
        if len(pts) > N:
            pts = pts[:N]
        else:
            # Pad with points near center (should not happen)
            pad = np.zeros((N - len(pts), 2))
            pts = np.vstack([pts, pad])
    
    return pts
# EVOLVE-BLOCK-END
