"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_radius(pts):
    """Compute the common radius for a set of circle centers."""
    if len(pts) < 2:
        return 1.0
    # Distance to boundary
    dist_to_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    r_boundary = np.min(dist_to_boundary)
    
    # Minimum pairwise distance
    from scipy.spatial.distance import pdist
    min_pair_dist = np.min(pdist(pts))
    r_pair = min_pair_dist / 2.0
    
    return min(r_boundary, r_pair)


def _soft_min(values, temperature=0.01):
    """Compute a smooth approximation to the minimum using log-sum-exp."""
    # Use the soft-min: -log-sum-exp(-values/T) * T
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    max_val = np.max(values)
    if max_val < -700:  # Avoid underflow
        return max_val
    exp_vals = np.exp(-values / temperature)
    log_sum = np.log(np.sum(exp_vals))
    return -temperature * log_sum + max_val


def _ring_init(n, center_frac=0.15):
    """Generate initial points using concentric rings."""
    points = []
    
    # Center point
    points.append([0.0, 0.0])
    
    # Rings
    remaining = n - 1
    ring_radius = 0.1
    while remaining > 0:
        # How many points fit in this ring
        circumference = 2 * np.pi * ring_radius
        spacing = 0.08  # Rough spacing
        count = max(1, int(circumference / spacing))
        count = min(count, remaining)
        
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for angle in angles:
            points.append([ring_radius * np.cos(angle), ring_radius * np.sin(angle)])
        
        remaining -= count
        ring_radius += 0.08
    
    points = np.array(points)
    
    # If we have more than needed, trim
    if len(points) > n:
        points = points[:n]
    elif len(points) < n:
        # Fill remaining with random points
        extra = np.random.randn(n - len(points), 2) * 0.5
        points = np.vstack([points, extra])
    
    return points


def _hex_lattice_init(n):
    """Generate initial points using a hexagonal lattice pattern."""
    points = []
    spacing = np.sqrt(np.pi / n) * 0.95
    
    max_r = 1.5
    n_max = int(max_r / spacing) + 2
    
    for j in range(-n_max, n_max + 1):
        for i in range(-n_max, n_max + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) > n:
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:n]
        points = points[idx]
    elif len(points) < n:
        spacing *= 0.9
        points = _hex_lattice_init(n)
    
    return points


def _local_optimize(pts, max_iter=100, tol=1e-8):
    """Perform local optimization to maximize the minimum distance."""
    n = len(pts)
    
    def objective(x):
        pts = x.reshape(-1, 2)
        r = _compute_radius(pts)
        return -r
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    step_size = 0.01
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            for attempt in range(10):
                delta = np.random.randn(2) * step_size
                new_pts = best_pts.copy()
                new_pts[i] += delta
                
                if np.linalg.norm(new_pts[i]) > 1.0:
                    continue
                
                new_r = _compute_radius(new_pts)
                if new_r > best_r + tol:
                    best_pts = new_pts
                    best_r = new_r
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best_pts, best_r


def _polish_lbfgsb(pts, max_iter=200, tol=1e-10):
    """Polish using L-BFGS-B with a soft-min objective."""
    n = len(pts)
    x0 = pts.flatten()
    
    def soft_objective(x):
        pts = x.reshape(-1, 2)
        # Boundary distances
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        # Pairwise distances (use a subset for speed)
        from scipy.spatial.distance import pdist
        pair_dists = pdist(pts) / 2.0
        
        # Combine all distances
        all_dists = np.concatenate([boundary_dists, pair_dists])
        
        # Soft-min approximation
        return -_soft_min(all_dists, temperature=0.005)
    
    def soft_gradient(x):
        pts = x.reshape(-1, 2)
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        from scipy.spatial.distance import pdist
        pair_dists = pdist(pts) / 2.0
        
        all_dists = np.concatenate([boundary_dists, pair_dists])
        T = 0.005
        
        # Gradient of soft-min
        exp_vals = np.exp(-all_dists / T)
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros_like(x)
        
        # Boundary gradient
        for i in range(n):
            if np.linalg.norm(pts[i]) > 1e-10:
                dir = -pts[i] / np.linalg.norm(pts[i])
            else:
                dir = np.array([0.0, 0.0])
            grad[2*i:2*i+2] += weights[i] * dir
        
        # Pairwise gradients
        pair_idx = 0
        for i in range(n):
            for j in range(i+1, n):
                w = weights[n + pair_idx]
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    grad[2*i:2*i+2] -= 0.5 * w * diff / dist
                    grad[2*j:2*j+2] += 0.5 * w * diff / dist
                pair_idx += 1
        
        return grad
    
    # Use L-BFGS-B
    result = minimize(
        soft_objective,
        x0,
        jac=soft_gradient,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
    )
    
    polished_pts = result.x.reshape(-1, 2)
    polished_r = _compute_radius(polished_pts)
    
    return polished_pts, polished_r


def construct():
    """Construct the packing of N equal circles in the unit circle."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial solution using ring-based packing
    pts = _ring_init(N)
    
    if pts.shape != (N, 2):
        pts = _hex_lattice_init(N)
    
    current_r = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = current_r
    
    # Try multiple restarts with different initial configurations
    n_restarts = 4
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            init_pts = _ring_init(N)
        elif restart == 1:
            init_pts = _hex_lattice_init(N)
        elif restart == 2:
            init_pts = _ring_init(N) + np.random.randn(N, 2) * 0.01
        else:
            init_pts = _hex_lattice_init(N) + np.random.randn(N, 2) * 0.01
        
        norms = np.linalg.norm(init_pts, axis=1)
        init_pts[norms > 0.95] *= 0.95 / norms[norms > 0.95, None]
        
        # Local optimization
        opt_pts, opt_r = _local_optimize(init_pts, max_iter=40)
        
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
    
    # Final polish using L-BFGS-B with soft-min
    if time.time() < deadline - 5:
        polished_pts, polished_r = _polish_lbfgsb(best_pts, max_iter=150, tol=1e-10)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts
    
    return best_pts

# EVOLVE-BLOCK-END
