"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_radius(points):
    """Compute the achievable common radius for given centers."""
    n = len(points)
    # Min distance between any pair / 2
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair:
                min_pair = d
    # Min distance to boundary
    min_boundary = np.min(1.0 - np.linalg.norm(points, axis=1))
    return min(min_pair / 2.0, min_boundary)


def _objective_and_gradient(x, n):
    """Objective: negative min pair distance (to maximize it).
    We use a smooth approximation via the minimum of pairwise distances."""
    points = x.reshape(n, 2)
    
    # Compute all pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Also compute boundary distances
    boundary_dists = 1.0 - np.sqrt(np.sum(points**2, axis=1))
    
    # The radius is min(min_pair_dist/2, min_boundary_dist)
    # We want to maximize this, so minimize -radius
    # Use the minimum of (dists/2) and boundary_dists
    
    # For SLSQP, we need smooth objectives. Let's minimize the negative
    # of the minimum distance, which is non-smooth. Instead, let's use
    # a different approach: minimize the sum of squared violations of
    # a target radius, or use the min directly with careful gradient.
    
    # Actually, let's just use the min and let SLSQP handle it with
    # numerical gradients if needed, or provide analytical gradient.
    
    min_val = np.inf
    idx_min = None
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d < min_val:
                min_val = d
                idx_min = (i, j, 'pair')
        bd = boundary_dists[i]
        if bd < min_val:
            min_val = bd
            idx_min = (i, -1, 'boundary')
    
    radius = min(min_val / 2.0, min_val) if idx_min[2] == 'pair' else min_val
    # Actually radius = min(min_pair/2, min_boundary)
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d < min_pair:
                min_pair = d
    min_boundary = np.min(boundary_dists)
    radius = min(min_pair / 2.0, min_boundary)
    
    # Gradient: we need d(-radius)/dx
    # This is complex; let SLSQP use numerical differentiation
    return -radius


def _constraints(n):
    """Define constraints for SLSQP."""
    cons = []
    
    # Each center must be inside the unit circle (with some margin)
    for i in range(n):
        def boundary_con(x, i=i):
            p = x.reshape(n, 2)[i]
            return 1.0 - np.linalg.norm(p) - 0.001  # small margin
        cons.append({'type': 'ineq', 'fun': boundary_con})
    
    return cons


def _initial_hex_grid(n):
    """Generate initial hexagonal grid points, select best n."""
    spacing = 2.0 / (np.sqrt(n) + 1.5)
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        points = np.array(rows)
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii <= 1.0 - spacing / 2.0 - 0.001
        points = points[mask]
        radii = radii[mask]
        if len(points) >= n:
            # Select points closest to center
            idx = np.argsort(radii, kind='stable')[:n]
            return points[idx]
        spacing *= 0.97


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Phase 1: Initial hexagonal grid
    points = _initial_hex_grid(N)
    best_radius = _compute_radius(points)
    best_points = points.copy()
    
    # Phase 2: SLSQP optimization to maximize packing radius
    # We'll maximize the minimum distance by minimizing -min_dist
    # Use a smooth surrogate: minimize the sum of (target - dist)^2 for dist < target
    # Or simply use SLSQP with the min objective
    
    x0 = best_points.flatten()
    
    # Define objective: minimize negative of the minimum achievable radius
    def obj(x):
        pts = x.reshape(N, 2)
        # Check if all inside unit circle
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms > 1.0 - 1e-10):
            return 1e6 + np.sum(norms[norms > 1.0 - 1e-10])
        
        # Compute min pair distance
        min_pair = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_pair:
                    min_pair = d
        
        min_boundary = np.min(1.0 - norms)
        radius = min(min_pair / 2.0, min_boundary)
        return -radius
    
    def obj_grad(x):
        pts = x.reshape(N, 2)
        norms = np.linalg.norm(pts, axis=1)
        
        min_pair = np.inf
        min_pair_idx = None
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_pair:
                    min_pair = d
                    min_pair_idx = (i, j)
        
        min_boundary = np.min(1.0 - norms)
        min_boundary_idx = np.argmin(1.0 - norms)
        
        radius = min(min_pair / 2.0, min_boundary)
        
        grad = np.zeros(2 * N)
        if min_pair / 2.0 <= min_boundary:
            # Gradient w.r.t. the closest pair
            i, j = min_pair_idx
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d > 1e-12:
                unit = diff / d
                grad[2*i:2*i+2] = -unit  # d(-radius)/dx_i = -d(min_pair/2)/dx_i
                grad[2*j:2*j+2] = unit
        else:
            # Gradient w.r.t. the point closest to boundary
            i = min_boundary_idx
            p = pts[i]
            norm = np.linalg.norm(p)
            if norm > 1e-12:
                unit = p / norm
                grad[2*i:2*i+2] = -unit  # d(-radius)/dx_i = -d(1-norm)/dx_i
        
        return grad
    
    # Constraints: keep points inside unit circle
    constraints = []
    for i in range(N):
        def make_constraint(i):
            def con(x):
                p = x.reshape(N, 2)[i]
                return 1.0 - np.linalg.norm(p)
            return con
        constraints.append({'type': 'ineq', 'fun': make_constraint(i)})
    
    # Bounds
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Run SLSQP
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            result = minimize(
                obj, x0, method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if result.success and result.fun < 0:
                new_points = result.x.reshape(N, 2)
                new_radius = _compute_radius(new_points)
                if new_radius > best_radius:
                    best_points = new_points
                    best_radius = new_radius
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    
    # Phase 3: Local refinement with more iterations if time allows
    if elapsed < deadline - 10:
        try:
            x0 = best_points.flatten()
            result = minimize(
                obj, x0, method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            if result.fun < 0:
                new_points = result.x.reshape(N, 2)
                new_radius = _compute_radius(new_points)
                if new_radius > best_radius:
                    best_points = new_points
                    best_radius = new_radius
        except Exception:
            pass
    
    # Validate final result
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(~np.isfinite(best_points)) or np.any(norms > 1.0 + 1e-9):
        # Fallback to hex grid
        best_points = _initial_hex_grid(N)
    
    return best_points
# EVOLVE-BLOCK-END
