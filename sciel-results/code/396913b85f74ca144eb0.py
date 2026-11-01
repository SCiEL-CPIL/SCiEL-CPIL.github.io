"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time


def _log_energy_vectorized(points):
    """Compute logarithmic energy using vectorized NumPy operations."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Use upper triangle only (i < j)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clip to avoid log(0)
    d = np.maximum(d, 1e-15)
    return math.fsum(math.log(1.0 / di) for di in d)


def _project_to_sphere(points):
    """Project each point to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms


def _initial_fibonacci():
    """Generate initial Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _fekeete_greedy():
    """
    Construct points using Fekete's greedy algorithm.
    Each new point is chosen to maximize the minimum distance to existing points.
    """
    points = []
    # Start with a point at the north pole
    points.append(np.array([0.0, 0.0, 1.0]))
    
    for k in range(1, N):
        # We need to find a point on the sphere that maximizes min distance to points[0:k]
        # This is a non-convex problem. We use a grid search + local refinement.
        
        # Coarse grid search
        best_pt = None
        best_min_dist = -1.0
        
        # Use a Fibonacci spiral for the search grid to cover the sphere well
        n_grid = 2000
        i_grid = np.arange(n_grid) + 0.5
        z_grid = 1.0 - 2.0 * i_grid / n_grid
        phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
        r_grid = np.sqrt(1.0 - z_grid * z_grid)
        grid_pts = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
        
        existing = np.array(points)
        
        # Compute min distance from each grid point to existing points
        # diff shape: (n_grid, k, 3)
        diff = grid_pts[:, np.newaxis, :] - existing[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        min_dists = np.min(dists, axis=1)
        
        idx = np.argmax(min_dists)
        best_pt = grid_pts[idx]
        best_min_dist = min_dists[idx]
        
        # Local refinement using gradient ascent on min_dist
        # We want to maximize min_j |p - p_j|
        # This is non-smooth. We can use the gradient of the squared min distance?
        # Simpler: Use a few steps of "move away from closest point".
        
        for _ in range(50):
            # Find closest existing point
            d_to_existing = np.linalg.norm(existing - best_pt, axis=1)
            idx_closest = np.argmin(d_to_existing)
            closest_pt = existing[idx_closest]
            
            # Direction to move: away from closest_pt
            # On sphere, move along the geodesic away from closest_pt
            # Vector v = best_pt - closest_pt. Normalize.
            v = best_pt - closest_pt
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-9:
                break
            v = v / norm_v
            
            # Move a small step
            step = 0.01
            new_pt = best_pt + step * v
            new_pt = _project_to_sphere(new_pt)
            
            # Check if this improved the min distance
            new_dists = np.linalg.norm(existing - new_pt, axis=1)
            new_min_dist = np.min(new_dists)
            
            if new_min_dist > best_min_dist:
                best_pt = new_pt
                best_min_dist = new_min_dist
            else:
                # If it didn't improve, try a smaller step or break?
                # For simplicity, break if no improvement
                break
        
        points.append(best_pt)
        
    return np.array(points)


def _ripley_refine(points, n_iter=10, n_steps=100):
    """
    Refine points using Ripley's algorithm.
    Iteratively move each point to maximize the product of distances to all other points.
    This is equivalent to minimizing the sum of log(1/d) for that point's contribution.
    """
    pts = points.copy()
    
    for it in range(n_iter):
        for i in range(N):
            # Fix all points except i
            others = np.delete(pts, i, axis=0)
            p = pts[i].copy()
            
            # We want to maximize sum_{j} log(1/|p - p_j|) = - sum log|p - p_j|
            # Let f(p) = sum_{j} log|p - p_j|. We want to maximize f(p).
            # Gradient of f(p) w.r.t p is sum_{j} (p - p_j) / |p - p_j|^2
            # This is the same as the gradient of the energy term for point i.
            
            for step in range(n_steps):
                # Compute gradient of log product of distances
                diff = p - others
                dists = np.linalg.norm(diff, axis=1)
                # Avoid division by zero
                dists = np.maximum(dists, 1e-15)
                
                # grad = sum (p - p_j) / |p - p_j|^2
                # This points in the direction of increasing log product?
                # d/dp log|p-p_j| = (p-p_j)/|p-p_j|^2
                # Yes, gradient ascent on sum log|p-p_j|
                
                grad = np.sum(diff / dists[:, np.newaxis], axis=0)
                
                # Project gradient to tangent space
                grad_tan = grad - np.dot(grad, p) * p
                
                # Normalize gradient step
                norm_grad = np.linalg.norm(grad_tan)
                if norm_grad < 1e-10:
                    break
                
                # Step size: adaptive? Let's use a small fixed step for stability
                step_size = 0.05
                new_p = p + step_size * grad_tan
                new_p = _project_to_sphere(new_p)
                
                # Check improvement
                # Energy contribution of p is sum log(1/|p - p_j|)
                # We want to minimize this, so maximize sum log|p - p_j|
                old_val = np.sum(np.log(dists))
                new_dists = np.linalg.norm(new_p - others, axis=1)
                new_dists = np.maximum(new_dists, 1e-15)
                new_val = np.sum(np.log(new_dists))
                
                if new_val > old_val:
                    p = new_p
                else:
                    # Try smaller step
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break
                    new_p = p + step_size * grad_tan
                    new_p = _project_to_sphere(new_p)
                    new_dists = np.linalg.norm(new_p - others, axis=1)
                    new_dists = np.maximum(new_dists, 1e-15)
                    new_val = np.sum(np.log(new_dists))
                    if new_val > old_val:
                        p = new_p
                    else:
                        break
            
            pts[i] = p
    
    return pts


def construct():
    start_time = time.time()
    
    best_pts = None
    best_energy = float('inf')
    
    # Candidate 1: Fibonacci Spiral + Ripley Refinement
    init_fib = _initial_fibonacci()
    pts1 = _ripley_refine(init_fib, n_iter=5, n_steps=50)
    e1 = _log_energy_vectorized(pts1)
    if e1 < best_energy:
        best_energy = e1
        best_pts = pts1.copy()
        
    # Candidate 2: Fekete Greedy + Ripley Refinement
    # Fekete is slower but provides a different global structure
    if time.time() - start_time < 30:
        init_fekeete = _fekeete_greedy()
        pts2 = _ripley_refine(init_fekeete, n_iter=5, n_steps=50)
        e2 = _log_energy_vectorized(pts2)
        if e2 < best_energy:
            best_energy = e2
            best_pts = pts2.copy()

    # Candidate 3: Random Init + Ripley Refinement (Multiple seeds)
    if time.time() - start_time < 60:
        for seed in [42, 123, 999]:
            rng = np.random.default_rng(seed)
            pts_rand = rng.normal(size=(N, 3))
            pts_rand = _project_to_sphere(pts_rand)
            pts3 = _ripley_refine(pts_rand, n_iter=3, n_steps=30)
            e3 = _log_energy_vectorized(pts3)
            if e3 < best_energy:
                best_energy = e3
                best_pts = pts3.copy()
            if time.time() - start_time > 90:
                break

    # Final Polish with L-BFGS if time permits
    if best_pts is not None and time.time() - start_time < 100:
        try:
            def objective(x):
                pts = _project_to_sphere(x.reshape(N, 3))
                return _log_energy_vectorized(pts)
            
            def gradient(x):
                pts = _project_to_sphere(x.reshape(N, 3))
                # Compute gradient of energy
                n = len(pts)
                grad = np.zeros_like(pts)
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff ** 2, axis=2))
                dist = np.maximum(dist, 1e-15)
                term = -diff / (dist ** 2)[:, :, np.newaxis]
                for i in range(n):
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    grad[i] = np.sum(term[i, mask], axis=0)
                # Project to tangent space
                for i in range(n):
                    grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
                return grad.flatten()
            
            x0 = best_pts.flatten()
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
            final_pts = _project_to_sphere(res.x.reshape(N, 3))
            e_final = _log_energy_vectorized(final_pts)
            if e_final < best_energy:
                best_pts = final_pts
                best_energy = e_final
        except Exception:
            pass

    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                rng = np.random.default_rng(42 + i)
                best_pts[j] += 1e-6 * rng.randn(3)
                best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
