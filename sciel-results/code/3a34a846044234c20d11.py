"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _energy(points):
    """Calculate the logarithmic energy."""
    if not np.all(np.isfinite(points)):
        return float('inf')
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return float('inf')
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    d[d < 1e-9] = 1e-9
    energy = -np.sum(np.log(d))
    return energy

def _energy_and_grad(points):
    """Calculate the logarithmic energy and its gradient."""
    if not np.all(np.isfinite(points)):
        return float('inf'), np.zeros_like(points)
    
    # Compute pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, 1.0)
    
    # Check for invalid close points
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return float('inf'), np.zeros_like(points)
    
    # Energy
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    d[d < 1e-9] = 1e-9
    energy = -np.sum(np.log(d))
    
    # Gradient: dE/dp_i = -sum_{j!=i} (1/d_ij) * (p_i - p_j)/d_ij
    # = -sum_{j!=i} (p_i - p_j) / d_ij^2
    grad = np.zeros_like(points)
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        diffs_i = points[i] - points[mask]
        dists_i = dists[i, mask]
        dists_i[dists_i < 1e-9] = 1e-9
        # grad_i = -sum_j (p_i - p_j) / d_ij^2
        grad_i = -np.sum(diffs_i / (dists_i ** 2)[:, np.newaxis], axis=0)
        # Project to tangent space: grad_tan = grad_i - (grad_i . p_i) p_i
        grad_i = grad_i - np.dot(grad_i, points[i]) * points[i]
        grad[i] = grad_i
    
    return energy, grad

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1)
    return pts / norms[:, np.newaxis]

def _get_delaunay_edges(points):
    """Get Delaunay triangulation edges for the points on the sphere."""
    # Project to 2D or use 3D Delaunay? 
    # 3D Delaunay of points on a sphere gives a tetrahedralization. 
    # The boundary of the convex hull is the spherical Delaunay triangulation.
    # scipy.spatial.Delaunay on 3D points gives the convex hull.
    # We need the edges of the boundary faces.
    
    tri = Delaunay(points)
    # tri.simplices gives the tetrahedra.
    # We need the edges that appear in exactly one tetrahedron (boundary edges).
    # Or more simply, just use the edges of the convex hull.
    
    # Alternative: Use scipy.spatial.ConvexHull
    # But Delaunay is already computed.
    # Let's extract edges from simplices.
    edges_set = set()
    for simplex in tri.simplices:
        # 4 vertices in a tetrahedron
        for i in range(4):
            for j in range(i+1, 4):
                edges_set.add((simplex[i], simplex[j]))
    
    # This includes internal edges too. 
    # For points on a sphere, the "Delaunay triangulation" is the convex hull faces.
    # A robust way is to use ConvexHull.
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    # hull.simplices are the triangular faces on the boundary
    # Extract edges from these faces
    hull_edges_set = set()
    for face in hull.simplices:
        v0, v1, v2 = face
        hull_edges_set.add((min(v0, v1), max(v0, v1)))
        hull_edges_set.add((min(v1, v2), max(v1, v2)))
        hull_edges_set.add((min(v0, v2), max(v0, v2)))
        
    return list(hull_edges_set)

def _spectral_laplacian_flow(points, steps=500, learning_rate=0.01):
    """
    Optimize points using a Laplacian-like flow based on Delaunay neighbors.
    This enforces geometric consistency by smoothing points towards their neighbors.
    The weights are adapted from the energy gradient to minimize log-energy.
    """
    pts = points.copy()
    best_energy = _energy(pts)
    best_pts = pts.copy()
    
    for step in range(steps):
        # 1. Get Delaunay neighbors (edges)
        try:
            edges = _get_delaunay_edges(pts)
        except Exception:
            break
            
        # 2. Build adjacency structure
        # We want to compute a weighted Laplacian step.
        # The gradient of the log energy is: grad_i = - sum_{j in N(i)} (p_i - p_j) / d_ij^2
        # This looks like a Laplacian with weights w_ij = 1/d_ij^2.
        # So we can simply compute this gradient and take a step.
        # But the "Spectral" part suggests using the eigenstructure or a specific relaxation.
        # Let's stick to the gradient form which IS the weighted Laplacian action.
        
        # Compute gradient
        e, grad = _energy_and_grad(pts)
        
        # If energy is inf, restart or break
        if not np.isfinite(e):
            break
            
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            
        # 3. Take a step in the direction of the negative gradient (which is the Laplacian flow)
        # grad points in the direction of increasing energy.
        # So we want to move in the direction of -grad.
        # Note: grad is already projected to the tangent space in _energy_and_grad.
        
        # Adaptive learning rate? 
        # Start with a small step.
        step_size = learning_rate
        
        # Update
        new_pts = pts - step_size * grad
        
        # 4. Project back to sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        new_pts = new_pts / norms
        
        # Check for validity
        if not np.all(np.isfinite(new_pts)):
            break
            
        # Check min distance
        # Quick check: if any distance < 1e-9, we might have collapsed.
        # We can check this periodically or rely on the energy check.
        # Let's check min distance every 10 steps to save time.
        if step % 10 == 0:
             dists = np.linalg.norm(new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :], axis=-1)
             np.fill_diagonal(dists, 1.0)
             if np.min(dists) < 1e-9:
                 break
                 
        pts = new_pts
        
        # Early stopping if gradient is small
        if np.linalg.norm(grad) < 1e-8:
            break
            
    return best_pts, best_energy

def _optimize_from_start(start_points, max_time=30.0):
    """Run Spectral Laplacian Flow from a given starting configuration."""
    start_time = time.time()
    
    # Phase 1: Coarse Spectral Flow
    pts, energy = _spectral_laplacian_flow(start_points, steps=200, learning_rate=0.05)
    
    # Phase 2: Fine Spectral Flow
    if time.time() < start_time + max_time - 5.0:
         pts, energy = _spectral_laplacian_flow(pts, steps=500, learning_rate=0.005)
         
    # Phase 3: Final Polish with L-BFGS-B (to ensure we are at a local minimum)
    if time.time() < start_time + max_time - 2.0:
        def objective_and_grad(x):
            pts_flat = x.reshape(-1, 3)
            norms = np.linalg.norm(pts_flat, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts_flat = pts_flat / norms
            e, g = _energy_and_grad(pts_flat)
            return e, g.flatten()
            
        x0 = pts.flatten()
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
        )
        
        if result.success or result.fun < float('inf'):
            best_x = result.x
            best_pts = best_x.reshape(-1, 3)
            norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            best_pts = best_pts / norms
            best_energy = _energy(best_pts)
            if best_energy < energy:
                pts = best_pts
                energy = best_energy

    elapsed = time.time() - start_time
    return pts, energy, elapsed

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_points = None
    best_energy = float('inf')
    
    # Strategy: Use Spectral Laplacian Flow from diverse starts
    # The flow enforces a "smooth" distribution which is geometrically consistent.
    
    starts = []
    
    # Start 1: Fibonacci
    fib_pts = _fibonacci_sphere(N)
    starts.append(fib_pts)
    
    # Starts 2-8: Perturbed Fibonacci
    rng = np.random.default_rng(123)
    for scale in [0.05, 0.1, 0.2, 0.3]:
        for seed in [42, 43]:
            if time.time() > deadline - 10.0:
                break
            perturbed = fib_pts.copy()
            noise = rng.standard_normal((N, 3)) * scale
            perturbed = perturbed + noise
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            perturbed = perturbed / norms
            starts.append(perturbed)
            
    # Starts 9-11: Random
    for seed in [1, 2, 3]:
        if time.time() > deadline - 10.0:
            break
        random_pts = _random_sphere(N, seed=seed)
        starts.append(random_pts)

    # Run optimization from each start
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline - 15.0:
            break
            
        # Estimate time: Spectral flow is fast (vectorized gradient + projection).
        # Each run should take < 5s.
        pts, energy, elapsed = _optimize_from_start(start_pts, max_time=12.0)
        
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
            
    # Final polish on the best found
    if best_points is not None and time.time() < deadline - 5.0:
        def objective_and_grad_final(x):
            pts_flat = x.reshape(-1, 3)
            norms = np.linalg.norm(pts_flat, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts_flat = pts_flat / norms
            e, g = _energy_and_grad(pts_flat)
            return e, g.flatten()
            
        x0 = best_points.flatten()
        result = minimize(
            objective_and_grad_final,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
        )
        
        if result.fun < best_energy:
            best_pts = result.x.reshape(-1, 3)
            norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            best_pts = best_pts / norms
            best_energy = result.fun
            best_points = best_pts

    # Ensure valid output
    if best_points is None:
        best_points = fib_pts
        
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = fib_pts
        
    # Normalize
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
