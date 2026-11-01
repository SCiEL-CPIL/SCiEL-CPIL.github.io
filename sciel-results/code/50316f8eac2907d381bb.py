"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def _energy_and_grad(points):
    """Compute logarithmic energy and tangent-space gradient on the sphere."""
    if not np.all(np.isfinite(points)):
        return float('inf'), np.zeros_like(points)
    
    # Pairwise distances
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, 1.0)
    
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return float('inf'), np.zeros_like(points)
    
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient: for each pair (i,j), contribution to grad[i] and grad[j]
    # d/dx_i [-log|xi-xj|] = (xi-xj)/|xi-xj|^2
    grad = np.zeros_like(points)
    for i in range(N):
        diff = points[i] - points  # shape (N, 3)
        dist_sq = dists[i, :] ** 2
        dist_sq[i] = 1.0
        term = diff / dist_sq[:, np.newaxis]
        term[i] = 0.0
        grad[i] = -np.sum(term, axis=0)
        
        # Project onto tangent space
        dot = np.dot(grad[i], points[i])
        grad[i] -= dot * points[i]
    
    return energy, grad

def _energy(points):
    """Compute just the energy."""
    e, _ = _energy_and_grad(points)
    return e

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on the sphere."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1)
    return pts / norms[:, np.newaxis]

def _riemannian_lbfgs(points, deadline, maxiter=300, gtol=1e-8):
    """
    Riemannian L-BFGS-B optimization with correct gradient sign.
    We minimize the energy by following the negative gradient in tangent space.
    """
    if time.time() > deadline - 10.0:
        return points, _energy(points)
    
    x0 = points.flatten()
    
    def objective(x):
        if time.time() > deadline:
            return float('inf'), np.zeros_like(x)
        
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms < 1e-9):
            return float('inf'), np.zeros_like(x)
        pts = pts / norms[:, np.newaxis]
        
        e, g = _energy_and_grad(pts)
        if not np.isfinite(e):
            return float('inf'), np.zeros_like(x)
        
        return e, g.flatten()
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': 500, 'gtol': gtol,
                                'ftol': 1e-12})
        
        if res.success or np.isfinite(res.fun):
            best_pts = res.x.reshape(N, 3)
            norms = np.linalg.norm(best_pts, axis=1)
            if np.all(norms > 1e-9):
                best_pts = best_pts / norms[:, np.newaxis]
                return best_pts, res.fun
    except Exception:
        pass
    
    return points, _energy(points)

def _delaunay_relaxation(points, deadline, n_iters=50):
    """
    Structural relaxation using Delaunay triangulation on the sphere.
    Uses the graph Laplacian to enforce geometric consistency.
    """
    if time.time() > deadline - 20.0:
        return points, _energy(points)
    
    best_pts = points.copy()
    best_e = _energy(points)
    
    for it in range(n_iters):
        if time.time() > deadline - 20.0:
            break
        
        # Build Delaunay triangulation
        try:
            tri = Delaunay(best_pts)
        except Exception:
            break
        
        # Build adjacency from Delaunay faces
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = int(simplex[i]), int(simplex[j])
                    edges.add((min(a,b), max(a,b)))
        
        # Compute graph Laplacian relaxation step
        # For each vertex, move toward the average of its neighbors
        # This is a form of spectral smoothing that preserves the mesh structure
        lap_pts = best_pts.copy()
        for i in range(N):
            neighbors = []
            for (a, b) in edges:
                if a == i:
                    neighbors.append(b)
                elif b == i:
                    neighbors.append(a)
            
            if len(neighbors) > 0:
                # Move toward centroid of neighbors, but keep on sphere
                neighbor_avg = np.mean(best_pts[neighbors], axis=0)
                # Small step toward neighbor average
                step = 0.01
                new_pt = best_pts[i] + step * (neighbor_avg - best_pts[i])
                # Project back to sphere
                norm = np.linalg.norm(new_pt)
                if norm > 1e-9:
                    lap_pts[i] = new_pt / norm
        
        # Combine with gradient step
        e, g = _energy_and_grad(lap_pts)
        if np.isfinite(e) and e < best_e:
            best_pts = lap_pts.copy()
            best_e = e
        else:
            # Try smaller step
            step = 0.005
            lap_pts = best_pts.copy()
            for i in range(N):
                neighbors = []
                for (a, b) in edges:
                    if a == i:
                        neighbors.append(b)
                    elif b == i:
                        neighbors.append(a)
                
                if len(neighbors) > 0:
                    neighbor_avg = np.mean(best_pts[neighbors], axis=0)
                    new_pt = best_pts[i] + step * (neighbor_avg - best_pts[i])
                    norm = np.linalg.norm(new_pt)
                    if norm > 1e-9:
                        lap_pts[i] = new_pt / norm
            
            e, g = _energy_and_grad(lap_pts)
            if np.isfinite(e) and e < best_e:
                best_pts = lap_pts.copy()
                best_e = e
            else:
                # No improvement, try gradient-only step
                grad_pts = best_pts.copy()
                lr = 0.001
                for i in range(N):
                    new_pt = best_pts[i] - lr * g[i]
                    norm = np.linalg.norm(new_pt)
                    if norm > 1e-9:
                        grad_pts[i] = new_pt / norm
                
                e, g2 = _energy_and_grad(grad_pts)
                if np.isfinite(e) and e < best_e:
                    best_pts = grad_pts.copy()
                    best_e = e
                else:
                    break  # Stagnation
    
    return best_pts, best_e

def _multi_start_optimize(deadline):
    """
    Multi-start strategy: try multiple initial configurations and keep the best.
    Uses Fibonacci, random, and slightly perturbed Fibonacci starts.
    """
    best_points = None
    best_energy = float('inf')
    
    starts = []
    
    # Fibonacci lattice
    fib = _fibonacci_sphere(N)
    starts.append(fib)
    
    # Slightly perturbed Fibonacci
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        perturbed = fib + 0.05 * np.random.randn(N, 3)
        norms = np.linalg.norm(perturbed, axis=1)
        perturbed = perturbed / norms[:, np.newaxis]
        starts.append(perturbed)
    
    # Random starts
    for seed in [42, 123, 456]:
        starts.append(_random_sphere(N, seed=seed))
    
    for start in starts:
        if time.time() > deadline - 15.0:
            break
        
        # First do Delaunay relaxation
        if time.time() < deadline - 20.0:
            try:
                pts, e = _delaunay_relaxation(start, deadline)
                if e < best_energy:
                    best_energy = e
                    best_points = pts.copy()
            except Exception:
                pass
        
        # Then do Riemannian L-BFGS
        if time.time() < deadline - 10.0:
            try:
                pts, e = _riemannian_lbfgs(pts if best_points is None else best_points, 
                                           deadline, maxiter=200)
                if e < best_energy:
                    best_energy = e
                    best_points = pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = fib
        best_energy = _energy(fib)
    
    return best_points, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # Strategy:
    # 1. Multi-start with Fibonacci, perturbed Fibonacci, and random inits
    # 2. For each start: Delaunay graph Laplacian relaxation, then Riemannian L-BFGS
    # 3. Keep the best configuration found
    # 4. Final refinement pass on the best result
    
    best_points, best_energy = _multi_start_optimize(deadline)
    
    # Final refinement on the best result
    if time.time() < deadline - 5.0:
        try:
            refined_pts, refined_e = _riemannian_lbfgs(best_points, deadline, maxiter=100, gtol=1e-10)
            if refined_e < best_energy:
                best_energy = refined_e
                best_points = refined_pts
        except Exception:
            pass
    
    # Final validation
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_sphere(N)
    
    # Normalize
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-9):
        best_points = _fibonacci_sphere(N)
    else:
        best_points = best_points / norms[:, np.newaxis]
    
    # Check minimum distance
    dists = np.linalg.norm(best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        best_points = _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
