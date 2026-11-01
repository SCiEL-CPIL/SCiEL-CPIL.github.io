"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

def _energy_and_grad(points):
    """
    Calculate the logarithmic energy and its gradient.
    E = sum_{i<j} log(1 / |x_i - x_j|) = -sum_{i<j} log(d_ij)
    d_ij = |x_i - x_j|
    
    Gradient with respect to x_k:
    grad_k = sum_{j != k} (x_k - x_j) / dist_kj^3
    """
    x_sq = np.sum(points * points, axis=1)
    dist_sq = x_sq[:, np.newaxis] + x_sq[np.newaxis, :] - 2.0 * (points @ points.T)
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    dist = np.sqrt(dist_sq)
    
    np.fill_diagonal(dist, 1.0)
    np.fill_diagonal(dist_sq, 1.0)
    
    iu = np.triu_indices(N, k=1)
    d_ij = dist[iu]
    energy = -np.sum(np.log(d_ij))
    
    C = 1.0 / (dist_sq * dist)
    np.fill_diagonal(C, 0.0)
    
    row_sums = np.sum(C, axis=1)
    grad = row_sums[:, np.newaxis] * points - (C @ points)
    
    return energy, grad

def _energy_and_grad_sparse(points, edges):
    """
    Calculate energy and gradient using only a subset of pairs (edges).
    This is an approximation used for the coarse-grained phase.
    """
    if len(edges) == 0:
        return 0.0, np.zeros_like(points)
    
    i, j = edges[:, 0], edges[:, 1]
    diff = points[i] - points[j]
    dist_sq = np.sum(diff * diff, axis=1)
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    dist = np.sqrt(dist_sq)
    
    # Energy contribution from these edges
    # Note: This is a partial energy, not the total.
    # We use it to guide the gradient, but we must be careful with the step size.
    # The gradient of the partial energy is a valid descent direction for the full energy
    # if the omitted terms are small or if we treat this as a stochastic approximation.
    
    coeff = 1.0 / (dist_sq * dist)
    
    grad = np.zeros_like(points)
    # grad_i += coeff * (x_i - x_j)
    # grad_j -= coeff * (x_i - x_j)
    
    np.add.at(grad, i, coeff[:, np.newaxis] * diff)
    np.add.at(grad, j, -coeff[:, np.newaxis] * diff)
    
    # Approximate energy for monitoring (underestimates true energy)
    energy = -np.sum(np.log(dist))
    
    return energy, grad

def _get_delaunay_edges(points):
    """
    Compute Delaunay triangulation edges for points in 3D.
    """
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    u, v = simplex[i], simplex[j]
                    if u > v: u, v = v, u
                    edges.add((u, v))
        return np.array(list(edges), dtype=int)
    except Exception:
        # Fallback to k-nearest neighbors if Delaunay fails (e.g., coplanar points)
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, inds = tree.query(points, k=10)
        edges = set()
        for i in range(N):
            for j in inds[i]:
                if j != i:
                    u, v = i, j
                    if u > v: u, v = v, u
                    edges.add((u, v))
        return np.array(list(edges), dtype=int)

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    v = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1)
    return v / norms[:, np.newaxis]

def _optimize_sparse_then_dense(points_init, time_limit):
    """
    Two-phase optimizer:
    1. Sparse gradient descent using Delaunay graph (fast, coarse).
    2. Dense L-BFGS-B polish (accurate, fine).
    """
    current = points_init.copy()
    current_energy, _ = _energy_and_grad(current)
    
    phase1_deadline = time_limit - 30.0 # Reserve 30s for polish
    
    # Phase 1: Sparse Relaxation
    step_size = 0.05
    iter_count = 0
    while time.time() < phase1_deadline:
        iter_count += 1
        # Update graph every 10 iterations to save time
        if iter_count % 10 == 1 or iter_count == 1:
            edges = _get_delaunay_edges(current)
        
        # Get sparse gradient
        e_sparse, grad_sparse = _energy_and_grad_sparse(current, edges)
        
        # Project to tangent space
        dot = np.sum(grad_sparse * current, axis=1)
        grad_tangent = grad_sparse - dot[:, np.newaxis] * current
        
        # Check convergence (rough)
        if np.linalg.norm(grad_tangent) < 1e-6:
            break
            
        # Line search with full energy evaluation
        alpha = step_size
        accepted = False
        for _ in range(10):
            new_points = current + alpha * grad_tangent
            norms = np.linalg.norm(new_points, axis=1)
            new_points = new_points / norms[:, np.newaxis]
            
            new_energy, _ = _energy_and_grad(new_points)
            
            # Armijo condition
            if new_energy < current_energy - 1e-4 * alpha * np.sum(grad_tangent * grad_tangent):
                current = new_points
                current_energy = new_energy
                accepted = True
                break
            alpha *= 0.5
            
        if not accepted:
            step_size *= 0.8
            if step_size < 1e-6:
                break

    # Phase 2: Dense Polish with L-BFGS-B
    # We use the sparse result as the initial guess for L-BFGS-B.
    # L-BFGS-B is better for the final convergence.
    
    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        e, g = _energy_and_grad(pts)
        return e, g.flatten()
        
    # We need to handle the sphere constraint.
    # L-BFGS-B in R^{3N} doesn't enforce ||x_i||=1.
    # We will use a penalty method or just run L-BFGS-B and hope it stays close?
    # Better: Use the fact that we are already on the sphere.
    # Actually, scipy minimize with method='L-BFGS-B' doesn't support manifold constraints.
    # We can use a parameterization, but that's complex.
    # Alternative: Run a few steps of projected gradient descent again but with dense gradient.
    
    # Let's do a final dense gradient descent with backtracking.
    step_size = 0.01
    while time.time() < time_limit - 5.0:
        energy, grad = _energy_and_grad(current)
        
        dot = np.sum(grad * current, axis=1)
        grad_tangent = grad - dot[:, np.newaxis] * current
        
        if np.linalg.norm(grad_tangent) < 1e-9:
            break
            
        alpha = step_size
        for _ in range(20):
            new_points = current + alpha * grad_tangent
            norms = np.linalg.norm(new_points, axis=1)
            new_points = new_points / norms[:, np.newaxis]
            
            new_energy, _ = _energy_and_grad(new_points)
            
            if new_energy < current_energy - 1e-4 * alpha * np.sum(grad_tangent * grad_tangent):
                current = new_points
                current_energy = new_energy
                break
            alpha *= 0.5
        else:
            step_size *= 0.5
            if step_size < 1e-8:
                break

    return current, current_energy

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_points = _fibonacci_sphere(N)
    best_energy, _ = _energy_and_grad(best_points)
    
    # Multi-start
    starts = [best_points]
    for i in range(8):
        starts.append(_random_sphere(N, seed=i))
        
    for start in starts:
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
            
        # Run the hybrid optimizer
        try:
            optimized, energy = _optimize_sparse_then_dense(start, deadline)
            
            # Final validation and normalization
            norms = np.linalg.norm(optimized, axis=1)
            optimized = optimized / norms[:, np.newaxis]
            
            final_energy, _ = _energy_and_grad(optimized)
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_points = optimized.copy()
                
        except Exception:
            continue
            
    return best_points
# EVOLVE-BLOCK-END
