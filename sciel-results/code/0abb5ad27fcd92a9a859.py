"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
import math

N = 100

# EVOLVE-BLOCK-START
from scipy.spatial import ConvexHull

def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    and its gradient with respect to pts (N x 3).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid log(0) by clipping
    dist_safe = np.maximum(dist, 1e-15)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -np.sum(np.log(d_upper))
    
    # Gradient: grad[i] = -sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
    dist_sq = dist_safe ** 2
    
    W = np.zeros_like(dist_sq)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    row_sum = W.sum(axis=1)
    grad = -pts * row_sum[:, np.newaxis] + W @ pts
    
    return energy, grad

def _mesh_energy_and_grad(pts, edges):
    """
    Compute mesh regularization energy and gradient.
    Energy: sum over edges (d_ij - mean_d)^2
    This encourages uniform edge lengths, promoting a rigid, regular mesh structure.
    """
    if len(edges) == 0:
        return 0.0, np.zeros_like(pts)
    
    # Compute distances for edges
    edge_dists = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
    mean_d = np.mean(edge_dists)
    
    # Energy: sum (d_ij - mean_d)^2
    energy = np.sum((edge_dists - mean_d) ** 2)
    
    # Gradient
    # dE/dd_ij = 2 * (d_ij - mean_d)
    # d(d_ij)/d(pts[i]) = (pts[i] - pts[j]) / d_ij
    # d(d_ij)/d(pts[j]) = (pts[j] - pts[i]) / d_ij
    
    grad = np.zeros_like(pts)
    d_edge_grad = 2.0 * (edge_dists - mean_d)
    
    for k in range(len(edges)):
        i, j = edges[k]
        diff = pts[i] - pts[j]
        d = edge_dists[k]
        if d < 1e-15:
            continue
        factor = d_edge_grad[k] / d
        grad[i] += factor * diff
        grad[j] -= factor * diff
        
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _riemannian_gradient_descent(pts, edges, max_iter=500, tol=1e-8, lambda_reg=0.1):
    """
    Perform Riemannian Gradient Descent on the sphere with mesh regularization.
    """
    pts = _project_to_sphere(pts.copy())
    
    # Initial energy and gradient
    e_log, g_log = _log_energy_and_grad(pts)
    e_mesh, g_mesh = _mesh_energy_and_grad(pts, edges)
    current_energy = e_log + lambda_reg * e_mesh
    current_grad = g_log + lambda_reg * g_mesh
    
    # Project gradient to tangent space
    grad_dot = np.einsum('ij,ij->i', current_grad, pts)
    grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
    
    # Initial direction
    d = -grad_tangent
    
    for k in range(max_iter):
        # Line search (Backtracking)
        alpha = 0.1
        improved = False
        for backtrack in range(30):
            # Step in tangent space
            pts_new = pts + alpha * d
            # Retraction (project to sphere)
            pts_new = _project_to_sphere(pts_new)
            
            e_log_new, g_log_new = _log_energy_and_grad(pts_new)
            e_mesh_new, g_mesh_new = _mesh_energy_and_grad(pts_new, edges)
            new_energy = e_log_new + lambda_reg * e_mesh_new
            new_grad = g_log_new + lambda_reg * g_mesh_new
            
            if new_energy < current_energy:
                pts = pts_new
                current_energy = new_energy
                current_grad = new_grad
                improved = True
                break
            else:
                alpha *= 0.5
        
        if not improved:
            break
            
        # Check convergence
        grad_dot = np.einsum('ij,ij->i', current_grad, pts)
        grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
        if np.linalg.norm(grad_tangent) < tol:
            break
            
        # Update direction (Simple Gradient Descent, no momentum/conjugate)
        d = -grad_tangent
        
    return pts, current_energy

def construct():
    # Start from a Fibonacci spiral (good initial guess)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_energy = np.inf
    best_pts = pts_init.copy()
    
    n_restarts = 10 # Reduce restarts to allow more iterations per restart
    
    for restart in range(n_restarts):
        if restart == 0:
            x0 = pts_init.copy()
        elif restart == 1:
            # Random rotation of Fibonacci
            theta = np.random.uniform(0, 2*np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
            x0 = pts_init @ rot.T
        else:
            # Random perturbation
            noise = np.random.normal(0, 0.1, size=(N, 3))
            x0 = pts_init + noise
            
        x0 = _project_to_sphere(x0)
        
        try:
            # Compute convex hull to get edges
            hull = ConvexHull(x0)
            # Extract edges from simplices
            simplices = hull.simplices
            edges_set = set()
            for simplex in simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        edge = tuple(sorted((simplex[i], simplex[j])))
                        edges_set.add(edge)
            edges = np.array(list(edges_set))
            
            res_pts, res_energy = _riemannian_gradient_descent(x0, edges, max_iter=2000, tol=1e-9, lambda_reg=0.05)
            
            # Calculate true log energy for comparison
            e_log, _ = _log_energy_and_grad(res_pts)
            if e_log < best_energy:
                best_energy = e_log
                best_pts = res_pts
        except Exception as e:
            pass

    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dist = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # If points are too close, fall back to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_pts
# EVOLVE-BLOCK-END
