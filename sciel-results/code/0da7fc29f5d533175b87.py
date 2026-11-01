"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import math

N = 100

# EVOLVE-BLOCK-START
def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    and its gradient with respect to pts (N x 3).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    dist_safe = np.maximum(dist, 1e-15)
    
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -math.fsum(np.log(d_upper).tolist())
    
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    W = np.zeros_like(dist_sq_safe)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq_safe
    np.fill_diagonal(W, 0.0)
    
    row_sum = W.sum(axis=1)
    grad = -pts * row_sum[:, np.newaxis] + W @ pts
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _mesh_energy_and_grad(pts, edges, target_lengths):
    """
    Compute mesh regularization energy: sum (d_ij - L_ij)^2
    and its gradient.
    """
    n = pts.shape[0]
    energy = 0.0
    grad = np.zeros_like(pts)
    
    # Precompute distances for edges
    # For efficiency, we can just loop over edges if not too many
    # Delaunay on 100 points has ~300 edges. Looping is fine.
    
    for (i, j) in edges:
        diff = pts[i] - pts[j]
        d = np.linalg.norm(diff)
        d_safe = max(d, 1e-15)
        L = target_lengths[(i, j)]
        
        # Energy term: (d - L)^2
        energy += (d - L) ** 2
        
        # Gradient w.r.t pts[i] and pts[j]
        # d/dd (d - L)^2 = 2(d - L)
        # d/dd d = (pts[i] - pts[j]) / d
        # grad_i += 2(d - L) * (pts[i] - pts[j]) / d
        # grad_j += 2(d - L) * (pts[j] - pts[i]) / d
        
        factor = 2.0 * (d - L) / d_safe
        grad[i] += factor * diff
        grad[j] -= factor * diff
        
    return energy, grad

def _get_mesh_edges_and_targets(pts):
    """
    Compute Delaunay triangulation edges and target lengths (mean edge length).
    """
    # Delaunay in 3D is tricky for points on a sphere.
    # We can use a k-NN graph instead, which is more robust for spherical codes.
    # k=6 is typical for triangular tiling.
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    distances, indices = tree.query(pts, k=7) # k=7 includes self, so k-1=6 neighbors
    
    edges = set()
    for i in range(len(pts)):
        for j in indices[i, 1:]: # skip self
            if i < j:
                edges.add((i, j))
            else:
                edges.add((j, i))
    
    edges = list(edges)
    
    # Calculate mean edge length
    lengths = []
    for i, j in edges:
        d = np.linalg.norm(pts[i] - pts[j])
        lengths.append(d)
    mean_len = np.mean(lengths)
    
    targets = {}
    for i, j in edges:
        targets[(i, j)] = mean_len
        targets[(j, i)] = mean_len
        
    return edges, targets

def _riemannian_gradient_descent(pts, max_iter=2000, lr=0.01, edges=None, targets=None, lambda_mesh=1.0):
    """
    Riemannian Gradient Descent with optional mesh regularization.
    """
    pts = _project_to_sphere(pts.copy())
    
    for k in range(max_iter):
        # Compute total energy and gradient
        e_log, g_log = _log_energy_and_grad(pts)
        
        e_mesh = 0.0
        g_mesh = np.zeros_like(pts)
        if edges is not None and targets is not None and lambda_mesh > 0:
            e_mesh, g_mesh = _mesh_energy_and_grad(pts, edges, targets)
            e_total = e_log + lambda_mesh * e_mesh
            g_total = g_log + lambda_mesh * g_mesh
        else:
            e_total = e_log
            g_total = g_log
            
        # Project gradient to tangent space
        grad_dot = np.einsum('ij,ij->i', g_total, pts)
        g_tangent = g_total - grad_dot[:, np.newaxis] * pts
        
        # Step
        pts_new = pts - lr * g_tangent
        pts_new = _project_to_sphere(pts_new)
        
        # Simple line search: accept if energy decreases
        e_new_log, _ = _log_energy_and_grad(pts_new)
        if edges is not None and targets is not None and lambda_mesh > 0:
            e_new_mesh, _ = _mesh_energy_and_grad(pts_new, edges, targets)
            e_new_total = e_new_log + lambda_mesh * e_new_mesh
        else:
            e_new_total = e_new_log
            
        if e_new_total < e_total:
            pts = pts_new
            # Decay learning rate
            lr *= 0.99
        else:
            lr *= 0.5
            
        if lr < 1e-8:
            break
            
    return pts, e_total

def _fibonacci_spiral(n=N, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    # 1. Initial configuration: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    
    # 2. Define mesh structure based on initial configuration
    edges, targets = _get_mesh_edges_and_targets(pts)
    
    # 3. Phase 1: Optimize with Mesh Regularization
    # This helps maintain a "good" structural distribution
    pts, _ = _riemannian_gradient_descent(
        pts, 
        max_iter=1000, 
        lr=0.05, 
        edges=edges, 
        targets=targets, 
        lambda_mesh=1.0
    )
    
    # 4. Phase 2: Pure Log Energy Optimization
    # Use RCG or similar for high precision
    # We can reuse the logic from the original code or just do more GD steps
    
    # Let's do a few rounds of GD with decreasing lambda_mesh
    for lam in [0.1, 0.01, 0.0]:
        pts, _ = _riemannian_gradient_descent(
            pts, 
            max_iter=1000, 
            lr=0.01, 
            edges=edges, 
            targets=targets, 
            lambda_mesh=lam
        )
        
    # 5. Final Refinement with RCG (from original code logic, simplified)
    # We can just call a standard RCG implementation here
    # For brevity and to ensure it works, we'll do a final GD pass with very small steps
    
    pts, _ = _riemannian_gradient_descent(
        pts, 
        max_iter=2000, 
        lr=0.001, 
        edges=None, 
        targets=None, 
        lambda_mesh=0.0
    )
    
    # Ensure valid output
    pts = _project_to_sphere(pts)
    
    # Check for close points
    from scipy.spatial import distance_matrix
    dist = distance_matrix(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        pts = _fibonacci_spiral(N)
        
    if not np.all(np.isfinite(pts)):
        pts = _fibonacci_spiral(N)
        
    return pts
# EVOLVE-BLOCK-END
