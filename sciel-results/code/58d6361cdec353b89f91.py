"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix, cKDTree
import math

N = 100

# EVOLVE-BLOCK-START
def _log_energy(pts):
    """Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    dist_safe = np.maximum(dist, 1e-15)
    iu = np.triu_indices(N, k=1)
    d_upper = dist_safe[iu]
    return -math.fsum(np.log(d_upper).tolist())

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. pts."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    
    W = np.zeros_like(dist_sq_safe)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq_safe
    np.fill_diagonal(W, 0.0)
    
    row_sum = W.sum(axis=1)
    grad = -pts * row_sum[:, np.newaxis] + W @ pts
    return grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _build_mesh(pts, k=6):
    """Build a k-NN mesh. Returns adjacency list and edge list."""
    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=k+1)
    edges = set()
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in indices[i, 1:]:
            if i < j:
                edges.add((i, j))
                adj[i].append(j)
                adj[j].append(i)
    return adj, list(edges)

def _mesh_regularization_grad(pts, edges):
    """
    Compute gradient of mesh regularization that encourages uniform edge lengths.
    R = sum_{(i,j) in edges} (|xi - xj| - L0)^2 where L0 is mean edge length.
    """
    grad_reg = np.zeros_like(pts)
    if not edges:
        return grad_reg
    
    edge_lengths = []
    for (i, j) in edges:
        edge_lengths.append(np.linalg.norm(pts[i] - pts[j]))
    if not edge_lengths:
        return grad_reg
    L0 = np.mean(edge_lengths)
    
    for (i, j) in edges:
        diff = pts[i] - pts[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-15:
            continue
        factor = 2.0 * (dist - L0) / dist
        grad_reg[i] += factor * diff
        grad_reg[j] -= factor * diff
    
    return grad_reg

def _mesh_constrained_rgrad(pts, edges, max_iter=3000, lr_init=0.01, tol=1e-10, mesh_weight=0.1):
    """
    Riemannian Adam gradient descent with mesh structural constraints.
    """
    pts = _project_to_sphere(pts.copy())
    current_energy = _log_energy(pts)
    
    lr = lr_init
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    
    for k in range(max_iter):
        grad_energy = _log_energy_grad(pts)
        grad_mesh = _mesh_regularization_grad(pts, edges)
        total_grad = grad_energy + mesh_weight * grad_mesh
        
        grad_dot = np.einsum('ij,ij->i', total_grad, pts)
        grad_tangent = total_grad - grad_dot[:, np.newaxis] * pts
        
        t += 1
        m = beta1 * m + (1 - beta1) * grad_tangent
        v = beta2 * v + (1 - beta2) * (grad_tangent ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        step = lr * m_hat / (np.sqrt(v_hat) + eps)
        
        pts_new = pts + step
        pts_new = _project_to_sphere(pts_new)
        
        new_energy = _log_energy(pts_new)
        
        if new_energy < current_energy:
            pts = pts_new
            current_energy = new_energy
            lr = lr_init * (0.9995 ** k)
        else:
            lr *= 0.5
        
        if np.linalg.norm(grad_tangent) < tol:
            break
    
    return pts, current_energy

def _fibonacci_spiral(n=N, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    np.random.seed(42)
    
    pts = _fibonacci_spiral(N, offset=0.5)
    
    adj, edges = _build_mesh(pts, k=6)
    
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    inits = [
        _fibonacci_spiral(N, offset=0.5),
        _fibonacci_spiral(N, offset=0.0),
        _project_to_sphere(_fibonacci_spiral(N) + np.random.normal(0, 0.1, size=(N, 3))),
        _project_to_sphere(np.random.normal(size=(N, 3))),
    ]
    
    for x0 in inits:
        try:
            adj, edges = _build_mesh(x0, k=6)
            res_pts, res_energy = _mesh_constrained_rgrad(x0, edges, max_iter=5000, lr_init=0.02, tol=1e-10, mesh_weight=0.1)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    for refine_pass in range(3):
        try:
            adj, edges = _build_mesh(best_pts, k=6)
            res_pts, res_energy = _mesh_constrained_rgrad(best_pts, edges, max_iter=3000, lr_init=0.01, tol=1e-11, mesh_weight=0.05)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    best_pts = _project_to_sphere(best_pts)
    
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    dist = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist, np.inf)
    if np.min(dist) < 1e-9:
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
