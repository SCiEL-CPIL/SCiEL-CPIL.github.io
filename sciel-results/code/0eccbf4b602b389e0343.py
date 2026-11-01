"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix, Delaunay, ConvexHull
import math

N = 100

# EVOLVE-BLOCK-START
def _compute_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|xi - xj|)."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    d_upper = np.maximum(dist[iu], 1e-15)
    return -math.fsum(np.log(d_upper).tolist())

def _compute_gradient(pts):
    """Compute gradient of logarithmic energy."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    W = np.zeros_like(dist_sq_safe)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq_safe
    row_sum = W.sum(axis=1)
    grad = -pts * row_sum[:, np.newaxis] + W @ pts
    return grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_gradient(pts, grad):
    """Project gradient to tangent space of sphere."""
    grad_dot = np.einsum('ij,ij->i', grad, pts)
    return grad - grad_dot[:, np.newaxis] * pts

def _create_mesh_connectivity(pts):
    """
    Create mesh connectivity from initial points using convex hull triangulation.
    Returns adjacency dict and simplices array.
    """
    try:
        hull = ConvexHull(pts)
        simplices = hull.simplices
    except Exception:
        # Fallback: k-nearest neighbor based triangles
        dist_mat = distance_matrix(pts, pts)
        np.fill_diagonal(dist_mat, np.inf)
        k = 6
        k_nn = np.argsort(dist_mat, axis=1)[:, :k]
        simplices = []
        for i in range(len(pts)):
            for j in range(k):
                for l in range(j+1, k):
                    a, b, c = i, k_nn[i][j], k_nn[i][l]
                    if a < b < c:
                        simplices.append([a, b, c])
        simplices = np.array(simplices, dtype=int) if simplices else np.zeros((0,3), dtype=int)
    
    adj = {i: set() for i in range(len(pts))}
    for tri in simplices:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    
    return adj, simplices

def _mesh_guided_optimize(pts, adj, simplices, max_iter=5000, tol=1e-10):
    """
    Optimize points on sphere using mesh-guided gradient descent with momentum.
    """
    pts = _project_to_sphere(pts.copy())
    current_energy = _compute_energy(pts)
    current_grad = _compute_gradient(pts)
    grad_tangent = _tangent_gradient(pts, current_grad)
    
    velocity = np.zeros_like(pts)
    momentum = 0.85
    base_alpha = 0.03
    
    for k in range(max_iter):
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-15:
            break
        
        alpha = base_alpha / (1.0 + 0.05 * grad_norm)
        
        # Momentum update in tangent space
        velocity = momentum * velocity - alpha * grad_tangent
        
        # Step and project
        pts_new = pts + velocity
        pts_new = _project_to_sphere(pts_new)
        
        new_energy = _compute_energy(pts_new)
        
        if new_energy < current_energy:
            pts = pts_new
            current_energy = new_energy
            current_grad = _compute_gradient(pts)
            grad_tangent = _tangent_gradient(pts, current_grad)
        else:
            velocity *= 0.5
            base_alpha *= 0.7
            if base_alpha < 1e-9:
                break
            continue
        
        if grad_norm < tol:
            break
    
    return pts, current_energy

def _fibonacci_spiral(n=N):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _perturb_points(pts, scale=0.01):
    """Add small random perturbation to points."""
    noise = np.random.normal(0, scale, size=pts.shape)
    return _project_to_sphere(pts + noise)

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy
    using mesh-based optimization with fixed connectivity.
    """
    np.random.seed(42)
    
    # Initial configuration: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    
    # Create mesh connectivity
    adj, simplices = _create_mesh_connectivity(pts)
    
    # Phase 1: Initial optimization
    pts, energy1 = _mesh_guided_optimize(pts, adj, simplices, max_iter=3000, tol=1e-9)
    best_energy = energy1
    best_pts = pts.copy()
    
    # Phase 2: Multiple perturbation trials to escape local minima
    for trial in range(8):
        scale = 0.01 * (trial + 1)
        pts_perturbed = _perturb_points(best_pts, scale)
        
        # Recreate mesh for perturbed points
        adj_p, simplices_p = _create_mesh_connectivity(pts_perturbed)
        
        pts_opt, energy = _mesh_guided_optimize(pts_perturbed, adj_p, simplices_p, max_iter=2000, tol=1e-9)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt.copy()
    
    # Phase 3: Fine-tuning with small perturbations
    for fine in range(5):
        scale = 0.002 * (fine + 1)
        pts_perturbed = _perturb_points(best_pts, scale)
        adj_f, simplices_f = _create_mesh_connectivity(pts_perturbed)
        pts_opt, energy = _mesh_guided_optimize(pts_perturbed, adj_f, simplices_f, max_iter=3000, tol=1e-10)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt.copy()
    
    # Phase 4: Final L-BFGS-B refinement with spherical parameterization
    def sphere_to_cart(theta_phi):
        """Convert (theta, phi) to 3D Cartesian on sphere."""
        theta, phi = theta_phi[:, 0], theta_phi[:, 1]
        return np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
    
    def param_energy(x):
        theta_phi = x.reshape(N, 2)
        pts = sphere_to_cart(theta_phi)
        return _compute_energy(pts)
    
    # Initialize from best_pts
    theta = np.arccos(np.clip(best_pts[:, 2], -1.0, 1.0))
    phi = np.arctan2(best_pts[:, 1], best_pts[:, 0])
    x0 = np.column_stack([theta, phi]).flatten()
    
    try:
        res = minimize(param_energy, x0, method='L-BFGS-B',
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        pts_res = sphere_to_cart(res.x.reshape(N, 2))
        energy_res = _compute_energy(pts_res)
        if energy_res < best_energy:
            best_energy = energy_res
            best_pts = pts_res
    except Exception:
        pass
    
    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    
    dist_mat = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist_mat, np.inf)
    min_dist = np.min(dist_mat)
    
    if min_dist < 1e-9:
        best_pts = _fibonacci_spiral(N)
    
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
