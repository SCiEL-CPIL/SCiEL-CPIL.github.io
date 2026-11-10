"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100

# EVOLVE-BLOCK-START

def _fibonacci_spiral(n):
    """Generate n points on the unit sphere using a Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    # Squared distances via dot product: |x-y|^2 = 2 - 2*x.y
    dots = pts @ pts.T
    sq_dists = 2.0 - 2.0 * dots
    np.fill_diagonal(sq_dists, 1.0)  # avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-30)
    # Only upper triangle
    iu = np.triu_indices(n, 1)
    d = np.sqrt(sq_dists[iu])
    return math.fsum(math.log(1.0 / di) for di in d)


def _log_energy_fast(pts):
    """Vectorized energy computation for speed."""
    n = len(pts)
    dots = pts @ pts.T
    sq_dists = 2.0 - 2.0 * dots
    np.fill_diagonal(sq_dists, 1.0)
    sq_dists = np.maximum(sq_dists, 1e-30)
    iu = np.triu_indices(n, 1)
    d = np.sqrt(sq_dists[iu])
    return -np.sum(np.log(d))


def _log_energy_and_grad(pts):
    """Compute logarithmic energy and its gradient on the sphere."""
    n = len(pts)
    # Pairwise squared distances
    dots = pts @ pts.T
    sq_dists = 2.0 - 2.0 * dots
    np.fill_diagonal(sq_dists, 1.0)
    sq_dists = np.maximum(sq_dists, 1e-30)
    
    d = np.sqrt(sq_dists)  # pairwise distances
    np.fill_diagonal(d, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(n, 1)
    energy = -np.sum(np.log(d[iu]))
    
    # Gradient: d/dx_i [-log(d_ij)] = (x_i - x_j) / d_ij^2  for each pair
    # Sum over j != i
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d2 = sq_dists[i, j]
            coeff = 1.0 / d2
            grad[i] += coeff * diff
            grad[j] -= coeff * diff
    
    # Project gradient onto tangent space: grad_t = grad - (grad . x) * x
    for i in range(n):
        grad[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
    
    return energy, grad


def _log_energy_and_grad_fast(pts):
    """Vectorized gradient computation."""
    n = len(pts)
    dots = pts @ pts.T
    sq_dists = 2.0 - 2.0 * dots
    np.fill_diagonal(sq_dists, 1.0)
    sq_dists = np.maximum(sq_dists, 1e-30)
    
    d = np.sqrt(sq_dists)
    np.fill_diagonal(d, 1.0)
    
    iu = np.triu_indices(n, 1)
    energy = -np.sum(np.log(d[iu]))
    
    # grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # = sum_j x_i/d_ij^2 - sum_j x_j/d_ij^2
    # Let S = 1/sq_dists with diagonal zero
    S = 1.0 / sq_dists
    np.fill_diagonal(S, 0.0)
    
    # grad_i = x_i * sum_j S_ij - sum_j S_ij * x_j
    row_sums = S.sum(axis=1)
    grad = pts * row_sums[:, np.newaxis] - S @ pts
    
    # Project to tangent
    for i in range(n):
        grad[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    return pts / norms


def _spectral_init(n, n_clusters=12):
    """
    Spectral clustering-based initialization:
    1. Start with Fibonacci points
    2. Build k-NN graph
    3. Compute graph Laplacian
    4. Use eigenvectors for better distribution
    """
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(n)
    
    # Build k-NN graph
    tree = cKDTree(pts)
    k = 10
    _, indices = tree.query(pts, k=k + 1)  # +1 because first is self
    
    # Build adjacency matrix
    adj = np.zeros((n, n))
    for i in range(n):
        for j in indices[i, 1:]:  # skip self
            adj[i, j] = 1.0
    
    # Symmetrize
    adj = (adj + adj.T) / 2.0
    
    # Graph Laplacian
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    
    # Get eigenvectors (skip trivial first one)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Use eigenvectors 1 through n_clusters as new coordinates
        # Then project back to sphere
        new_pts = eigenvectors[:, 1:n_clusters + 1]
        # Add some randomness to break symmetry
        new_pts += 0.01 * np.random.randn(n, n_clusters)
        # Project to sphere via PCA-like reduction
        # Use first 3 principal components
        from scipy.linalg import svd
        U, s, Vt = svd(new_pts, full_matrices=False)
        new_pts_3d = U[:, :3]
        new_pts_3d = _project_to_sphere(new_pts_3d)
        return new_pts_3d
    except:
        return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate multiple initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_spiral(N))
    
    # 2. Spectral init
    inits.append(_spectral_init(N))
    
    # 3. Random init with jitter
    for trial in range(3):
        pts = np.random.randn(N, 3)
        pts = _project_to_sphere(pts)
        inits.append(pts)
    
    best_pts = None
    best_energy = np.inf
    
    for init_idx, init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        # L-BFGS-B optimization on the sphere using tangent space
        # We parameterize using the fact that points are on the sphere
        # Use a simple approach: optimize in R^3 then project
        
        def objective_and_grad(x_flat):
            pts = x_flat.reshape(N, 3)
            pts = _project_to_sphere(pts)
            e, g = _log_energy_and_grad_fast(pts)
            # Gradient in ambient space, but we only move in tangent
            # For L-BFGS-B, we need the gradient in the parameter space
            # Since we project, the effective gradient is the tangent projection
            return e, g.flatten()
        
        x0 = init.flatten()
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 20,
                    'maxcor': 10
                }
            )
            
            pts = _project_to_sphere(result.x.reshape(N, 3))
            energy = _log_energy_fast(pts)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        except Exception:
            pass
        
        # Polish pass with tighter tolerances
        if best_pts is not None and time.time() < deadline - 10:
            x0 = best_pts.flatten()
            try:
                result2 = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 500,
                        'ftol': 1e-16,
                        'gtol': 1e-12,
                        'maxls': 30,
                        'maxcor': 15
                    }
                )
                
                pts2 = _project_to_sphere(result2.x.reshape(N, 3))
                energy2 = _log_energy_fast(pts2)
                
                if energy2 < best_energy:
                    best_energy = energy2
                    best_pts = pts2.copy()
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Final check: ensure all points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-9) or not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    return _project_to_sphere(best_pts)

# EVOLVE-BLOCK-END
