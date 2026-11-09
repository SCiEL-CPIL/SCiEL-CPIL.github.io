"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _cover_radius_grid(pts, n_grid=8000):
    """Compute covering radius using a Fibonacci grid."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    dots = grid @ pts.T
    min_dots = np.max(dots, axis=1)
    return np.arccos(np.clip(min_dots, -1.0, 1.0)).max()

def _lse_objective_and_grad(x, grid, beta):
    """
    Log-sum-exp surrogate for the covering radius.
    """
    p = x.reshape(-1, 3)
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    p = p / norms
    
    dots = grid @ p.T
    max_dots = np.max(dots, axis=1)
    
    max_dots_clipped = np.clip(max_dots, -1.0 + 1e-15, 1.0 - 1e-15)
    angular_dists = np.arccos(max_dots_clipped)
    
    max_ad = np.max(angular_dists)
    lse = max_ad + np.log(np.sum(np.exp(beta * (angular_dists - max_ad))))
    
    weights = beta * np.exp(beta * (angular_dists - max_ad))
    weights /= np.sum(weights)
    
    denom = np.sqrt(np.clip(1.0 - max_dots * max_dots, 1e-15, None))
    dad = -weights / denom
    
    argmax_j = np.argmax(dots, axis=1)
    
    grad = np.zeros_like(x)
    for i in range(len(grid)):
        j = argmax_j[i]
        grad[j*3:(j+1)*3] += dad[i] * grid[i]
    
    return lse, grad

def _pairwise_repulsion_obj_grad(x, sigma, n_grid):
    """
    Pairwise repulsive potential: minimize sum of exp(-d_ij/sigma) for all pairs.
    This pushes points apart uniformly.
    """
    p = x.reshape(-1, 3)
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    p = p / norms
    
    # Pairwise dot products
    dots = p @ p.T
    np.fill_diagonal(dots, 1.0)
    
    # Angular distances
    dots_clipped = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    ang_dists = np.arccos(dots_clipped)
    
    # Objective: sum of exp(-d_ij / sigma) for i != j
    off_diag_mask = ~np.eye(N, dtype=bool)
    potential = np.sum(np.exp(-ang_dists[off_diag_mask] / sigma))
    
    # Gradient: d/dp_k sum_{i,j} exp(-d_ij/sigma)
    # = sum_{i,j} (-1/sigma) * exp(-d_ij/sigma) * d(d_ij)/d(p_k)
    # d(d_ij)/d(p_k) = -1/sqrt(1-dot_ij^2) * (delta_ik * p_j - delta_jk * p_i)
    
    grad = np.zeros_like(x)
    exp_vals = np.exp(-ang_dists / sigma)
    np.fill_diagonal(exp_vals, 0.0)
    
    denom = np.sqrt(np.clip(1.0 - dots * dots, 1e-15, None))
    np.fill_diagonal(denom, 1.0)
    
    # For each pair (i,j), contribution to grad:
    # (-1/sigma) * exp(-d_ij/sigma) * (-1/denom_ij) * (delta_ik*p_j - delta_jk*p_i)
    # For point k: sum over i != k of [(-1/sigma)*exp(-d_ik/sigma)*(-1/denom_ik)*p_i]
    #             + sum over j != k of [(-1/sigma)*exp(-d_kj/sigma)*(-1/denom_kj)*(-p_j)]
    # = (1/sigma) * sum_{i != k} exp(-d_ik/sigma)/denom_ik * p_i
    #   - (1/sigma) * sum_{j != k} exp(-d_kj/sigma)/denom_kj * p_j
    
    # Since the matrix is symmetric, both sums are the same:
    # grad_k = (2/sigma) * sum_{i != k} exp(-d_ik/sigma)/denom_ik * p_i ... wait, let me redo
    
    # d(d_ij)/dp_k where d_ij = arccos(dot(p_i, p_j)):
    # = -1/sqrt(1-dot_ij^2) * d(dot(p_i,p_j))/dp_k
    # = -1/denom_ij * (delta_ik * p_j - delta_jk * p_i)
    
    # So d(obj)/dp_k = sum_{i,j} (-1/sigma)*exp(-d_ij/sigma) * (-1/denom_ij) * (delta_ik*p_j - delta_jk*p_i)
    # = (1/sigma) * sum_{i,j} exp(-d_ij/sigma)/denom_ij * (delta_ik*p_j - delta_jk*p_i)
    # For fixed k: (1/sigma) * [sum_j exp(-d_kj/sigma)/denom_kj * p_j - sum_i exp(-d_ik/sigma)/denom_ik * p_i]
    # Since symmetric: (1/sigma) * [sum_j exp(-d_kj/sigma)/denom_kj * p_j - sum_j exp(-d_jk/sigma)/denom_jk * p_j] = 0?
    # No wait, the first term has j as the "other" point when k is first index, second term has i as "other" when k is second.
    # Let me be more careful.
    
    # obj = sum_{i<j} exp(-d_ij/sigma) (counting each pair once)
    # d(obj)/dp_k = sum_{i<j} (-1/sigma)*exp(-d_ij/sigma) * d(d_ij)/dp_k
    # d(d_ij)/dp_k = -1/denom_ij * (delta_ik*p_j - delta_jk*p_i)
    # So d(obj)/dp_k = (1/sigma) * sum_{i<j} exp(-d_ij/sigma)/denom_ij * (delta_ik*p_j - delta_jk*p_i)
    
    # For a given k:
    # Term 1: pairs where i=k, j>k: sum_{j>k} exp(-d_kj/sigma)/denom_kj * p_j
    # Term 2: pairs where j=k, i<k: sum_{i<k} exp(-d_ik/sigma)/denom_ik * (-p_i)
    # Total: (1/sigma) * [sum_{j>k} exp(-d_kj/sigma)/denom_kj * p_j - sum_{i<k} exp(-d_ik/sigma)/denom_ik * p_i]
    
    # This is asymmetric! Let me just compute it directly.
    
    # Actually, let me use the full sum formulation (each pair counted twice) for symmetry:
    # obj_full = sum_{i != j} exp(-d_ij/sigma) / 2
    # d(obj_full)/dp_k = (1/2) * (1/sigma) * sum_{i != j} exp(-d_ij/sigma)/denom_ij * (delta_ik*p_j - delta_jk*p_i)
    # = (1/(2*sigma)) * [sum_j exp(-d_kj/sigma)/denom_kj * p_j - sum_i exp(-d_ik/sigma)/denom_ik * p_i]
    # = (1/(2*sigma)) * [sum_j exp(-d_kj/sigma)/denom_kj * p_j - sum_j exp(-d_jk/sigma)/denom_jk * p_j]
    # = 0 by symmetry?!
    
    # That can't be right. Let me think again...
    # Actually d(d_ij)/dp_k when k != i,j is 0. So only terms where k=i or k=j contribute.
    # When k=i: d(d_ik)/dp_k = -1/denom_ik * (p_k*p_k... no)
    # dot(p_i, p_j) = p_i · p_j
    # d(p_i · p_j)/dp_i = p_j
    # d(p_i · p_j)/dp_j = p_i
    # So d(d_ij)/dp_k = -1/denom_ij * (delta_ik * p_j - delta_jk * p_i)
    # This is correct.
    
    # For the full double sum: sum_{i!=j} f(d_ij) * d(d_ij)/dp_k
    # = sum_{i!=j} f(d_ij) * (-1/denom_ij) * (delta_ik*p_j - delta_jk*p_i)
    # = -sum_{j!=k} f(d_kj)/denom_kj * p_j + sum_{i!=k} f(d_ik)/denom_ik * p_i
    # = -sum_{j!=k} f(d_kj)/denom_kj * p_j + sum_{j!=k} f(d_kj)/denom_kj * p_j (renaming i->j in second)
    # = 0
    
    # So the gradient of the symmetric pairwise sum is zero?! That means I need to use a different formulation.
    # 
    # The issue is that the potential is rotationally symmetric around each point.
    # I need to add a directional component or use a different approach.
    
    # Alternative: use the covering radius directly via a different surrogate.
    # Let's use: sum of exp(beta * (d_i - d_max)) where d_i is the angular distance
    # from grid point i to nearest config point, but only for the "worst" grid points.
    
    # Actually, let me just return the LSE objective and fall back to that.
    # For now, let me implement a proper local search using Voronoi centroids.
    
    return potential, np.zeros_like(x)  # Placeholder, we'll use LSE instead

def _voronoi_centroid_move(pts, n_grid=5000, step_size=0.05):
    """
    Move each point toward the centroid of its Voronoi cell (points on sphere
    that are closer to it than to any other point).
    """
    p = pts.copy()
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = grid @ p.T  # (n_grid, N)
    nearest = np.argmax(dots, axis=1)  # (n_grid,)
    
    # For each point, compute centroid of grid points assigned to it
    new_pts = p.copy()
    for k in range(N):
        mask = nearest == k
        if np.sum(mask) > 0:
            centroid = np.mean(grid[mask], axis=0)
            norm_c = np.linalg.norm(centroid)
            if norm_c > 1e-10:
                centroid /= norm_c
                new_pts[k] = p[k] + step_size * (centroid - p[k])
                new_pts[k] /= np.linalg.norm(new_pts[k])
    
    return new_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_fibonacci_sphere(N))
    
    rng = np.random.RandomState(42)
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    best_pts = None
    best_radius = np.inf
    
    # Precompute grids
    def make_grid(n_grid):
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, None))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    grid_coarse = make_grid(3000)
    grid_fine = make_grid(6000)
    grid_vf = make_grid(10000)
    
    for init_idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 25.0:
            break
        
        current_pts = init_pts.copy()
        current_radius = _cover_radius_grid(current_pts, n_grid=5000)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
        
        # Phase 1: L-BFGS with LSE surrogate (coarse)
        if time.time() > deadline - 20.0:
            break
        
        beta_coarse = 20.0
        def obj_grad_coarse(x):
            return _lse_objective_and_grad(x, grid_coarse, beta_coarse)
        
        try:
            result = minimize(
                obj_grad_coarse,
                current_pts.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 400}
            )
            new_pts = result.x.reshape(-1, 3)
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_radius = _cover_radius_grid(new_pts, n_grid=5000)
            if new_radius < current_radius:
                current_pts = new_pts
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = current_pts.copy()
        except Exception:
            pass
        
        # Phase 2: Voronoi centroid moves
        if time.time() > deadline - 15.0:
            break
        
        for _ in range(3):
            if time.time() > deadline - 12.0:
                break
            new_pts = _voronoi_centroid_move(current_pts, n_grid=5000, step_size=0.03)
            new_radius = _cover_radius_grid(new_pts, n_grid=5000)
            if new_radius < current_radius:
                current_pts = new_pts
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = current_pts.copy()
        
        # Phase 3: L-BFGS with LSE surrogate (fine)
        if time.time() > deadline - 10.0:
            break
        
        for beta in [50.0, 100.0]:
            if time.time() > deadline - 8.0:
                break
            
            def obj_grad_fine(x, grid=grid_fine, beta=beta):
                return _lse_objective_and_grad(x, grid, beta)
            
            try:
                result = minimize(
                    obj_grad_fine,
                    current_pts.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
                )
                new_pts = result.x.reshape(-1, 3)
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_radius = _cover_radius_grid(new_pts, n_grid=8000)
                if new_radius < current_radius:
                    current_pts = new_pts
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = current_pts.copy()
            except Exception:
                pass
        
        # Phase 4: More Voronoi centroid moves
        if time.time() > deadline - 5.0:
            break
        
        for _ in range(5):
            if time.time() > deadline - 4.0:
                break
            new_pts = _voronoi_centroid_move(current_pts, n_grid=8000, step_size=0.02)
            new_radius = _cover_radius_grid(new_pts, n_grid=8000)
            if new_radius < current_radius:
                current_pts = new_pts
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = current_pts.copy()
    
    # Final polish
    if best_pts is not None and time.time() < deadline - 5.0:
        for beta in [200.0, 400.0]:
            if time.time() > deadline - 3.0:
                break
            
            def obj_grad_vf(x, grid=grid_vf, beta=beta):
                return _lse_objective_and_grad(x, grid, beta)
            
            try:
                result = minimize(
                    obj_grad_vf,
                    best_pts.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 40, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 150}
                )
                new_pts = result.x.reshape(-1, 3)
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_radius = _cover_radius_grid(new_pts, n_grid=12000)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = new_pts.copy()
            except Exception:
                pass
        
        # Final Voronoi moves
        if time.time() < deadline - 2.0:
            for _ in range(3):
                if time.time() > deadline - 1.0:
                    break
                new_pts = _voronoi_centroid_move(best_pts, n_grid=10000, step_size=0.01)
                new_radius = _cover_radius_grid(new_pts, n_grid=12000)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = new_pts.copy()
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
