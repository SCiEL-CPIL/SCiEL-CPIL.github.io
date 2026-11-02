"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree

DEADLINE_SECONDS = 115.0  # Safety margin before 120s kill
EPS = 1e-12

def _energy_and_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|)
    and grad is the Euclidean gradient w.r.t. pts.
    
    Energy E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    dE/dp_i = -sum_{j!=i} d/dp_i log(|p_i - p_j|)
            = -sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2
            = sum_{j!=i} (p_j - p_i) / |p_i - p_j|^2
    """
    # Pairwise differences: (N, N, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    # Squared distances
    dist2 = np.sum(diff * diff, axis=2)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    # Safe distances
    dist = np.sqrt(np.where(mask, np.maximum(dist2, EPS), 1.0))
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # We sum over all i!=j and divide by 2
    log_terms = np.where(mask, -np.log(dist), 0.0)
    energy = 0.5 * np.sum(log_terms)
    
    # Gradient:
    # dE/dp_i = sum_{j!=i} (p_j - p_i) / |p_i - p_j|^2
    inv_dist2 = np.where(mask, 1.0 / np.maximum(dist2, EPS), 0.0)
    
    # grad[i] = sum_j (pts[j] - pts[i]) * inv_dist2[i, j]
    # = sum_j pts[j] * inv_dist2[i, j] - pts[i] * sum_j inv_dist2[i, j]
    weight_sums = np.sum(inv_dist2, axis=1, keepdims=True)
    weighted_pts = inv_dist2 @ pts
    
    grad = weighted_pts - weight_sums * pts
    
    return energy, grad

def _tangent_grad(pts, grad):
    """Project Euclidean gradient onto the tangent space."""
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - radial * pts

def _retract(pts, step):
    """Move points along step and project back to sphere."""
    new_pts = pts + step
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < EPS, 1.0, norms)
    return new_pts / norms

def _optimize_sphere(pts0, max_iters=2000, tol=1e-10, time_limit=None):
    """
    Riemannian Gradient Descent with Backtracking Line Search.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.01
    start_time = time.time()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Check convergence
        if np.linalg.norm(g_t) < tol:
            break
            
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(20):
            new_pts = _retract(pts, -alpha * g_t)  # Negative gradient = descent
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(g_t * g_t):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-12:
                break
            continue
        
        # Update gradient
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Slowly decay learning rate
        lr = max(0.001, lr * 0.999)
        
    return best_pts, best_energy

def _rcg_polish(pts0, max_iters=500, tol=1e-12, time_limit=None):
    """
    Riemannian Conjugate Gradient for faster convergence.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    # Initial search direction
    d_t = -g_t.copy()
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        if np.linalg.norm(g_t) < tol:
            break
        
        # Line search along d_t
        alpha = 0.01
        improved = False
        for _ in range(30):
            new_pts = _retract(pts, alpha * d_t)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(g_t * d_t):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            break
        
        # Update gradient
        _, grad = _energy_and_grad(pts)
        g_t_new = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Polak-Ribiere beta
        dot_new = np.sum(g_t_new * g_t)
        dot_old = np.sum(g_t * g_t)
        if dot_old > 1e-16:
            beta = max(0.0, dot_new / dot_old)
        else:
            beta = 0.0
        
        # Update search direction
        d_t = -g_t_new + beta * d_t
        # Ensure descent direction
        if np.sum(g_t_new * d_t) > 0:
            d_t = -g_t_new
        
        g_t = g_t_new
    
    return best_pts, best_energy

def _fibonacci_spiral(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _random_points(n, seed=42):
    """Generate n random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _perturbed_points(base_pts, sigma, seed=0):
    """Perturb base points by adding Gaussian noise and reprojecting."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*base_pts.shape) * sigma
    pts = base_pts + noise
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _polish_lbfgs(pts, max_iter=200, gtol=1e-14):
    """Polish using L-BFGS-B with tight tolerances."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = x_pts / np.linalg.norm(x_pts, axis=1, keepdims=True)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-16})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
            e_opt, _ = _energy_and_grad(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _energy_and_grad(pts)[0]

def _lattice_gas_refine(pts, n_grid=20, n_iters=50, time_limit=None):
    """
    Lattice gas occupancy refinement:
    1. Project points onto a 2D grid (equirectangular)
    2. Compute occupancy and local density
    3. Apply mean-field forces to adjust point positions
    4. Alternate with coordinate refinement
    
    This helps escape local minima by considering global density patterns.
    """
    pts = pts.copy()
    start_time = time.time()
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)
    
    for it in range(n_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Project to equirectangular grid
        theta = np.arccos(np.clip(pts[:, 2], -1, 1))  # polar angle [0, pi]
        phi = np.arctan2(pts[:, 1], pts[:, 0])  # azimuthal angle [-pi, pi]
        
        # Grid indices
        i_idx = np.clip(((theta / np.pi) * n_grid).astype(int), 0, n_grid - 1)
        j_idx = np.clip(((phi + np.pi) / (2 * np.pi) * n_grid).astype(int), 0, n_grid - 1)
        
        # Occupancy grid
        occ = np.zeros((n_grid, n_grid), dtype=float)
        for i, j in zip(i_idx, j_idx):
            occ[i, j] += 1.0
        
        # Compute density field (smoothed)
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(occ, sigma=2.0)
        
        # Compute density gradient (force direction)
        # Points should move away from high-density regions
        dy, dx = np.gradient(density)
        
        # For each point, compute the force from density gradient
        for k in range(N):
            i, j = i_idx[k], j_idx[k]
            # Interpolate density gradient at point location
            gy = dy[i, j]
            gx = dx[i, j]
            
            # Convert grid gradient to spherical coordinates
            # d/dtheta = -sin(theta) * d/dz + ...
            # Simplified: use the gradient direction in 3D space
            
            # Compute 3D gradient direction
            # The density increases in direction (dy, dx) in (theta, phi) space
            # Convert to 3D tangent vector
            sin_theta = np.sin(theta[k])
            if sin_theta < 1e-6:
                continue
            
            # Tangent vectors in spherical coordinates
            e_theta = np.array([-np.cos(theta[k]) * np.cos(phi[k]),
                                -np.cos(theta[k]) * np.sin(phi[k]),
                                np.sin(theta[k])])
            e_phi = np.array([-np.sin(phi[k]), np.cos(phi[k]), 0.0])
            
            # Force: move away from high density
            force_3d = -gy * e_theta - gx * e_phi * sin_theta
            force_3d = force_3d / (np.linalg.norm(force_3d) + 1e-10)
            
            # Small step in force direction
            step_size = 0.01 / (1.0 + 0.1 * it)
            pts[k] += step_size * force_3d
            pts[k] = pts[k] / np.linalg.norm(pts[k])
        
        # Refine with a few gradient steps
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        for _ in range(5):
            new_pts = _retract(pts, -0.005 * g_t)
            new_e, _ = _energy_and_grad(new_pts)
            if new_e < best_energy:
                pts = new_pts
                if new_e < best_energy:
                    best_energy = new_e
                    best_pts = pts.copy()
            _, grad = _energy_and_grad(pts)
            g_t = _tangent_grad(pts, grad)
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(('fib', _fibonacci_spiral(N)))
    
    # 2. Random with different seeds
    for seed in [1, 2, 3, 4, 5]:
        inits.append((f'rand{seed}', _random_points(N, seed=seed)))
    
    # 3. Perturbed Fibonacci with different scales
    fib_base = _fibonacci_spiral(N)
    for sigma, seed in [(0.1, 10), (0.05, 11), (0.2, 12)]:
        inits.append((f'pert{sigma}_{seed}', _perturbed_points(fib_base, sigma, seed=seed)))
    
    best_pts = None
    best_energy = np.inf
    
    # Optimize each initialization
    for name, init_pts in inits:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 20:
            break
        
        # Riemannian Gradient Descent
        pts_opt, energy_opt = _optimize_sphere(init_pts, max_iters=300)
        
        # RCG Polish
        if time.time() - start_time < DEADLINE_SECONDS - 15:
            pts_opt, energy_opt = _rcg_polish(pts_opt, max_iters=200, time_limit=30)
        
        # L-BFGS-B Polish
        if time.time() - start_time < DEADLINE_SECONDS - 10:
            pts_opt, energy_opt = _polish_lbfgs(pts_opt, max_iter=100, gtol=1e-13)
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Lattice gas refinement on best result
    if best_pts is not None and time.time() - start_time < DEADLINE_SECONDS - 15:
        lg_pts, lg_energy = _lattice_gas_refine(best_pts, n_grid=15, n_iters=30, time_limit=20)
        if lg_energy < best_energy:
            best_pts = lg_pts
            best_energy = lg_energy
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() - start_time < DEADLINE_SECONDS - 8:
        best_pts, best_energy = _rcg_polish(best_pts, max_iters=300, time_limit=25)
    
    if best_pts is not None and time.time() - start_time < DEADLINE_SECONDS - 5:
        best_pts, best_energy = _polish_lbfgs(best_pts, max_iter=200, gtol=1e-15)
    
    # Final Check and Return
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
# EVOLVE-BLOCK-END
