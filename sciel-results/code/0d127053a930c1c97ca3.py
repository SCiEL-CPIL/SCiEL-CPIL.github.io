import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 145.0
EPS = 1e-14

def _energy_and_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|)
    and grad is the gradient w.r.t. pts in R^{N x 3}.
    """
    # Pairwise squared distances: dist2[i,j] = |pts[i] - pts[j]|^2
    # = 2 - 2 * pts[i] . pts[j]  (since all on unit sphere)
    dot = pts @ pts.T  # (N, N)
    dist2 = 2.0 - 2.0 * dot
    np.fill_diagonal(dist2, 0.0)
    dist2 = np.maximum(dist2, EPS)
    
    # Energy: sum_{i<j} log(1/dist) = -0.5 * sum_{i!=j} log(dist)
    # = -0.5 * sum_{i!=j} 0.5 * log(dist2)
    # = -0.25 * sum_{i!=j} log(dist2)
    log_dist2 = np.log(dist2)
    np.fill_diagonal(log_dist2, 0.0)
    energy = -0.25 * np.sum(log_dist2)
    
    # Gradient:
    # d/dp_i log(1/|p_i - p_j|) = -(p_i - p_j) / |p_i - p_j|^2
    # grad_i = sum_{j!=i} (p_j - p_i) / |p_i - p_j|^2
    # = sum_j p_j / dist2[i,j] - p_i * sum_j 1/dist2[i,j]
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    weight_sums = np.sum(inv_dist2, axis=1, keepdims=True)  # (N, 1)
    weighted_pts = inv_dist2 @ pts  # (N, 3)
    
    grad = weighted_pts - weight_sums * pts
    
    return energy, grad

def _tangent_grad(pts, grad):
    """Project Euclidean gradient onto the tangent space of the sphere."""
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - radial * pts

def _retract(pts, step):
    """Move points along step and project back to sphere."""
    new_pts = pts + step
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms = np.where(norms < EPS, 1.0, norms)
    return new_pts / norms

def _riemannian_descent(pts0, max_iters=3000, tol=1e-12, time_limit=None, t_start=None):
    """
    Riemannian Gradient Descent with adaptive step size and momentum.
    """
    if t_start is None:
        t_start = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.05
    momentum = 0.0
    prev_g_t = np.zeros_like(g_t)
    
    for it in range(max_iters):
        # Time check
        if time_limit is not None and (time.time() - t_start) > time_limit:
            break
        
        # Convergence check
        g_norm = np.linalg.norm(g_t)
        if g_norm < tol:
            break
        
        # Momentum update
        momentum = 0.9 * momentum + 0.1 * (1.0 / (1.0 + g_norm))
        step_dir = -g_t * momentum
        
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(30):
            new_pts = _retract(pts, alpha * step_dir)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(step_dir * step_dir):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-14:
                break
            # Reset momentum on failure
            momentum = 0.0
            continue
        
        # Update gradient
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Adaptive learning rate
        lr = max(0.01, min(0.1, lr * 1.01))
        
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

def _icosahedron_subdivide(n_target):
    """Generate points via icosahedron subdivision."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Subdivide until we have enough points
    while len(verts) < n_target:
        new_verts = []
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 0.9:  # Adjacent vertices
                    mid = (verts[i] + verts[j]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)]) if new_verts else verts
        # Remove duplicates approximately
        unique = [verts[0]]
        for v in verts[1:]:
            if all(np.linalg.norm(v - u) > 1e-6 for u in unique):
                unique.append(v)
        verts = np.array(unique)
    
    return verts[:n_target] / np.linalg.norm(verts[:n_target], axis=1, keepdims=True)

def _perturbed_points(base_pts, sigma, seed=0):
    """Perturb base points by adding Gaussian noise and reprojecting."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*base_pts.shape) * sigma
    pts = base_pts + noise
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _polish_lbfgs(pts, max_iter=500, gtol=1e-15, time_limit=None, t_start=None):
    """Polish using L-BFGS-B with tight tolerances."""
    if t_start is None:
        t_start = time.time()
    
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        norms = np.linalg.norm(x_pts, axis=1, keepdims=True)
        norms = np.where(norms < EPS, 1.0, norms)
        x_pts = x_pts / norms
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-18,
                                'maxls': 50, 'iprint': 0})
        if np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
            e_opt, _ = _energy_and_grad(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _energy_and_grad(pts)[0]

def _simulated_annealing(pts0, n_iter=500, t_start=None):
    """Simulated annealing to escape local minima."""
    if t_start is None:
        t_start = time.time()
    
    pts = pts0.copy()
    energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    T = 0.1
    T_min = 0.001
    cooling = 0.995
    
    for it in range(n_iter):
        if (time.time() - t_start) > DEADLINE_SECONDS - 20:
            break
        
        # Random perturbation
        rng = np.random.RandomState(it)
        idx = rng.randint(N)
        noise = rng.randn(3) * 0.05
        new_pts = pts.copy()
        new_pts[idx] = new_pts[idx] + noise
        new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_energy, _ = _energy_and_grad(new_pts)
        delta = new_energy - energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        
        T = max(T_min, T * cooling)
    
    return best_pts, best_energy

def construct():
    t_start = time.time()
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_spiral(N))
    
    # 2. Random with different seeds
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        inits.append(_random_points(N, seed=seed))
    
    # 3. Icosahedron-based
    try:
        inits.append(_icosahedron_subdivide(N))
    except:
        pass
    
    # 4. Perturbed Fibonacci
    fib_base = _fibonacci_spiral(N)
    for sigma, seed in [(0.1, 10), (0.05, 11), (0.2, 12), (0.15, 13)]:
        inits.append(_perturbed_points(fib_base, sigma, seed=seed))
    
    best_pts = None
    best_energy = np.inf
    
    # Optimize each initialization with time budget
    for idx, init_pts in enumerate(inits):
        elapsed = time.time() - t_start
        remaining = DEADLINE_SECONDS - elapsed
        if remaining < 5:
            break
        
        # Allocate time based on how many inits remain
        inits_left = len(inits) - idx
        time_per_init = min(remaining * 0.8 / inits_left, 15.0)
        
        # Riemannian Gradient Descent
        pts_opt, energy_opt = _riemannian_descent(
            init_pts, max_iters=2000, tol=1e-12,
            time_limit=time_per_init * 0.6, t_start=t_start
        )
        
        # L-BFGS-B Polish if time permits
        if time.time() - t_start < DEADLINE_SECONDS - 10:
            pts_opt, energy_opt = _polish_lbfgs(
                pts_opt, max_iter=300, gtol=1e-14,
                time_limit=DEADLINE_SECONDS - time.time() - 8, t_start=t_start
            )
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Simulated annealing on best result if time permits
    if best_pts is not None and time.time() - t_start < DEADLINE_SECONDS - 15:
        sa_pts, sa_energy = _simulated_annealing(best_pts, n_iter=300, t_start=t_start)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_pts = sa_pts.copy()
    
    # Final high-precision polish
    if best_pts is not None and time.time() - t_start < DEADLINE_SECONDS - 5:
        best_pts, best_energy = _polish_lbfgs(
            best_pts, max_iter=500, gtol=1e-15,
            time_limit=DEADLINE_SECONDS - time.time() - 3, t_start=t_start
        )
    
    # Final Check and Return
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
