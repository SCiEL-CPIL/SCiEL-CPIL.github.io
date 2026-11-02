import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill
EPS = 1e-12

def _energy_and_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|)
    and grad is the gradient w.r.t. pts in R^{N x 3}.
    """
    # Squared chordal distances: |p_i - p_j|^2 = 2 - 2 p_i . p_j
    dot = pts @ pts.T
    dist2 = 2.0 - 2.0 * dot
    np.fill_diagonal(dist2, 1.0)
    dist2 = np.maximum(dist2, EPS)
    
    # Energy: sum_{i<j} log(1/dist) = -0.5 * sum_{i!=j} log(dist)
    dist = np.sqrt(dist2)
    log_dist = np.log(dist)
    np.fill_diagonal(log_dist, 0.0)
    energy = -0.5 * np.sum(log_dist)
    
    # Gradient: d/dp_i sum_{j!=i} log(1/|p_i-p_j|) = -sum_{j!=i} (p_i-p_j)/|p_i-p_j|^2
    # = sum_{j!=i} (p_j-p_i)/|p_i-p_j|^2
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
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
    norms = np.where(norms < EPS, 1.0, norms)
    return new_pts / norms

def _optimize_sphere(pts0, max_iters=3000, tol=1e-12, time_limit=None, t_start=None):
    """
    Riemannian Gradient Descent with Backtracking Line Search and momentum.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.01
    momentum = np.zeros_like(pts)
    beta = 0.9
    
    for it in range(max_iters):
        if time_limit is not None:
            if time.time() - t_start > time_limit:
                break
                
        # Check convergence
        if np.linalg.norm(g_t) < tol:
            break
            
        # Momentum update
        momentum = beta * momentum + (1 - beta) * g_t
        
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(25):
            new_pts = _retract(pts, -alpha * momentum)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(momentum * momentum):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-14:
                break
            continue
        
        # Update gradient
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Slowly decay learning rate
        lr = max(0.001, lr * 0.998)
        
    return best_pts, best_energy

def _rcg_polish(pts0, max_iters=200, tol=1e-12, time_limit=None, t_start=None):
    """
    Riemannian Conjugate Gradient polish for faster convergence near optimum.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    # RCG state
    d = -g_t.copy()  # initial direction = negative gradient
    beta_r = 0.0
    
    for it in range(max_iters):
        if time_limit is not None:
            if time.time() - t_start > time_limit:
                break
        
        gnorm = np.linalg.norm(g_t)
        if gnorm < tol:
            break
        
        # Line search for step size
        alpha = 1.0
        for _ in range(20):
            new_pts = _retract(pts, alpha * d)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy:
                pts = new_pts
                energy = new_energy
                break
            alpha *= 0.5
        else:
            # Couldn't improve, try smaller step
            alpha = 1e-4
            new_pts = _retract(pts, alpha * d)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy:
                pts = new_pts
                energy = new_energy
            else:
                break
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Compute new gradient
        _, grad = _energy_and_grad(pts)
        g_new_t = _tangent_grad(pts, grad)
        
        # Polak-Ribiere beta
        if beta_r > 0:
            g_old_t = g_t
            num = np.sum(g_new_t * (g_new_t - g_old_t))
            den = np.sum(g_old_t * g_old_t)
            if den > 1e-16:
                beta_r = max(0.0, num / den)
            else:
                beta_r = 0.0
        else:
            beta_r = 0.0
        
        # New direction
        d = -g_new_t + beta_r * d
        
        # Project direction to tangent space
        radial = np.sum(d * pts, axis=1, keepdims=True)
        d = d - radial * pts
        
        g_t = g_new_t
    
    return best_pts, best_energy

def _fibonacci_spiral(n, offset=0.5):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + offset
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

def _polish_lbfgs(pts, max_iter=500, gtol=1e-14, time_limit=None, t_start=None):
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
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-16,
                                'maxls': 50, 'm': 20})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
            e_opt, _ = _energy_and_grad(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _energy_and_grad(pts)[0]

def _newton_step(pts, g_t):
    """
    Approximate Newton step using the Hessian-vector product structure.
    For logarithmic energy, the Hessian has a specific form that can be exploited.
    Here we use a simplified approach: solve (I + lambda * H) dx = -g
    where H is approximated by the gradient structure.
    """
    # Simple approach: use the fact that for small steps, the Hessian
    # can be approximated by the gradient's dependence on position
    # For now, use a damped Newton step with adaptive damping
    gnorm = np.linalg.norm(g_t)
    if gnorm < 1e-15:
        return g_t
    
    # Adaptive step size based on gradient norm
    alpha = min(0.1, 1.0 / (gnorm + 1e-10))
    return -alpha * g_t

def construct():
    t_start = time.time()
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral with different offsets
    for offset in [0.3, 0.5, 0.7]:
        inits.append((f'fib_{offset}', _fibonacci_spiral(N, offset)))
    
    # 2. Random with different seeds
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        inits.append((f'rand{seed}', _random_points(N, seed=seed)))
    
    # 3. Perturbed Fibonacci with different scales
    fib_base = _fibonacci_spiral(N)
    for sigma, seed in [(0.05, 10), (0.1, 11), (0.15, 12), (0.2, 13), (0.3, 14)]:
        inits.append((f'pert{sigma}_{seed}', _perturbed_points(fib_base, sigma, seed=seed)))
    
    # 4. Perturbed random
    rand_base = _random_points(N, seed=100)
    for sigma, seed in [(0.1, 20), (0.2, 21)]:
        inits.append((f'randpert{sigma}_{seed}', _perturbed_points(rand_base, sigma, seed=seed)))
    
    best_pts = None
    best_energy = np.inf
    
    # Optimize each initialization
    for name, init_pts in inits:
        elapsed = time.time() - t_start
        if elapsed > DEADLINE_SECONDS - 25:
            break
        
        # Riemannian Gradient Descent with momentum
        pts_opt, energy_opt = _optimize_sphere(init_pts, max_iters=800, tol=1e-10,
                                               time_limit=DEADLINE_SECONDS - 20, t_start=t_start)
        
        # RCG Polish for faster convergence
        if time.time() - t_start < DEADLINE_SECONDS - 15:
            pts_opt, energy_opt = _rcg_polish(pts_opt, max_iters=100, tol=1e-12,
                                              time_limit=DEADLINE_SECONDS - 12, t_start=t_start)
        
        # L-BFGS-B Polish
        if time.time() - t_start < DEADLINE_SECONDS - 10:
            pts_opt, energy_opt = _polish_lbfgs(pts_opt, max_iter=200, gtol=1e-13,
                                                time_limit=DEADLINE_SECONDS - 8, t_start=t_start)
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() - t_start < DEADLINE_SECONDS - 8:
        # RCG polish first (faster)
        best_pts, best_energy = _rcg_polish(best_pts, max_iters=200, tol=1e-13,
                                            time_limit=DEADLINE_SECONDS - 5, t_start=t_start)
        
        # L-BFGS-B for final refinement
        if time.time() - t_start < DEADLINE_SECONDS - 5:
            best_pts, best_energy = _polish_lbfgs(best_pts, max_iter=500, gtol=1e-14,
                                                  time_limit=DEADLINE_SECONDS - 3, t_start=t_start)
        
        # Additional fine-tuning with small steps
        if time.time() - t_start < DEADLINE_SECONDS - 2:
            for _ in range(200):
                if time.time() - t_start > DEADLINE_SECONDS - 1:
                    break
                e, g = _energy_and_grad(best_pts)
                g_t = _tangent_grad(best_pts, g)
                gnorm = np.linalg.norm(g_t)
                if gnorm < 1e-14:
                    break
                # Adaptive step: use gradient norm to set step size
                alpha = min(0.01, 0.1 / (gnorm + 1e-10))
                new_pts = _retract(best_pts, -alpha * g_t)
                new_e, _ = _energy_and_grad(new_pts)
                if new_e < e:
                    best_pts = new_pts
                    best_energy = new_e
                else:
                    alpha *= 0.5
                    new_pts = _retract(best_pts, -alpha * g_t)
                    new_e, _ = _energy_and_grad(new_pts)
                    if new_e < e:
                        best_pts = new_pts
                        best_energy = new_e
    
    # Final Check and Return
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
