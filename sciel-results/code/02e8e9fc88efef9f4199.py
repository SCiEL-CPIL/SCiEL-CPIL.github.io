"""Optimized logarithmic energy minimization of 150 points on the sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s limit


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1 / |x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    pts: (N, 3) array of points on the unit sphere
    Returns: (energy, gradient) where gradient is the unconstrained gradient
             (caller must project to tangent space)
    """
    n = len(pts)
    # Compute pairwise squared distances efficiently
    dot = pts @ pts.T  # (N, N)
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero / log(0)
    
    # Energy: sum over i<j of -log(dist_ij) = -0.5*sum_{i<j} log(dist_sq)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dist_sq[iu]))
    
    # Gradient: dE/dx_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    inv_dist_sq = 1.0 / dist_sq  # (N, N)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    row_sum = inv_dist_sq.sum(axis=1)  # (N,)
    W_pts = inv_dist_sq @ pts  # (N, 3)
    grad = W_pts - row_sum[:, np.newaxis] * pts  # (N, 3)
    
    return energy, grad


def _energy_only(pts):
    """Energy only, for fast inner loops."""
    n = len(pts)
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    return -0.5 * np.sum(np.log(dist_sq[iu]))


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _tangent_project(pts, grad):
    """Project gradient onto tangent space at each point."""
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - dot * pts


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci sphere points, optionally with a random offset."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _perturbed_sphere(n, base_pts, sigma, seed):
    """Generate perturbed sphere points."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(n, 3) * sigma
    pts = base_pts + noise
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _icosahedron_based(n):
    """Generate points using icosahedron vertices for a better initial distribution."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    fib = _fibonacci_sphere(n)
    result = fib.copy()
    for i in range(12):
        dists = np.linalg.norm(fib - ico[i], axis=1)
        closest = np.argmin(dists)
        result[closest] = ico[i]
    return _project_to_sphere(result)


def _optimize_l_bfgs(start_pts, max_time=30.0):
    """Run L-BFGS-B optimization from a starting configuration."""
    x0 = start_pts.flatten()
    t0 = time.time()
    
    def objective(x):
        if time.time() - t0 > max_time:
            return np.inf
        pts = _project_to_sphere(x.reshape(N, 3))
        e, _ = _energy_and_grad(pts)
        return e
    
    def jac(x):
        if time.time() - t0 > max_time:
            return np.zeros(len(x))
        pts = _project_to_sphere(x.reshape(N, 3))
        _, g = _energy_and_grad(pts)
        g_t = _tangent_project(pts, g)
        return g_t.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=jac,
            method='L-BFGS-B',
            options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
        )
        best_pts = _project_to_sphere(result.x.reshape(N, 3))
        best_energy, _ = _energy_and_grad(best_pts)
        return best_pts, best_energy
    except Exception:
        return start_pts, float('inf')


def _adam_optimize(pts, time_limit=10.0):
    """Adam optimizer on the sphere with adaptive learning rate and patience."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    
    lr_initial = 0.02
    lr_min = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr_decay = 0.995
    
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    lr = lr_initial
    
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)
    
    no_improve_count = 0
    patience = 500
    
    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        m_tangent = m_hat - np.sum(m_hat * pts, axis=1, keepdims=True) * pts
        
        pts = pts - lr * m_tangent / (np.sqrt(v_hat) + eps)
        pts = _project_to_sphere(pts)
        
        lr = max(lr_min, lr * lr_decay)
        
        if t % 100 == 0 and np.linalg.norm(grad_tangent) < 1e-10:
            break
    
    return best_pts, best_energy


def _simulated_annealing(pts, time_limit=15.0, seed=42):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    
    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    max_steps = 5000
    
    rng = np.random.RandomState(seed)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        noise_scale = T * 0.1
        perturbation = rng.randn(N, 3) * noise_scale
        new_pts = pts + perturbation
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _energy_only(new_pts)
        
        if new_energy < current_energy or rng.rand() < math.exp(-(new_energy - current_energy) / T):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        T *= cooling
        if T < T_min:
            break
    
    return best_pts, best_energy


def _polish(pts, time_limit=5.0):
    """Final high-precision polish with very tight tolerances."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    
    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e
    
    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    x0 = pts.flatten()
    
    result = minimize(
        objective_flat, x0, jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 5000,
            'maxfun': 10000,
            'ftol': 1e-15,
            'gtol': 1e-12,
            'maxls': 100
        }
    )
    
    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _gradient_descent_polish(pts0, max_time):
    """Polish with gradient descent using backtracking line search and momentum."""
    start_time = time.time()
    pts = _project_to_sphere(pts0.copy())
    e, g = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_e = e
    
    lr = 0.01
    momentum = np.zeros_like(pts)
    beta = 0.9
    
    for _ in range(50000):
        if time.time() - start_time >= max_time:
            break
        
        g_tan = _tangent_project(pts, g)
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-13:
            break
        
        momentum = beta * momentum - lr * g_tan
        
        step = 1.0
        new_pts = pts
        new_e = e
        found = False
        for _ in range(25):
            candidate = pts + step * momentum
            candidate = _project_to_sphere(candidate)
            cand_e, _ = _energy_and_grad(candidate)
            if cand_e < e:
                new_pts = candidate
                new_e = cand_e
                found = True
                break
            step *= 0.5
        if not found:
            step = lr * 0.1
            candidate = pts - step * g_tan
            candidate = _project_to_sphere(candidate)
            cand_e, _ = _energy_and_grad(candidate)
            if cand_e < e:
                new_pts = candidate
                new_e = cand_e
                found = True
                momentum = -step * g_tan
            else:
                break
        
        if not found:
            break
        
        pts = new_pts
        e = new_e
        g = _energy_and_grad(pts)[1]
        
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
            if best_e < e + 1e-8:
                lr = min(lr * 1.02, 0.05)
    
    return best_pts, best_e


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    
    # Best result so far
    best_pts = _fibonacci_sphere(N)
    best_energy, _ = _energy_and_grad(best_pts)
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Fibonacci sphere (baseline)
    starts.append(_fibonacci_sphere(N))
    
    # 2. Icosahedron-based (better initial distribution)
    starts.append(_icosahedron_based(N))
    
    # 3. A few good random starts
    for seed in [42, 123, 789, 2022, 5055]:
        starts.append(_random_sphere(N, seed))
    
    # 4. Perturbed Fibonacci starts (moderate perturbations)
    for sigma in [0.1, 0.2, 0.3]:
        for seed in [1, 2]:
            starts.append(_perturbed_sphere(N, _fibonacci_sphere(N), sigma, seed))
    
    # 5. Slightly anisotropic scaling
    fib = _fibonacci_sphere(N)
    for scale in [0.85, 1.15]:
        scaled = fib * np.array([scale, 1.0, 1.0])
        starts.append(_project_to_sphere(scaled))
    
    # Optimize from each start
    for i, start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 30:
            break
        
        try:
            # First pass: L-BFGS
            lbfgs_time = min(15.0, TIME_LIMIT - 35 - elapsed)
            if lbfgs_time < 2:
                break
            pts, energy = _optimize_l_bfgs(start, max_time=lbfgs_time)
            
            # Second pass: Adam with adaptive LR
            remaining = TIME_LIMIT - 30 - (time.time() - start_time)
            if remaining > 3:
                pts, energy = _adam_optimize(pts, time_limit=min(8.0, remaining))
            
            # Third pass: gradient descent polish
            remaining = TIME_LIMIT - 25 - (time.time() - start_time)
            if remaining > 2:
                pts, energy = _gradient_descent_polish(pts, remaining)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Simulated annealing on the best result to escape local minima
    remaining = TIME_LIMIT - 20 - (time.time() - start_time)
    if remaining > 5:
        try:
            pts, energy = _simulated_annealing(best_pts, time_limit=min(12.0, remaining), seed=42)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        except Exception:
            pass
    
    # Final high-precision polish on the best result
    remaining = TIME_LIMIT - 10 - (time.time() - start_time)
    if remaining > 3:
        try:
            pts, energy = _polish(best_pts, time_limit=min(8.0, remaining))
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        except Exception:
            pass
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify validity
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    # Check minimum distance
    dot = best_pts @ best_pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    if min_dist < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_pts


# EVOLVE-BLOCK-START
# (The entire logic is above; this block marker is kept for compatibility)
# EVOLVE-BLOCK-END
