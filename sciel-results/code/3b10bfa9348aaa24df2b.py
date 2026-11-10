"""Optimize 100 points on unit sphere to minimize logarithmic energy."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy(pts):
    """Compute logarithmic energy using math.fsum for precision."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_dists = dist[mask]
    # Use math.fsum for the score as specified
    energy_terms = [-math.log(d) for d in valid_dists]
    return math.fsum(energy_terms)

def _energy_np(pts):
    """Compute logarithmic energy using numpy for speed in optimization."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_dists = dist[mask]
    return -np.sum(np.log(valid_dists))

def _energy_grad(pts):
    """Compute gradient of logarithmic energy."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    # Gradient of -log(d) w.r.t x_i is (x_i - x_j) / d^2
    grad_terms = diff / dist_sq[..., np.newaxis]
    grad = np.sum(grad_terms, axis=1)
    return grad

def _project_tangent(pts, grad):
    """Project gradient onto tangent space of sphere."""
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - dot * pts

def _retract_sphere(pts, grad):
    """Geodesic retraction on the sphere."""
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    # Use geodesic: p + sin(t*||g||)/||g|| * g, where t=1
    return pts + np.sin(norms) / norms * grad

def _fibonacci_sphere(n):
    """Generate Fibonacci lattice points on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_sphere(n, seed=12345):
    """Generate Halton sequence points on sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(1, n + 1):
        u1 = halton(i, 2)
        u2 = halton(i, 3)
        z = 1.0 - 2.0 * u1
        phi = 2.0 * np.pi * u2
        r = np.sqrt(1.0 - z * z)
        points.append([r * np.cos(phi), r * np.sin(phi), z])
    return np.array(points)

def _random_sphere(n, seed=0):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, 3)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

def _optimize_riemannian(pts, max_iter=300, lr_init=0.01, time_budget=10.0):
    """Riemannian gradient descent with geodesic retraction."""
    start_time = time.time()
    current_pts = pts.copy()
    current_energy = _energy_np(current_pts)
    lr = lr_init
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        grad = _energy_grad(current_pts)
        grad_tan = _project_tangent(current_pts, grad)
        
        # Adaptive step size
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-10:
            break
            
        # Line search with backtracking
        alpha = lr
        best_alpha = alpha
        best_energy = current_energy
        
        for _ in range(10):
            new_pts = _retract_sphere(current_pts, -alpha * grad_tan)
            new_energy = _energy_np(new_pts)
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
            else:
                alpha *= 0.5
                if alpha < 1e-8:
                    break
        
        current_pts = best_pts
        current_energy = best_energy
        
        # Reduce learning rate
        lr *= 0.95
        
    return current_pts, current_energy

def _simulated_annealing(pts, steps=500, T_init=1.0, T_final=0.01, time_budget=8.0):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    current_pts = pts.copy()
    current_energy = _energy_np(current_pts)
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    T = T_init
    cool_rate = (T_final / T_init) ** (1.0 / steps)
    
    for step in range(steps):
        if time.time() - start_time > time_budget:
            break
        
        # Random perturbation
        rng = np.random.RandomState(42 + step)
        perturb = rng.randn(N, 3) * 0.1
        perturb = _project_tangent(current_pts, perturb)
        new_pts = _retract_sphere(current_pts, perturb)
        
        new_energy = _energy_np(new_pts)
        delta_E = new_energy - current_energy
        
        # Acceptance criterion
        if delta_E < 0 or rng.rand() < math.exp(-delta_E / max(T, 1e-10)):
            current_pts = new_pts
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
        
        # Cool down
        T *= cool_rate
    
    return best_pts, best_energy

def _newton_refine(pts, max_iter=50, time_budget=5.0):
    """Newton-Raphson refinement using the Hessian approximation."""
    start_time = time.time()
    current_pts = pts.copy()
    current_energy = _energy_np(current_pts)
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        grad = _energy_grad(current_pts)
        grad_tan = _project_tangent(current_pts, grad)
        
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-8:
            break
        
        # Approximate Hessian using diagonal of gradient change
        # Use simple gradient descent with adaptive step as approximation
        alpha = 1.0 / (grad_norm + 1e-8)
        
        new_pts = _retract_sphere(current_pts, -alpha * grad_tan)
        new_energy = _energy_np(new_pts)
        
        if new_energy < current_energy:
            current_pts = new_pts
            current_energy = new_energy
        else:
            # Try smaller step
            alpha *= 0.5
            new_pts = _retract_sphere(current_pts, -alpha * grad_tan)
            new_energy = _energy_np(new_pts)
            if new_energy < current_energy:
                current_pts = new_pts
                current_energy = new_energy
    
    return current_pts, current_energy

def _lbfbs_polish(pts, time_budget=5.0):
    """Final L-BFGS-B polish step."""
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        pts_local = x.reshape(-1, 3)
        norms = np.linalg.norm(pts_local, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts_local = pts_local / norms
        
        e = _energy_np(pts_local)
        g = _energy_grad(pts_local)
        g_tan = _project_tangent(pts_local, g)
        return e, g_tan.flatten()
    
    res = minimize(
        fun_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
    )
    
    final_pts = res.x.reshape(-1, 3)
    norms = np.linalg.norm(final_pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    final_pts = final_pts / norms
    
    return final_pts, _energy_np(final_pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # List of diverse initializers
    initializers = [
        _fibonacci_sphere,
        _halton_sphere,
        lambda n: _random_sphere(n, seed=1),
        lambda n: _random_sphere(n, seed=2),
        lambda n: _random_sphere(n, seed=3),
        lambda n: _random_sphere(n, seed=4),
        lambda n: _random_sphere(n, seed=5),
        lambda n: _random_sphere(n, seed=6),
        lambda n: _random_sphere(n, seed=7),
        lambda n: _random_sphere(n, seed=8),
    ]
    
    for init_func in initializers:
        if time.time() > deadline:
            break
        
        try:
            start_pts = init_func(N)
            if not np.all(np.isfinite(start_pts)):
                continue
            
            # Stage 1: Riemannian gradient descent
            pts1, e1 = _optimize_riemannian(start_pts, max_iter=150, time_budget=5.0)
            
            # Stage 2: Simulated annealing to escape local minima
            pts2, e2 = _simulated_annealing(pts1, steps=300, time_budget=6.0)
            
            # Stage 3: Newton refinement
            pts3, e3 = _newton_refine(pts2, max_iter=30, time_budget=4.0)
            
            # Stage 4: L-BFGS-B polish
            pts4, e4 = _lbfbs_polish(pts3, time_budget=5.0)
            
            # Check validity
            if not np.all(np.isfinite(pts4)):
                continue
            
            # Check min distance
            diff = pts4[:, np.newaxis, :] - pts4[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, 1.0)
            if np.min(dists) < 1e-9:
                continue
            
            # Use precise energy for comparison
            precise_e = _energy(pts4)
            
            if precise_e < best_energy:
                best_energy = precise_e
                best_pts = pts4
                
        except Exception:
            continue
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END
