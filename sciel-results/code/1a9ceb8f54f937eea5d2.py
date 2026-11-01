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
    dots = pts @ pts.T
    dist2 = 2.0 - 2.0 * dots
    dist2 = np.maximum(dist2, 0.0)
    mask = ~np.eye(N, dtype=bool)
    dist = np.sqrt(np.where(mask, np.maximum(dist2, EPS), 1.0))
    
    log_terms = np.where(mask, -np.log(dist), 0.0)
    energy = 0.5 * np.sum(log_terms)
    
    inv_dist2 = np.where(mask, 1.0 / np.maximum(dist2, EPS), 0.0)
    
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

def _spherical_grid(n):
    """Generate points using a more uniform spherical grid."""
    golden = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    z = 1.0 - 2.0 * (i + 0.5) / n
    theta = golden * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _icosahedron_subdivide(n_target=100):
    """Generate points by subdividing an icosahedron and projecting to sphere."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    points = list(verts)
    
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i+1) % 3]
            edge = (min(a, b), max(a, b))
            edges.add(edge)
    
    for a, b in edges:
        mid = (verts[a] + verts[b]) / 2.0
        mid = mid / np.linalg.norm(mid)
        points.append(mid)
    
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        center = center / np.linalg.norm(center)
        points.append(center)
    
    rng = np.random.RandomState(123)
    while len(points) < n_target:
        base_idx = rng.randint(0, len(points))
        base = points[base_idx]
        new_pt = base + rng.randn(3) * 0.3
        new_pt = new_pt / np.linalg.norm(new_pt)
        points.append(new_pt)
    
    return np.array(points[:n_target])

def _riemannian_descent(pts0, max_iters=500, tol=1e-10, deadline=None):
    """
    Riemannian Gradient Descent with adaptive step size and backtracking.
    Uses a more aggressive initial step size and better convergence criteria.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.05  # More aggressive initial learning rate
    min_lr = 1e-12
    patience = 0
    max_patience = 50
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if np.linalg.norm(g_t) < tol:
            break
            
        alpha = lr
        improved = False
        for _ in range(40):
            new_pts = _retract(pts, -alpha * g_t)  # Negative gradient direction
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(g_t * g_t):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            patience += 1
            lr *= 0.5
            if lr < min_lr or patience > max_patience:
                break
            continue
        
        patience = 0
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Adaptive learning rate
        if energy < best_energy - 1e-6:
            lr = min(0.2, lr * 1.1)
        else:
            lr = max(min_lr, lr * 0.99)
        
    return best_pts, best_energy

def _polish_lbfgs(pts0, maxiter=200, gtol=1e-12, ftol=1e-15):
    """Polish using L-BFGS-B with tangent projection."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        norms = np.linalg.norm(x_pts, axis=1, keepdims=True)
        norms = np.where(norms < EPS, 1.0, norms)
        x_pts = x_pts / norms
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
        
    x0 = pts0.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol,
                                'maxls': 50, 'm': 20})
        x_opt = res.x.reshape(N, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        e_opt, _ = _energy_and_grad(x_opt)
        return x_opt, e_opt
    except Exception:
        return pts0, float('inf')

def _polish_bfgs(pts0, maxiter=100, gtol=1e-14):
    """Polish using BFGS for higher precision."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        norms = np.linalg.norm(x_pts, axis=1, keepdims=True)
        norms = np.where(norms < EPS, 1.0, norms)
        x_pts = x_pts / norms
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
        
    x0 = pts0.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='BFGS',
                       options={'maxiter': maxiter, 'gtol': gtol,
                                'maxls': 50})
        x_opt = res.x.reshape(N, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        e_opt, _ = _energy_and_grad(x_opt)
        return x_opt, e_opt
    except Exception:
        return pts0, float('inf')

def _high_precision_polish(pts0, deadline):
    """
    Final high-precision polish pass with multiple stages:
    1. L-BFGS with very tight tolerance
    2. Riemannian gradient descent with very small tolerance
    3. BFGS for final convergence
    """
    best_pts = pts0.copy()
    best_energy = _energy_and_grad(pts0)[0]
    
    # Stage 1: L-BFGS with very tight tolerance
    if time.time() < deadline - 5:
        pts_lbfgs, energy_lbfgs = _polish_lbfgs(best_pts, maxiter=800, gtol=1e-14, ftol=1e-16)
        if energy_lbfgs < best_energy:
            best_pts = pts_lbfgs
            best_energy = energy_lbfgs
    
    # Stage 2: Riemannian gradient descent with very small tolerance
    if time.time() < deadline - 3:
        pts_rd, energy_rd = _riemannian_descent(best_pts, max_iters=200, tol=1e-13, deadline=deadline - 3)
        if energy_rd < best_energy:
            best_pts = pts_rd
            best_energy = energy_rd
    
    # Stage 3: BFGS for final convergence
    if time.time() < deadline - 1:
        pts_bfgs, energy_bfgs = _polish_bfgs(best_pts, maxiter=200, gtol=1e-15)
        if energy_bfgs < best_energy:
            best_pts = pts_bfgs
            best_energy = energy_bfgs
    
    return best_pts, best_energy

def _generate_diverse_inits():
    """Generate a diverse set of initial configurations."""
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_spiral(N))
    
    # 2. Random points with different seeds
    for seed in [42, 123, 777, 2024, 314, 271, 999, 555]:
        inits.append(_random_points(N, seed=seed))
    
    # 3. Spherical grid
    inits.append(_spherical_grid(N))
    
    # 4. Icosahedron subdivision
    inits.append(_icosahedron_subdivide(N))
    
    # 5. Fibonacci with slight perturbations
    base = _fibonacci_spiral(N)
    for scale in [0.01, 0.05, 0.1, 0.2, 0.3]:
        rng = np.random.RandomState(99 + int(scale * 100))
        perturbed = base + rng.randn(N, 3) * scale
        inits.append(perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True))
    
    # 6. Fibonacci with phase shifts
    for phase in [0.1, 0.3, 0.5, 0.7, 0.9]:
        i = np.arange(N) + 0.5 + phase
        z = 1.0 - 2.0 * i / N
        z = np.clip(z, -1.0, 1.0)
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        inits.append(pts / np.linalg.norm(pts, axis=1, keepdims=True))
    
    # 7. Structured grid with jitter
    for jitter_scale in [0.02, 0.05, 0.1]:
        grid = _spherical_grid(N)
        rng = np.random.RandomState(42 + int(jitter_scale * 100))
        jittered = grid + rng.randn(N, 3) * jitter_scale
        inits.append(jittered / np.linalg.norm(jittered, axis=1, keepdims=True))
    
    return inits

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Generate multiple diverse initializations
    inits = _generate_diverse_inits()
    
    # Optimize each initialization
    best_pts = None
    best_energy = float('inf')
    
    # Reserve time for final high-precision polish
    polish_time_reserve = 15.0
    
    for idx, init_pts in enumerate(inits):
        elapsed = time.time() - t_start
        if elapsed > DEADLINE_SECONDS - polish_time_reserve:
            break
        
        # Quick gradient descent to get close
        pts_opt, energy_opt = _riemannian_descent(init_pts, max_iters=300, deadline=deadline - polish_time_reserve)
        
        # L-BFGS polish
        pts_polished, energy_polished = _polish_lbfgs(pts_opt, maxiter=150)
        
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pts = pts_polished.copy()
    
    # Final high-precision polish
    elapsed = time.time() - t_start
    if best_pts is not None and elapsed < deadline - 2:
        pts_final, energy_final = _high_precision_polish(best_pts, deadline)
        if energy_final < best_energy:
            best_pts = pts_final
            best_energy = energy_final
    
    # Final check and return
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Verify no points are too close
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    if np.min(dist2) < 1e-18:
        rng = np.random.RandomState(0)
        best_pts = best_pts + rng.randn(N, 3) * 1e-10
        best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
