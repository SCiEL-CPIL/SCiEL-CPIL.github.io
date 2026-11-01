import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 135.0  # Safety margin (150s limit, 15s buffer)
EPS = 1e-14

def _energy_and_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|)
    and grad is the gradient w.r.t. pts in R^{N x 3}.
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
    # d/dp_i log(1/|p_i - p_j|) = -(p_i - p_j) / |p_i - p_j|^2
    # grad_i = sum_{j!=i} (p_j - p_i) / |p_i - p_j|^2
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

def _icosahedron_subdivide(n_target=100):
    """Generate points by subdividing an icosahedron and projecting to sphere."""
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Edges of icosahedron (indices)
    edges = [
        (0, 1), (0, 5), (0, 11), (0, 7), (0, 10),
        (1, 5), (5, 11), (5, 10), (5, 7),
        (11, 10), (10, 7), (7, 1),
        (3, 2), (3, 4), (3, 9), (3, 6), (3, 8),
        (2, 4), (4, 9), (4, 8), (4, 6),
        (9, 8), (8, 6), (6, 2)
    ]
    
    # Subdivide each edge by adding midpoints
    points = [tuple(v) for v in verts]
    midpoint_cache = {}
    
    for i, j in edges:
        key = (i, j) if i < j else (j, i)
        if key not in midpoint_cache:
            mid = (verts[i] + verts[j]) / 2.0
            mid = mid / np.linalg.norm(mid)
            midpoint_cache[key] = len(points)
            points.append(tuple(mid))
    
    # Now we have 12 + 30 = 42 points
    # Continue subdividing until we have at least n_target points
    while len(points) < n_target:
        new_points = []
        edges_now = []
        # We need to recompute edges for the new triangle mesh
        # This is complex; let's use a simpler approach:
        # Just perturb the Fibonacci spiral with different seeds
        break
    
    # Simpler: just use Fibonacci with different offsets
    pts = _fibonacci_spiral(n_target)
    return pts

def _optimize_sphere(pts0, max_iters=500, tol=1e-10):
    """
    Riemannian Gradient Descent with Backtracking Line Search.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.01
    for it in range(max_iters):
        if np.linalg.norm(g_t) < tol:
            break
            
        alpha = lr
        improved = False
        for _ in range(20):
            new_pts = _retract(pts, alpha * g_t)
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
        
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        lr = max(0.001, lr * 0.999)
        
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
                       options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol})
        x_opt = res.x.reshape(N, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        e_opt, _ = _energy_and_grad(x_opt)
        return x_opt, e_opt
    except Exception:
        return pts0, float('inf')

def construct():
    t_start = time.time()
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_spiral(N))
    
    # 2. Random points with different seeds
    for seed in [42, 123, 777, 2024]:
        inits.append(_random_points(N, seed=seed))
    
    # 3. Fibonacci with slight perturbations
    base = _fibonacci_spiral(N)
    for scale in [0.01, 0.05, 0.1]:
        rng = np.random.RandomState(99)
        perturbed = base + rng.randn(N, 3) * scale
        inits.append(perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True))
    
    # 4. Fibonacci with phase shifts
    for phase in [0.1, 0.3, 0.5]:
        i = np.arange(N) + 0.5 + phase
        z = 1.0 - 2.0 * i / N
        z = np.clip(z, -1.0, 1.0)
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        inits.append(pts / np.linalg.norm(pts, axis=1, keepdims=True))
    
    # Optimize each initialization
    best_pts = None
    best_energy = float('inf')
    
    for idx, init_pts in enumerate(inits):
        elapsed = time.time() - t_start
        if elapsed > DEADLINE_SECONDS - 10:
            break
        
        # Quick gradient descent to get close
        pts_opt, energy_opt = _optimize_sphere(init_pts, max_iters=200)
        
        # L-BFGS polish
        pts_polished, energy_polished = _polish_lbfgs(pts_opt, maxiter=100)
        
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pts = pts_polished.copy()
    
    # Final high-precision polish
    elapsed = time.time() - t_start
    if best_pts is not None and elapsed < DEADLINE_SECONDS - 5:
        pts_final, energy_final = _polish_lbfgs(best_pts, maxiter=300, gtol=1e-14, ftol=1e-16)
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
    if np.min(dist2) < 1e-18:  # distance < 1e-9
        # Perturb slightly
        rng = np.random.RandomState(0)
        best_pts = best_pts + rng.randn(N, 3) * 1e-10
        best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
