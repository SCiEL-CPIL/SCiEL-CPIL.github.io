"""Optimize 100 points on the unit sphere to minimize logarithmic energy."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 115.0  # safety margin below 120s

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _icosahedron_sphere(n):
    """Generate n points using icosahedron vertices as seeds, then fill with golden spiral."""
    # Start with icosahedron vertices (12 points)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    if n <= 12:
        return ico[:n]
    
    # Fill remaining points with golden spiral, avoiding existing points
    remaining = n - 12
    i = np.arange(remaining) + 0.5
    z = 1.0 - 2.0 * i / remaining
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    spiral = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    pts = np.vstack([ico, spiral])
    return pts

def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient for points on sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: for each point, sum over other points of (x_i - x_j)/|x_i - x_j|^2
    """
    n = len(pts)
    # Compute pairwise distances
    # dists[i,j] = |x_i - x_j|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Avoid division by zero (diagonal)
    mask = ~np.eye(n, dtype=bool)
    
    # Energy: -sum_{i<j} log(dists[i,j])
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: for point i, grad_i = sum_{j!=i} (x_i - x_j) / dists[i,j]^2
    # Note: d/dx_i [-log|x_i - x_j|] = -(x_i - x_j)/|x_i - x_j|^2
    # So grad_i = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    # Wait, let me be careful:
    # E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    # dE/dx_i for fixed i: sum_{j!=i} d/dx_i [-log|x_i - x_j|]
    # For j < i: term is -log|x_i - x_j|, d/dx_i = -(x_i - x_j)/|x_i - x_j|^2
    # For j > i: term is -log|x_j - x_i| = -log|x_i - x_j|, d/dx_i = -(x_i - x_j)/|x_i - x_j|^2
    # So grad_i = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    
    # Compute 1/dists^2 with mask
    inv_dists_sq = np.zeros((n, n))
    inv_dists_sq[mask] = 1.0 / (dists[mask] ** 2)
    
    # grad[i] = -sum_j (x_i - x_j) * inv_dists_sq[i,j]
    # = -x_i * sum_j inv_dists_sq[i,j] + sum_j x_j * inv_dists_sq[i,j]
    sum_inv = np.sum(inv_dists_sq, axis=1)
    grad = -pts * sum_inv[:, np.newaxis] + (inv_dists_sq @ pts)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.sqrt(np.sum(pts * pts, axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _momentum_step(pts, grad, lr, momentum, velocity):
    """Momentum gradient descent step on sphere."""
    velocity = momentum * velocity - lr * grad
    new_pts = pts + velocity
    return _project_to_sphere(new_pts), velocity

def _simulated_annealing(pts, energy_fn, n_iter=500, t_start=1.0, t_end=0.01, seed=42):
    """Simulated annealing to escape local minima."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_energy = energy_fn(current)
    best = current.copy()
    best_energy = current_energy
    
    for it in range(n_iter):
        t = t_start * (t_end / t_start) ** (it / n_iter)
        
        # Propose perturbation
        idx = rng.randint(len(current))
        perturbation = rng.randn(3) * 0.1
        proposed = current.copy()
        proposed[idx] += perturbation
        proposed = _project_to_sphere(proposed)
        
        proposed_energy = energy_fn(proposed)
        delta = proposed_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-10)):
            current = proposed
            current_energy = proposed_energy
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
    
    return best, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    def energy_only(pts):
        e, _ = _log_energy_and_grad(pts)
        return e
    
    def grad_only(pts):
        _, g = _log_energy_and_grad(pts)
        return g
    
    # Try multiple initializations
    initializations = [
        ("fibonacci", _fibonacci_sphere(N)),
        ("icosahedron", _icosahedron_sphere(N)),
    ]
    
    np.random.seed(42)
    for _ in range(3):
        random_pts = np.random.randn(N, 3)
        random_pts = _project_to_sphere(random_pts)
        initializations.append(("random", random_pts))
    
    best_pts = None
    best_energy = np.inf
    
    # Phase 1: Optimize from each initialization
    for name, init_pts in initializations:
        if time.time() > deadline - 30:
            break
        
        pts = init_pts.copy()
        e, _ = _log_energy_and_grad(pts)
        
        # L-BFGS-B from this start
        x0 = pts.flatten()
        try:
            res = minimize(
                energy_only, x0, jac=grad_only, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if np.all(np.isfinite(res.x)):
                candidate = _project_to_sphere(res.x.reshape(N, 3))
                e_cand, _ = _log_energy_and_grad(candidate)
                if e_cand < best_energy and np.all(np.isfinite(candidate)):
                    best_pts = candidate
                    best_energy = e_cand
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = initializations[0][1].copy()
        best_energy, _ = _log_energy_and_grad(best_pts)
    
    # Phase 2: Simulated annealing to escape local minima
    if time.time() < deadline - 40:
        sa_pts, sa_energy = _simulated_annealing(
            best_pts, energy_only, n_iter=200, t_start=0.5, t_end=0.01, seed=42
        )
        if sa_energy < best_energy:
            best_pts = sa_pts
            best_energy = sa_energy
        
        # L-BFGS-B polish after SA
        x0 = best_pts.flatten()
        try:
            res = minimize(
                energy_only, x0, jac=grad_only, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if np.all(np.isfinite(res.x)):
                candidate = _project_to_sphere(res.x.reshape(N, 3))
                e_cand, _ = _log_energy_and_grad(candidate)
                if e_cand < best_energy and np.all(np.isfinite(candidate)):
                    best_pts = candidate
                    best_energy = e_cand
        except Exception:
            pass
    
    # Phase 3: More random restarts with momentum + L-BFGS-B
    n_restarts = 4
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        pts = np.random.randn(N, 3)
        pts = _project_to_sphere(pts)
        
        # Adaptive momentum gradient descent
        lr = 0.02
        momentum = 0.9
        velocity = np.zeros_like(pts)
        prev_energy = np.inf
        
        for it in range(400):
            if time.time() > deadline - 20:
                break
            e, g = _log_energy_and_grad(pts)
            
            # Adaptive learning rate
            if e > prev_energy:
                lr *= 0.9
            elif it % 50 == 0:
                lr = min(lr * 1.1, 0.05)
            prev_energy = e
            
            pts, velocity = _momentum_step(pts, g, lr, momentum, velocity)
        
        # L-BFGS-B polish
        x0 = pts.flatten()
        try:
            res = minimize(
                energy_only, x0, jac=grad_only, method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if np.all(np.isfinite(res.x)):
                candidate = _project_to_sphere(res.x.reshape(N, 3))
                e_cand, _ = _log_energy_and_grad(candidate)
                if e_cand < best_energy and np.all(np.isfinite(candidate)):
                    best_pts = candidate
                    best_energy = e_cand
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                energy_only, x0, jac=grad_only, method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-15}
            )
            if np.all(np.isfinite(res.x)):
                candidate = _project_to_sphere(res.x.reshape(N, 3))
                e_cand, _ = _log_energy_and_grad(candidate)
                if e_cand < best_energy and np.all(np.isfinite(candidate)):
                    best_pts = candidate
                    best_energy = e_cand
        except Exception:
            pass
    
    # Ensure points are on unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Check for points too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        min_idx = np.unravel_index(np.argmin(dists[iu]), (N, N))
        perturbation = np.random.randn(3) * 1e-6
        best_pts[min_idx[0]] += perturbation
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
