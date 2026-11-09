"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy(points):
    """Compute logarithmic energy using math.fsum for precision."""
    terms = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e100
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(points):
    """Compute energy and gradient of logarithmic energy."""
    E = 0.0
    grad = np.zeros_like(points)
    for i in range(N):
        for j in range(i + 1, N):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                return 1e100, grad
            inv_d = 1.0 / d
            E += math.log(inv_d)
            # d/dxi log(1/di) = -1/di * (xi - xj)/di = -(xi - xj)/di^2
            # d/dxj log(1/di) = +1/di * (xi - xj)/di = +(xi - xj)/di^2
            factor = inv_d * inv_d
            grad[i] -= diff * factor
            grad[j] += diff * factor
    return E, grad


def _project_tangent(points, grad):
    """Project gradient onto tangent space of the sphere."""
    # Remove radial component: grad_tangent = grad - (grad . x) * x
    for i in range(N):
        radial = np.dot(grad[i], points[i])
        grad[i] -= radial * points[i]
    return grad


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _fibonacci_sphere(n, offset=0.0):
    """Generate points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)


def _gd_refine(points, max_iters=5000, initial_lr=0.01, min_lr=1e-6, tol=1e-12):
    """Gradient descent on sphere with adaptive learning rate."""
    points = _normalize(points.copy())
    E, grad = _energy_and_grad(points)
    best_E = E
    best_points = points.copy()
    
    lr = initial_lr
    for it in range(max_iters):
        grad = _project_tangent(points, grad)
        
        # Adaptive step: use gradient magnitude to scale
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Step
        step = lr / (grad_norm + 1e-10)
        new_points = points - step * grad
        new_points = _normalize(new_points)
        
        new_E, new_grad = _energy_and_grad(new_points)
        
        if new_E < E:
            E = new_E
            points = new_points
            grad = new_grad
            if E < best_E - tol:
                best_E = E
                best_points = points.copy()
            lr = min(lr * 1.05, 0.1)  # Increase LR slightly
        else:
            lr *= 0.7
            if lr < min_lr:
                break
    
    return best_points, best_E


def _lbfgs_refine(points, maxiter=1000):
    """Refine using L-BFGS-B with spherical constraint handled by projection."""
    points = _normalize(points.copy())
    
    def objective_and_grad(x):
        pts = _normalize(x.reshape(-1, 3))
        E, grad = _energy_and_grad(pts)
        # Project gradient to tangent space for proper optimization on manifold
        grad = _project_tangent(pts, grad)
        return E, grad.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-10}
        )
        new_points = _normalize(result.x.reshape(-1, 3))
        new_E = _energy(new_points)
        
        # Check validity
        if np.all(np.isfinite(new_points)) and np.all(np.linalg.norm(new_points, axis=1) > 1e-10):
            # Check minimum distance
            min_d = 1e10
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(new_points[i] - new_points[j])
                    if d < min_d:
                        min_d = d
            if min_d > 1e-9:
                return new_points, new_E
    except Exception:
        pass
    
    return points, _energy(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_E = 1e100
    
    # Multiple initialization strategies
    initializations = []
    
    # 1. Fibonacci spiral (standard)
    initializations.append(_fibonacci_sphere(N))
    
    # 2. Fibonacci with different offset
    initializations.append(_fibonacci_sphere(N, offset=0.25))
    
    # 3. Random with fixed seed
    initializations.append(_random_sphere(N, seed=42))
    
    # 4. Random with different seed
    initializations.append(_random_sphere(N, seed=123))
    
    # 5. Slightly perturbed Fibonacci
    rng = np.random.RandomState(999)
    perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.1
    initializations.append(_normalize(perturbed))
    
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        # Phase 1: Gradient descent refinement
        gd_points, gd_E = _gd_refine(init_pts, max_iters=3000, initial_lr=0.005)
        
        if gd_E < best_E:
            best_E = gd_E
            best_points = gd_points.copy()
        
        # Phase 2: L-BFGS refinement
        if time.time() > deadline - 10:
            break
        
        lbfgs_points, lbfgs_E = _lbfgs_refine(gd_points, maxiter=500)
        
        if lbfgs_E < best_E:
            best_E = lbfgs_E
            best_points = lbfgs_points.copy()
        
        # If we have time, do another GD pass from the L-BFGS result
        if time.time() > deadline - 5:
            break
        
        gd2_points, gd2_E = _gd_refine(lbfgs_points, max_iters=2000, initial_lr=0.001)
        
        if gd2_E < best_E:
            best_E = gd2_E
            best_points = gd2_points.copy()
    
    # Final validation
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    best_points = _normalize(best_points)
    
    # Verify validity
    if not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_sphere(N)
        best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
