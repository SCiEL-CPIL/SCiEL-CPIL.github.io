"""Optimized logarithmic energy minimization for 100 points on the unit sphere.
Combines Riemannian gradient descent with Newton polish for tight convergence."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100
DEADLINE = 140.0  # Safety margin before 150s kill


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(points):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Using chordal distance |x_i - x_j| = sqrt(2 - 2*x_i·x_j)
    """
    n = points.shape[0]
    # Compute pairwise dot products
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    # Chordal distances squared: |xi - xj|^2 = 2 - 2*dot
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)  # avoid log(0)
    
    # log(1/d) = -log(d) = -0.5*log(d2)
    log_d2 = np.log(d2)
    # Energy = sum_{i<j} -0.5 * log(d2_ij)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(log_d2[iu])
    
    # Gradient: d/dx_i sum_{j!=i} -0.5*log(2 - 2*x_i·x_j)
    # = sum_{j!=i} (x_i - x_j) / (2 - 2*x_i·x_j)
    # Project onto tangent space
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2_safe = d2.copy()
    np.fill_diagonal(d2_safe, 1.0)
    inv_d2 = 1.0 / d2_safe
    inv_d2[np.eye(n, dtype=bool)] = 0.0
    
    grad = np.einsum('ij,ijc->c', inv_d2, diff)
    
    # Project gradient onto tangent space
    radial = np.sum(grad * points, axis=1, keepdims=True)
    grad = grad - radial * points
    
    return energy, grad


def _energy_only(points):
    """Just the energy value (no gradient)."""
    energy, _ = _energy_and_grad(points)
    return energy


def _hessian_approx(points):
    """Approximate Hessian using finite differences of the gradient."""
    n = points.shape[0]
    e, g = _energy_and_grad(points)
    return e, g


def _exponential_map(points, tangent_vecs):
    """Riemannian exponential map on the sphere."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    unit_tangent = tangent_vecs / norms_safe
    cos_t = np.cos(norms)
    sin_t = np.sin(norms)
    new_points = points * cos_t + unit_tangent * sin_t
    return _normalize(new_points)


def _riemannian_gradient_descent(points, deadline, max_iter=3000, lr_init=0.01):
    """Riemannian gradient descent with backtracking line search."""
    current = points.copy()
    best_e = _energy_only(current)
    best_pts = current.copy()
    lr = lr_init
    c1 = 0.1
    rho = 0.5
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_and_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            break
        
        # Backtracking line search
        step = lr
        found = False
        for bt in range(15):
            tangent = -step * g
            new_pts = _exponential_map(current, tangent)
            new_e = _energy_only(new_pts)
            if new_e <= e + c1 * step * g_norm * g_norm:
                found = True
                break
            step *= rho
        
        if not found:
            step = max(lr * rho**15, 1e-15)
            tangent = -step * g
            new_pts = _exponential_map(current, tangent)
            new_e = _energy_only(new_pts)
        
        if new_e < best_e - 1e-14:
            current = new_pts
            best_e = new_e
            best_pts = current.copy()
            no_improve = 0
            lr = min(step * 1.3, 0.05)
        elif new_e < best_e:
            current = new_pts
            no_improve += 1
            lr = step
        else:
            lr = step * 0.5
            no_improve += 1
            if lr < 1e-14 or no_improve > 40:
                break
    
    return best_pts, best_e


def _newton_polish(points, deadline, max_iter=200):
    """Newton polish using approximate Hessian via finite differences."""
    current = points.copy()
    best_e = _energy_only(current)
    best_pts = current.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_and_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-13:
            break
        
        # Approximate Hessian using Gauss-Newton style
        # For log energy, the Hessian has a specific structure
        n = current.shape[0]
        dots = current @ current.T
        dots = np.clip(dots, -1.0, 1.0)
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0)
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        
        # Hessian approximation: H_ij = 2*(x_i - x_j)(x_i - x_j)^T / d2_ij^2 - delta_ij * sum_k (x_i - x_k)(x_i - x_k)^T / d2_ik^2
        # Simplified: use diagonal approximation for efficiency
        # H_ii ≈ sum_j (x_i - x_j)(x_i - x_j)^T / d2_ij^2 - (sum_j 1/d2_ij) * I
        # For Newton step, we need to solve H * delta = -g in tangent space
        
        # Use simplified Newton: delta = -g / max(diag(H), eps)
        # Compute diagonal of Hessian
        diag_H = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff_ij = current[i] - current[j]
                    diag_H[i] += np.dot(diff_ij, diff_ij) / (d2[i, j] ** 2)
            diag_H[i] -= np.sum(1.0 / d2[i, np.arange(n) != i] if n > 1 else 0)
        
        diag_H = np.maximum(diag_H, 1e-8)
        
        # Newton step in Euclidean space
        delta = -g / diag_H[:, np.newaxis]
        
        # Project to tangent space
        radial = np.sum(delta * current, axis=1, keepdims=True)
        delta = delta - radial * current
        
        # Line search
        step = 1.0
        for bt in range(10):
            new_pts = _exponential_map(current, step * delta)
            new_e = _energy_only(new_pts)
            if new_e < e:
                break
            step *= 0.5
        
        new_pts = _exponential_map(current, step * delta)
        new_e = _energy_only(new_pts)
        
        if new_e < best_e:
            current = new_pts
            best_e = new_e
            best_pts = current.copy()
        else:
            if step < 1e-6:
                break
    
    return best_pts, best_e


def _fibonacci_spiral(n, offset=0.5):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _random_points(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _shifted_spiral(n, shift):
    """Generate a shifted spiral pattern."""
    i = np.arange(n) + 0.5 + shift
    z = 1.0 - 2.0 * np.clip(i / n, 0, 1)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + shift * n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Standard Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # 2. Perturbed Fibonacci spiral
    base = _fibonacci_spiral(N)
    perturbed = base + 0.1 * rng.standard_normal((N, 3))
    initializations.append(_normalize(perturbed))
    
    # 3. Shifted spirals
    for shift in [0.1, 0.3, 0.7]:
        initializations.append(_shifted_spiral(N, shift))
    
    # 4. Random points
    for _ in range(3):
        initializations.append(_random_points(N, rng))
    
    # 5. Anti-symmetric pairs
    half = N // 2
    pts_half = _random_points(half, rng)
    pts_full = np.vstack([pts_half, -pts_half])
    if N % 2 == 1:
        pts_full = np.vstack([pts_full, np.array([[0, 0, 1.0]])])
    initializations.append(_normalize(pts_full))
    
    # Optimize each initialization
    best_points = initializations[0]
    best_energy = _energy_only(best_points)
    
    # Phase 1: Riemannian gradient descent for each init
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        try:
            pts, e = _riemannian_gradient_descent(init_pts, deadline - 25.0, max_iter=2000)
            if e < best_energy:
                best_energy = e
                best_points = pts
        except Exception:
            continue
    
    # Phase 2: Newton polish on best result
    if time.time() < deadline - 10.0:
        try:
            pts, e = _newton_polish(best_points, deadline - 5.0, max_iter=100)
            if e < best_energy:
                best_energy = e
                best_points = pts
        except Exception:
            pass
    
    # Phase 3: Final L-BFGS-B polish with tight tolerance
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                fun=lambda x: _energy_only(_normalize(x)),
                x0=best_points.flatten(),
                jac=lambda x: _energy_and_grad(_normalize(x))[1].flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxls': 30
                }
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_points = _normalize(result.x.reshape(N, 3))
        except Exception:
            pass
    
    # Ensure valid output
    best_points = _normalize(best_points)
    # Check for any points too close together
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                best_points[j] = best_points[j] + 1e-8 * rng.standard_normal(3)
                best_points = _normalize(best_points)
    
    return best_points
