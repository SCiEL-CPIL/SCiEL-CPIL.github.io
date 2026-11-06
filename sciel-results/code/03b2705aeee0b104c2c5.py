"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            d = np.linalg.norm(pi - points[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fast_energy(points):
    """Vectorized energy computation. Returns (energy, min_dist_sq)."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    min_dist_sq = float(np.min(dist_sq))
    if min_dist_sq < 1e-12:
        return float('inf'), min_dist_sq
    
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij) = -0.5 * sum_{i<j} log(d2_ij)
    # sum_{i<j} log(d2) = 0.5 * sum_{i!=j} log(d2)
    # E = -0.25 * sum_{i!=j} log(d2_ij)
    energy = -0.25 * float(np.sum(np.log(dist_sq)))
    return energy, min_dist_sq


def _fast_grad_tangent(points):
    """Compute tangent-space gradient of log energy using vectorized ops.
    
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    dE/dx_i = -sum_{k != i} (x_i - x_k) / d_ik^2
    Project to tangent: grad_t = grad - (grad . x_i) * x_i
    """
    n = points.shape[0]
    # diff[i,k] = x_i - x_k
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # grad[i] = -sum_{k!=i} (x_i - x_k) / d_ik^2
    # = -sum_{k!=i} diff[i,k] / dist_sq[i,k]
    inv_d2 = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-24
    inv_d2[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = -sum_k inv_d2[i,k] * diff[i,k]
    grad = -np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    # Project to tangent space
    dot = np.sum(grad * points, axis=1, keepdims=True)
    grad = grad - dot * points
    
    return grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).astype(np.float64)


def _random_sphere(n, seed=0):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _gradient_descent(points, deadline, max_iters=500, lr_init=0.01):
    """Run projected gradient descent with tangent-space gradient."""
    pts = points.copy()
    energy, _ = _fast_energy(pts)
    lr = lr_init
    lr_decay = 0.995
    lr_shrink = 0.5
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        grad = _fast_grad_tangent(pts)
        
        # Try the step
        new_pts = pts - lr * grad
        new_pts = _project_to_sphere(new_pts)
        new_energy, min_d2 = _fast_energy(new_pts)
        
        if min_d2 < 1e-12:
            # Points too close, perturb
            pts = pts + 0.01 * np.random.randn(N, 3)
            pts = _project_to_sphere(pts)
            continue
        
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
            lr *= lr_decay
        else:
            lr *= lr_shrink
            if lr < 1e-10:
                break
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
    
    return pts, energy


def _polish(points, deadline):
    """Final high-precision polish using L-BFGS-B with sphere constraint.
    
    We parameterize each point by 2 angles (theta, phi) to remove the sphere constraint.
    """
    pts = points.copy()
    energy, _ = _fast_energy(pts)
    
    # Convert to spherical coordinates
    # x = sin(theta)*cos(phi), y = sin(theta)*sin(phi), z = cos(theta)
    z = pts[:, 2]
    x = pts[:, 0]
    y = pts[:, 1]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    params = np.column_stack([theta, phi])
    
    def _params_to_sphere(p):
        th = p[::2]
        ph = p[1::2]
        x = np.sin(th) * np.cos(ph)
        y = np.sin(th) * np.sin(ph)
        z = np.cos(th)
        return np.column_stack([x, y, z])
    
    def _objective(p):
        sp = _params_to_sphere(p)
        e, _ = _fast_energy(sp)
        return e
    
    def _gradient(p):
        sp = _params_to_sphere(p)
        g = _fast_grad_tangent(sp)
        # Convert tangent gradient to parameter gradient
        # dx/dtheta = cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)
        # dx/dphi = -sin(theta)*sin(phi), sin(theta)*cos(phi), 0
        th = p[::2]
        ph = p[1::2]
        ct = np.cos(th)
        st = np.sin(th)
        cp = np.cos(ph)
        sp_ = np.sin(ph)
        
        # dE/dtheta = grad . (cos(th)*cos(ph), cos(th)*sin(ph), -sin(th))
        dtheta = g[:, 0] * ct * cp + g[:, 1] * ct * sp_ + g[:, 2] * (-st)
        # dE/dphi = grad . (-sin(th)*sin(ph), sin(th)*cos(ph), 0)
        dphi = g[:, 0] * (-st * sp_) + g[:, 1] * (st * cp)
        
        gp = np.empty_like(p)
        gp[::2] = dtheta
        gp[1::2] = dphi
        return gp
    
    # Run L-BFGS-B
    try:
        result = minimize(_objective, params.flatten(), method='L-BFGS-B',
                         jac=_gradient, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12,
                                                  'maxfun': 10000, 'disp': False})
        if time.time() < deadline:
            if result.fun < energy:
                pts = _params_to_sphere(result.x)
                energy, _ = _fast_energy(pts)
    except Exception:
        pass
    
    # Additional fine gradient descent polish
    lr = 0.0001
    for it in range(500):
        if time.time() > deadline:
            break
        grad = _fast_grad_tangent(pts)
        new_pts = pts - lr * grad
        new_pts = _project_to_sphere(new_pts)
        new_energy, min_d2 = _fast_energy(new_pts)
        if min_d2 < 1e-12:
            break
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
            lr *= 0.99
        else:
            lr *= 0.5
            if lr < 1e-14:
                break
        if np.linalg.norm(grad) < 1e-14:
            break
    
    return pts, energy


def construct():
    """Place N points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Start with Fibonacci sphere
    best_pts = _fibonacci_sphere(N)
    best_energy, _ = _fast_energy(best_pts)
    
    # Phase 1: Gradient descent from Fibonacci initialization
    pts, energy = _gradient_descent(best_pts, deadline - 40.0, max_iters=300, lr_init=0.01)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts.copy()
    
    # Phase 2: Try random restarts with gradient descent
    for attempt in range(25):
        if time.time() > deadline - 30.0:
            break
        seed = attempt * 7919 + 13
        rand_pts = _random_sphere(N, seed=seed)
        pts, energy = _gradient_descent(rand_pts, deadline - 20.0, max_iters=200, lr_init=0.005)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 3: Try noise-perturbed Fibonacci
    for noise_scale in [0.01, 0.05, 0.1]:
        if time.time() > deadline - 20.0:
            break
        rng = np.random.RandomState(42)
        noisy_pts = _fibonacci_sphere(N) + noise_scale * rng.randn(N, 3)
        noisy_pts = _project_to_sphere(noisy_pts)
        pts, energy = _gradient_descent(noisy_pts, deadline - 15.0, max_iters=200, lr_init=0.005)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 4: Final polish with L-BFGS-B and fine GD
    if time.time() < deadline - 5.0:
        best_pts, best_energy = _polish(best_pts, deadline - 1.0)
    
    # Ensure validity
    best_pts = _project_to_sphere(best_pts)
    
    # Check minimum distance
    min_dist = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9 or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
