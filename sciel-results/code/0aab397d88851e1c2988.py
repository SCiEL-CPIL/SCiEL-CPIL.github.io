"""Coupled variational optimization: charged particles in tunable spherical harmonic potential well."""
import numpy as np
import math

N = 100


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return -math.fsum(math.log(di) for di in d)


def _log_energy_grad(points):
    """Gradient of log-energy w.r.t. each point in ambient R^3."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    grad = np.zeros_like(points)
    for k in range(n):
        for j in range(n):
            if j == k:
                continue
            d = dists[k, j]
            if d < 1e-15:
                continue
            grad[k] -= (points[k] - points[j]) / (d * d)
    return grad


def _sphere_penalty(points, mu):
    """Soft spherical harmonic potential: mu * sum((|x_i|^2 - 1)^2)."""
    r2 = np.sum(points * points, axis=1)
    return mu * np.sum((r2 - 1.0) ** 2)


def _sphere_penalty_grad(points, mu):
    """Gradient of sphere penalty w.r.t. points."""
    r2 = np.sum(points * points, axis=1, keepdims=True)
    return 4.0 * mu * (r2 - 1.0) * points


def _total_energy_and_grad(points, mu):
    """Total energy (log + sphere penalty) and its gradient."""
    e_log = _log_energy(points)
    g_log = _log_energy_grad(points)
    e_sph = _sphere_penalty(points, mu)
    g_sph = _sphere_penalty_grad(points, mu)
    return e_log + e_sph, g_log + g_sph


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_spiral():
    """Generate initial Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_coupled(points, mu, max_iters=300):
    """
    Optimize points with fixed mu using adaptive gradient descent.
    Returns optimized points and final energy.
    """
    pts = points.copy()
    lr = 0.01
    for it in range(max_iters):
        e, g = _total_energy_and_grad(pts, mu)
        if np.max(np.abs(g)) < 1e-10:
            break
        new_pts = pts - lr * g
        new_e, _ = _total_energy_and_grad(new_pts, mu)
        if new_e < e:
            pts = new_pts
            lr = min(lr * 1.03, 0.05)
        else:
            lr *= 0.5
            new_pts = pts - lr * g
            new_e, _ = _total_energy_and_grad(new_pts, mu)
            if new_e < e:
                pts = new_pts
    return pts


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy via coupled variational approach."""
    np.random.seed(42)
    fib = _fibonacci_spiral()
    
    best_points = fib.copy()
    best_energy = _log_energy(best_points)
    
    # Multi-start with different initializations and confinement strengths
    for restart in range(5):
        if restart == 0:
            pts_init = fib + 0.01 * np.random.randn(N, 3)
            mu_init = 5.0
        elif restart == 1:
            pts_init = np.random.randn(N, 3)
            pts_init = _project_to_sphere(pts_init)
            mu_init = 10.0
        elif restart == 2:
            pts_init = fib + 0.05 * np.random.randn(N, 3)
            pts_init = _project_to_sphere(pts_init)
            mu_init = 20.0
        elif restart == 3:
            # Slightly off-sphere
            pts_init = fib.copy()
            pts_init *= 1.01
            mu_init = 50.0
        else:
            pts_init = fib + 0.02 * np.random.randn(N, 3)
            mu_init = 1.0
        
        # Phase 1: Optimize points for fixed mu
        pts = _optimize_coupled(pts_init, mu_init, max_iters=200)
        
        # Phase 2: Adjust mu based on how well points are confined
        r2 = np.sum(pts * pts, axis=1)
        avg_dev = np.mean((r2 - 1.0) ** 2)
        if avg_dev > 0.01:
            mu_new = min(mu_init * 2.0, 500.0)
        elif avg_dev < 0.0005:
            mu_new = max(mu_init * 0.5, 0.5)
        else:
            mu_new = mu_init
        
        # Phase 3: Re-optimize with adjusted mu
        if mu_new != mu_init:
            pts = _optimize_coupled(pts, mu_new, max_iters=150)
        
        # Project to sphere and evaluate actual log energy
        pts_sphere = _project_to_sphere(pts)
        e_log = _log_energy(pts_sphere)
        
        if e_log < best_energy:
            best_energy = e_log
            best_points = pts_sphere.copy()
    
    # Final polish: fine-tune on sphere with tangent-space gradient descent
    pts = best_points.copy()
    for it in range(1000):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, 1.0)
        
        grad = np.zeros_like(pts)
        for k in range(N):
            for j in range(N):
                if j == k:
                    continue
                d = dists[k, j]
                if d < 1e-15:
                    continue
                grad[k] -= (pts[k] - pts[j]) / (d * d)
        
        # Project to tangent space
        grad_tan = np.zeros_like(grad)
        for i in range(N):
            p = pts[i]
            g = grad[i]
            radial = np.dot(g, p)
            grad_tan[i] = g - radial * p
        
        if np.max(np.abs(grad_tan)) < 1e-11:
            break
        
        lr = 0.001
        new_pts = pts - lr * grad_tan
        new_pts = _project_to_sphere(new_pts)
        e_new = _log_energy(new_pts)
        e_curr = _log_energy(pts)
        
        if e_new < e_curr:
            pts = new_pts
            if e_new < best_energy:
                best_energy = e_new
                best_points = new_pts.copy()
        else:
            lr *= 0.5
            new_pts = pts - lr * grad_tan
            new_pts = _project_to_sphere(new_pts)
            e_new2 = _log_energy(new_pts)
            if e_new2 < e_curr:
                pts = new_pts
                if e_new2 < best_energy:
                    best_energy = e_new2
                    best_points = new_pts.copy()
    
    result = _project_to_sphere(best_points)
    
    # Validate: ensure no points too close, all finite
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                result[j] += 1e-8 * np.random.randn(3)
                result[j] = result[j] / np.linalg.norm(result[j])
    
    if not np.all(np.isfinite(result)):
        result = _fibonacci_spiral()
    
    return result
