import numpy as np
import math
import time

N = 150
TIME_LIMIT = 115.0  # Safety margin

def _energy_and_grad(points):
    """Compute logarithmic energy and its gradient (tangent to sphere)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    dot = np.sum(grad * points, axis=1, keepdims=True)
    grad_tangent = grad - dot * points
    
    return energy, grad_tangent

def _exp_sphere(x, v):
    """Exponential map on the unit sphere."""
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    # Avoid division by zero
    mask = norm_v > 1e-12
    result = np.zeros_like(x)
    
    # For points where norm_v is effectively zero
    result[~mask[:, 0]] = x[~mask[:, 0]]
    
    # For points where norm_v > 0
    # x' = x * cos(theta) + (v / norm_v) * sin(theta)
    theta = norm_v[mask[:, 0]]
    v_dir = v[mask[:, 0]] / theta
    result[mask[:, 0]] = x[mask[:, 0]] * np.cos(theta) + v_dir * np.sin(theta)
    
    return result

def _project_to_sphere(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_riemannian_adam(points, max_iter=20000, time_limit=100.0):
    """Optimize using Riemannian Adam."""
    n = len(points)
    start_time = time.time()
    
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = 0.01
    
    m = np.zeros_like(points)
    v = np.zeros_like(points)
    
    best_energy, best_pts = _energy_and_grad(points)[0], points.copy()
    
    for t in range(1, max_iter + 1):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_grad(points)
        
        if energy < best_energy - 1e-15:
            best_energy = energy
            best_pts = points.copy()
            
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update vector in tangent space
        step = -lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Ensure step is tangent
        # (grad is already tangent, so m, v, step should be tangent if initialized as such)
        # But numerical drift might occur. Project step to be safe?
        # step is a linear combination of grads, which are tangent. So step is tangent.
        
        # Retract using exponential map
        points = _exp_sphere(points, step)
        
        # Normalize to ensure we stay on sphere (exp map should do this, but floating point errors)
        points = _project_to_sphere(points)
        
        if np.linalg.norm(grad) < 1e-12:
            break
            
    return best_pts, best_energy

def _optimize_lbfgsb(points, max_iter=5000, time_limit=50.0):
    """Optimize using L-BFGS-B on flattened coordinates with sphere constraint handled by projection."""
    from scipy.optimize import minimize
    
    x0 = points.ravel()
    n = len(points)
    start_time = time.time()
    
    def fun_and_grad(x):
        pts = x.reshape(n, 3)
        pts = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts)
        return e, g.ravel()
        
    result = minimize(fun_and_grad, x0, method='L-BFGS-B', 
                      jac=True, options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                                          'maxfun': 50000})
    elapsed = time.time() - start_time
    best_pts = _project_to_sphere(result.x.reshape(n, 3))
    return best_pts, result.fun, elapsed

def construct():
    start_time = time.time()
    best_energy = float('inf')
    best_points = None
    
    # Try multiple initializations
    initializations = []
    
    # 1. Fibonacci
    initializations.append(_fibonacci_sphere(N))
    
    # 2. Random
    rng = np.random.default_rng(42)
    for _ in range(3):
        pts = rng.standard_normal((N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        initializations.append(pts)
        
    # 3. Perturbed Fibonacci
    fib = _fibonacci_sphere(N)
    perturbed = fib + 0.1 * rng.standard_normal((N, 3))
    perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
    initializations.append(perturbed)
    
    for init_pts in initializations:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        # Phase 1: L-BFGS-B (Fast convergence to local min)
        try:
            pts, energy, elapsed = _optimize_lbfgsb(init_pts, max_iter=1000, time_limit=20.0)
        except Exception:
            continue
            
        if energy < best_energy:
            best_energy = energy
            best_points = pts
            
        # Phase 2: Riemannian Adam (Polish and escape shallow local mins)
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        try:
            pts, energy2 = _optimize_riemannian_adam(pts, max_iter=5000, time_limit=10.0)
            if energy2 < best_energy:
                best_energy = energy2
                best_points = pts
        except Exception:
            pass

    if best_points is None:
        best_points = _fibonacci_sphere(N)
        
    # Final check and cleanup
    best_points = _project_to_sphere(best_points)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                best_points[j] += 1e-8 * np.random.randn(3)
                best_points[j] = best_points[j] / np.linalg.norm(best_points[j])
                
    return best_points
