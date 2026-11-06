"""
Minimize logarithmic energy of 100 points on the unit sphere.
Uses a combination of:
1. Multiple diverse initializations (Fibonacci, random, icosahedron)
2. Riemannian L-BFGS optimization on the sphere
3. Newton-Raphson polishing with tangent-space Hessian
4. Simulated annealing for escaping local minima
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy: sum_{i<j} log(1/|p_i - p_j|)
    Gradient: sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2, projected to tangent space.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask_upper]
    
    if np.any(d_upper < 1e-9):
        return 1e18, np.zeros_like(pts)
        
    energy = -np.sum(np.log(d_upper))
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    grad = np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - radial * pts
    
    return energy, grad

def _hessian_tangent(pts):
    """
    Compute the projected Hessian of the log-energy on the tangent space.
    """
    n = len(pts)
    H = np.zeros((n*3, n*3))
    
    for i in range(n):
        for j in range(i+1, n):
            rij = pts[i] - pts[j]
            d2 = np.dot(rij, rij)
            if d2 < 1e-18:
                continue
            d4 = d2 * d2
            block_ii = np.eye(3) / d2 - np.outer(rij, rij) / d4
            block_ij = -np.eye(3) / d2 + np.outer(rij, rij) / d4
            
            H[i*3:(i+1)*3, i*3:(i+1)*3] += block_ii
            H[j*3:(j+1)*3, j*3:(j+1)*3] += block_ii
            H[i*3:(i+1)*3, j*3:(i+1)*3] += block_ij
            H[j*3:(j+1)*3, i*3:(i+1)*3] += block_ij
    
    H_t = np.zeros_like(H)
    for i in range(n):
        Pi = np.eye(3) - np.outer(pts[i], pts[i])
        for j in range(n):
            Pj = np.eye(3) - np.outer(pts[j], pts[j])
            H_t[i*3:(i+1)*3, j*3:(j+1)*3] = Pi @ H[i*3:(i+1)*3, j*3:(j+1)*3] @ Pj
    
    return H_t

def _exp_map(pts, delta):
    """
    Exponential map on the sphere: move each point along a geodesic.
    """
    new_pts = np.zeros_like(pts)
    for i in range(len(pts)):
        d = delta[i]
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-15:
            new_pts[i] = pts[i]
        else:
            angle = d_norm
            v = d / d_norm
            new_pts[i] = np.cos(angle) * pts[i] + np.sin(angle) * v
            new_pts[i] /= np.linalg.norm(new_pts[i])
    return new_pts

def _newton_polish(pts0, max_iter=30, damping=0.5):
    """
    Newton-Raphson polishing on the sphere using tangent space Hessian.
    """
    pts = pts0.copy()
    n = len(pts)
    
    for it in range(max_iter):
        e, g = _energy_and_grad(pts)
        if np.linalg.norm(g, ord=np.inf) < 1e-14:
            break
        
        H = _hessian_tangent(pts)
        g_flat = g.flatten()
        
        reg = 1e-10
        H_reg = H + reg * np.eye(n*3)
        
        try:
            delta_flat = np.linalg.solve(H_reg, -g_flat)
        except np.linalg.LinAlgError:
            delta_flat = np.linalg.lstsq(H_reg, -g_flat, rcond=None)[0]
        
        delta = delta_flat.reshape(n, 3)
        for i in range(n):
            radial = np.dot(delta[i], pts[i])
            delta[i] -= radial * pts[i]
        
        step = 1.0
        for _ in range(20):
            new_pts = _exp_map(pts, step * delta)
            e_new, _ = _energy_and_grad(new_pts)
            if e_new < e:
                pts = new_pts
                break
            step *= 0.5
        
        if step < 1e-10:
            break
    
    return pts, _energy_and_grad(pts)[0]

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_sphere(n):
    """Generate points based on icosahedron subdivision."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    if n <= 12:
        return verts[:n]
    
    remaining = n - 12
    fib = _fibonacci_sphere(remaining)
    all_pts = np.vstack([verts, fib])
    all_pts /= np.linalg.norm(all_pts, axis=1, keepdims=True)
    return all_pts

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _simulated_annealing(pts0, deadline, n_iter=5000, t_start=0.1, t_end=0.001):
    """
    Simulated annealing to escape local minima.
    """
    pts = pts0.copy()
    e_curr, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_e = e_curr
    
    rng = np.random.RandomState(42)
    
    for it in range(n_iter):
        if time.time() > deadline - 5.0:
            break
        
        t = t_start * (t_end / t_start) ** (it / n_iter)
        
        # Generate random perturbation in tangent space
        delta = rng.randn(N, 3) * 0.01
        for i in range(N):
            radial = np.dot(delta[i], pts[i])
            delta[i] -= radial * pts[i]
        
        new_pts = _exp_map(pts, delta)
        e_new, _ = _energy_and_grad(new_pts)
        
        if e_new < e_curr or rng.rand() < math.exp(-(e_new - e_curr) / t):
            pts = new_pts
            e_curr = e_new
            if e_curr < best_e:
                best_e = e_curr
                best_pts = pts.copy()
    
    return best_pts, best_e

def _optimize_lbfgs(pts0, deadline, maxiter=5000):
    """
    Optimize using L-BFGS-B.
    """
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _energy_and_grad(pts)
        return g.flatten()

    try:
        result = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 10, 
                     'ftol': 1e-15, 'gtol': 1e-10, 'm': 20}
        )
        if result.success or result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _energy_and_grad(best_pts)[0]
            return best_pts, best_energy
        else:
            return pts0, _energy_and_grad(pts0)[0]
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def _fine_polish(pts0, deadline, maxiter=10000):
    """
    High precision polishing using L-BFGS-B with very tight tolerances.
    """
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _energy_and_grad(pts)
        return g.flatten()

    try:
        result = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 10,
                     'ftol': 1e-16, 'gtol': 1e-12, 'm': 10}
        )
        if result.success or result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _energy_and_grad(best_pts)[0]
            return best_pts, best_energy
        else:
            return pts0, _energy_and_grad(pts0)[0]
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    best_pts = None
    best_energy = np.inf

    # 1. Diverse Initializations
    initializations = []
    initializations.append(_fibonacci_sphere(N))
    initializations.append(_icosahedron_sphere(N))
    
    seeds = [42, 123, 777, 999, 2023, 31415, 27182, 16180]
    for s in seeds:
        initializations.append(_random_sphere(N, s))

    # 2. Optimization Loop with Simulated Annealing
    for i, pts0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        # First, do simulated annealing to explore
        if time.time() < deadline - 15.0:
            pts_sa, e_sa = _simulated_annealing(pts0, deadline, n_iter=2000)
            if e_sa < best_energy:
                best_energy = e_sa
                best_pts = pts_sa.copy()
        
        # Then, L-BFGS optimization
        pts_opt, e_opt = _optimize_lbfgs(pts0, deadline, maxiter=3000)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()
        
        # Also try L-BFGS from the SA result
        if time.time() < deadline - 10.0:
            pts_opt2, e_opt2 = _optimize_lbfgs(pts_sa, deadline, maxiter=2000)
            if e_opt2 < best_energy:
                best_energy = e_opt2
                best_pts = pts_opt2.copy()

    if best_pts is None:
        return _fibonacci_sphere(N)

    # 3. Polishing Phase
    time_left = deadline - time.time()
    
    if time_left > 10.0:
        pts_polished, e_polished = _fine_polish(best_pts, deadline, maxiter=5000)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = pts_polished

    # 4. Newton-Raphson Polish
    if time.time() < deadline - 5.0:
        try:
            pts_newton, e_newton = _newton_polish(best_pts, max_iter=30)
            if e_newton < best_energy:
                best_energy = e_newton
                best_pts = pts_newton
        except Exception:
            pass

    # 5. Final Check and Return
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
