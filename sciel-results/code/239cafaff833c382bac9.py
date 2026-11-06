"""Optimized 100-point log-energy minimization on S^2 with multi-phase strategy."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0


def _log_energy(pts):
    """Compute logarithmic energy using math.fsum for accuracy."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-9) or not np.all(np.isfinite(d_upper)):
        return 1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())


def _log_energy_grad(pts):
    """Compute energy and gradient on sphere, avoiding sqrt where possible."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d2_upper = d2[mask]
    if np.any(d2_upper < 1e-24) or not np.all(np.isfinite(d2_upper)):
        return 1e18, np.zeros_like(pts)
    
    # Energy: sum log(1/d) = -0.5 * sum log(d^2)
    energy = -0.5 * math.fsum(np.log(d2_upper).tolist())
    
    # Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # = -sum_{j!=i} diff[i,j] / d2[i,j]
    mask_full = ~np.eye(n, dtype=bool)
    inv_d2 = np.zeros_like(d2)
    inv_d2[mask_full] = 1.0 / d2[mask_full]
    
    grad = -np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    # Project to tangent space: grad -= (grad . x_i) * x_i
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= radial * pts
    
    return energy, grad


def _project(pts):
    """Project to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, offset=0.5):
    """Fibonacci spiral initialization."""
    golden = 0.5 * (1 + np.sqrt(5))
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * golden * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Random points on sphere."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _so3_random_rotation(seed=None):
    """Generate a random SO(3) rotation matrix."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    # Use QR decomposition of a random matrix
    A = rng.randn(3, 3)
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = 1)
    d = np.sign(np.linalg.det(Q))
    if d == -1:
        Q[:, -1] *= -1
    return Q


def _optimize_lbfgs(pts0, deadline, maxiter=5000, ftol=1e-16, gtol=1e-14):
    """L-BFGS optimization with sphere projection."""
    x0 = pts0.flatten()
    
    def obj(x):
        pts = _project(x.reshape(-1, 3))
        e, _ = _log_energy_grad(pts)
        return e
    
    def grad(x):
        pts = _project(x.reshape(-1, 3))
        _, g = _log_energy_grad(pts)
        return g.flatten()
    
    try:
        result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'maxfun': maxiter * 10,
                                  'ftol': ftol, 'gtol': gtol, 'm': 100})
        if result.fun < 1e17:
            pts_opt = _project(result.x.reshape(-1, 3))
            return pts_opt, _log_energy(pts_opt)
    except Exception:
        pass
    return pts0, _log_energy(pts0)


def _gradient_descent(pts0, deadline, max_iter=50000):
    """Gradient descent with Nesterov momentum and adaptive step."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_e = _log_energy(pts)
    v = np.zeros_like(pts)
    lr = 0.003
    momentum = 0.9
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Nesterov lookahead
        pts_look = _project(pts + momentum * v)
        e, grad = _log_energy_grad(pts_look)
        
        if e < best_e - 1e-16:
            best_e = e
            best_pts = pts_look.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-16:
            break
        
        v = momentum * v - lr * grad
        pts_new = _project(pts + v)
        e_new = _log_energy(pts_new)
        
        if e_new < e:
            pts = pts_new
            if no_improve < 10:
                lr = min(lr * 1.02, 0.01)
        else:
            lr *= 0.75
            v *= 0.5
            if no_improve > 100:
                v = np.zeros_like(pts)
                lr = 0.003
                no_improve = 0
            if lr < 1e-18:
                break
    
    return best_pts, best_e


def _simulated_annealing(pts0, deadline, max_iter=20000):
    """Simulated annealing for escaping local minima."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_e = _log_energy(pts)
    
    T = 0.05  # Initial temperature
    T_min = 1e-8
    cooling = 0.9995
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e = _log_energy(pts)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
        
        # Random perturbation
        rng = np.random
        noise_scale = T * 0.1
        pts_perturbed = _project(pts + rng.randn(N, 3) * noise_scale)
        e_perturbed = _log_energy(pts_perturbed)
        
        delta_e = e_perturbed - e
        if delta_e < 0 or rng.rand() < math.exp(-delta_e / max(T, T_min)):
            pts = pts_perturbed
        
        T *= cooling
    
    return best_pts, best_e


def construct():
    """Construct 100 points minimizing log-energy on unit sphere."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_pts = None
    best_energy = np.inf
    
    # Phase 1: Generate diverse initial configurations
    starts = []
    # Fibonacci with various offsets
    for off in np.linspace(0.0, 0.9, 10):
        starts.append(_fibonacci_sphere(N, offset=off))
    
    # Random starts with different seeds
    for seed in range(30):
        starts.append(_random_sphere(N, seed=seed))
    
    # Rotated Fibonacci starts
    for seed in range(10):
        base = _fibonacci_sphere(N, offset=0.5)
        R = _so3_random_rotation(seed=seed)
        starts.append(_project(base @ R.T))
    
    # Phase 2: Quick optimization on all starts
    candidates = []
    for i, pts0 in enumerate(starts):
        if time.time() > deadline - 80:
            break
        pts_opt, e_opt = _optimize_lbfgs(pts0, deadline - 80, maxiter=500, ftol=1e-12, gtol=1e-8)
        candidates.append((e_opt, pts_opt))
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    
    candidates.sort(key=lambda x: x[0])
    
    # Phase 3: Deep optimization on top candidates
    for e_c, pts_c in candidates[:10]:
        if time.time() > deadline - 50:
            break
        pts_deep, e_deep = _optimize_lbfgs(pts_c, deadline - 50, maxiter=5000, ftol=1e-15, gtol=1e-12)
        if e_deep < best_energy:
            best_energy = e_deep
            best_pts = pts_deep
    
    # Phase 4: Gradient descent refinement with Nesterov momentum
    if best_pts is not None and time.time() < deadline - 30:
        pts_gd, e_gd = _gradient_descent(best_pts, deadline - 30, max_iter=30000)
        if e_gd < best_energy:
            best_energy = e_gd
            best_pts = pts_gd
    
    # Phase 5: Simulated annealing to escape local minima
    if best_pts is not None and time.time() < deadline - 20:
        pts_sa, e_sa = _simulated_annealing(best_pts, deadline - 20, max_iter=15000)
        if e_sa < best_energy:
            best_energy = e_sa
            best_pts = pts_sa
        # Follow up with L-BFGS polish
        if time.time() < deadline - 10:
            pts_polish, e_polish = _optimize_lbfgs(best_pts, deadline - 10, maxiter=3000, ftol=1e-15, gtol=1e-13)
            if e_polish < best_energy:
                best_energy = e_polish
                best_pts = pts_polish
    
    # Phase 6: Final high-precision polish
    if best_pts is not None and time.time() < deadline - 5:
        pts_final, e_final = _optimize_lbfgs(best_pts, deadline - 2, maxiter=10000, ftol=1e-17, gtol=1e-15)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final
    
    # Ensure validity
    if best_pts is None:
        best_pts = _fibonacci_sphere(N, offset=0.5)
    
    best_pts = _project(best_pts)
    
    # Check for near-duplicates
    min_d = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            min_d = min(min_d, d)
    
    if min_d < 1e-9:
        rng = np.random.RandomState(999)
        best_pts = _project(best_pts + rng.randn(N, 3) * 1e-5)
        if time.time() < deadline - 1:
            best_pts, _ = _optimize_lbfgs(best_pts, deadline - 0.5, maxiter=1000)
    
    return best_pts
