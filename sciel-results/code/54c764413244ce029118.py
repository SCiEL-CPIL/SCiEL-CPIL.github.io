"""Rewritten program: explicit direct approach for 100-point log-energy on S^2."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return -1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())


def _log_energy_grad(pts):
    """Compute logarithmic energy and gradient on sphere (vectorized)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)

    # Energy: sum_{i<j} -log(d_ij)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return -1e18, np.zeros_like(pts)
    energy = -np.sum(np.log(d_upper))

    # Gradient computation
    inv_d2 = np.zeros_like(d2)
    mask_non_diag = ~np.eye(n, dtype=bool)
    inv_d2[mask_non_diag] = 1.0 / d2[mask_non_diag]

    grad = -np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)

    # Project gradient onto tangent space
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= radial * pts

    return energy, grad


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _icosahedron_sphere(n):
    """Generate points based on icosahedron symmetry."""
    t = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    pts = []
    for i in range(n):
        v = verts[i % len(verts)]
        noise = np.random.RandomState(i).randn(3) * 0.1
        v = v + noise
        v /= np.linalg.norm(v)
        pts.append(v)
    return np.array(pts)


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _optimize_lbfgs(pts0, deadline, maxiter=500):
    """Optimize using L-BFGS-B with sphere projection."""
    x0 = pts0.flatten()

    def objective(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        e, _ = _log_energy_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        _, g = _log_energy_grad(pts)
        return g.flatten()

    result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'maxfun': maxiter * 4,
                               'ftol': 1e-12, 'gtol': 1e-8})

    if result.success or result.fun < objective(x0):
        best_pts = _project_to_sphere(result.x.reshape(-1, 3))
        best_energy = _log_energy(best_pts)
        return best_pts, best_energy
    else:
        return pts0, _log_energy(pts0)


def _optimize_momentum(pts0, deadline, max_iter=3000):
    """Optimize using gradient descent with momentum and adaptive step size."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)

    v = np.zeros_like(pts)
    lr = 0.005
    momentum = 0.9
    no_improve_count = 0
    best_lr = lr

    for it in range(max_iter):
        if time.time() > deadline:
            break

        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            no_improve_count = 0
            best_lr = lr
        else:
            no_improve_count += 1

        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-13:
            break

        # Adaptive momentum: increase as we converge
        current_momentum = min(0.95, 0.9 + 0.05 * (it / max_iter))

        v = current_momentum * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)

        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            lr = min(lr * 1.03, 0.05)
        else:
            lr *= 0.65
            v *= 0.4
            if no_improve_count > 50:
                lr = best_lr * 0.5
                v = np.zeros_like(pts)
                no_improve_count = 0
            if lr < 1e-12:
                break

    return best_pts, best_energy


def _optimize_l_bfgs_precise(pts0, deadline, maxiter=2000):
    """High-precision L-BFGS-B optimization with tighter tolerances."""
    x0 = pts0.flatten()

    def objective(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        e, _ = _log_energy_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        _, g = _log_energy_grad(pts)
        return g.flatten()

    result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'maxfun': maxiter * 5,
                               'ftol': 1e-15, 'gtol': 1e-10, 'm': 50})

    if result.success or result.fun < objective(x0):
        best_pts = _project_to_sphere(result.x.reshape(-1, 3))
        best_energy = _log_energy(best_pts)
        return best_pts, best_energy
    else:
        return pts0, _log_energy(pts0)


def _polish_high_precision(pts0, deadline, max_iter=5000):
    """Very fine-tuned momentum optimization for final polishing."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)

    v = np.zeros_like(pts)
    lr = 0.001
    momentum = 0.97
    no_improve_count = 0
    best_lr = lr

    for it in range(max_iter):
        if time.time() > deadline:
            break

        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            no_improve_count = 0
            best_lr = lr
        else:
            no_improve_count += 1

        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-14:
            break

        v = momentum * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)

        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            lr = min(lr * 1.01, 0.005)
        else:
            lr *= 0.7
            v *= 0.3
            if no_improve_count > 80:
                lr = best_lr * 0.5
                v = np.zeros_like(pts)
                no_improve_count = 0
            if lr < 1e-14:
                break

    return best_pts, best_energy


def _polish_ultra_fine(pts0, deadline, max_iter=8000):
    """Ultra-fine polishing with very small steps and high momentum."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)

    v = np.zeros_like(pts)
    lr = 0.0003
    momentum = 0.98
    no_improve_count = 0
    best_lr = lr

    for it in range(max_iter):
        if time.time() > deadline:
            break

        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            no_improve_count = 0
            best_lr = lr
        else:
            no_improve_count += 1

        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break

        v = momentum * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)

        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            lr = min(lr * 1.005, 0.002)
        else:
            lr *= 0.75
            v *= 0.2
            if no_improve_count > 100:
                lr = best_lr * 0.5
                v = np.zeros_like(pts)
                no_improve_count = 0
            if lr < 1e-16:
                break

    return best_pts, best_energy


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    best_pts = None
    best_energy = np.inf

    # Multiple diverse starting configurations
    starts = [
        ("fibonacci", _fibonacci_sphere(N)),
        ("random_42", _random_sphere(N, seed=42)),
        ("random_123", _random_sphere(N, seed=123)),
        ("random_999", _random_sphere(N, seed=999)),
        ("random_777", _random_sphere(N, seed=777)),
        ("random_555", _random_sphere(N, seed=555)),
        ("random_314", _random_sphere(N, seed=314)),
        ("icosahedron", _icosahedron_sphere(N)),
    ]

    # Phase 1: Quick L-BFGS-B on each start to find promising local minima
    for name, pts0 in starts:
        if time.time() > deadline - 40:
            break

        pts_opt, e_opt = _optimize_lbfgs(pts0, deadline, maxiter=300)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt

    # Phase 2: Momentum refinement on the best candidate
    if best_pts is not None and time.time() < deadline - 30:
        pts_ref, e_ref = _optimize_momentum(best_pts, deadline, max_iter=2000)
        if e_ref < best_energy:
            best_energy = e_ref
            best_pts = pts_ref

    # Phase 3: High-precision L-BFGS-B
    if best_pts is not None and time.time() < deadline - 20:
        pts_precise, e_precise = _optimize_l_bfgs_precise(best_pts, deadline, maxiter=1500)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = pts_precise

    # Phase 4: High-precision momentum polish
    if best_pts is not None and time.time() < deadline - 10:
        pts_polish, e_polish = _polish_high_precision(best_pts, deadline, max_iter=4000)
        if e_polish < best_energy:
            best_energy = e_polish
            best_pts = pts_polish

    # Phase 5: Ultra-fine polish if time allows
    if best_pts is not None and time.time() < deadline - 5:
        pts_ultra, e_ultra = _polish_ultra_fine(best_pts, deadline, max_iter=6000)
        if e_ultra < best_energy:
            best_energy = e_ultra
            best_pts = pts_ultra

    # Phase 6: Final L-BFGS-B pass with very tight tolerances if time allows
    if best_pts is not None and time.time() < deadline - 3:
        pts_final, e_final = _optimize_l_bfgs_precise(best_pts, deadline, maxiter=800)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final

    # Final safety check: ensure points are valid
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Ensure all points are on unit sphere
    best_pts = _project_to_sphere(best_pts)

    return best_pts
# EVOLVE-BLOCK-END
