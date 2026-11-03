"""Logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_START = time.time()
_DEADLINE = _START + 150.0
_SAFETY = 12.0

# Precompute upper-triangle indices for N=100
_IU = np.triu_indices(N, k=1)


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _energy_fast(pts):
    """Fast logarithmic energy (chordal distances) using dot products."""
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    return float(-np.sum(np.log(np.sqrt(dist_sq[_IU]))))


def _energy_grad(pts):
    """Energy and tangent-space gradient using dot-product distances."""
    pts = _normalize(pts)
    n = pts.shape[0]
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)

    energy = float(-np.sum(np.log(dist[_IU])))

    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)

    row_sums = W.sum(axis=1, keepdims=True)
    grad = (W @ pts) - row_sums * pts

    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts

    return energy, grad


def _precise_energy(pts):
    """Energy via math.fsum for final scoring accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _valid(pts):
    if pts.shape != (N, 3) or not np.all(np.isfinite(pts)):
        return False
    if np.any(np.linalg.norm(pts, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', pts, pts)[:, None] \
        + np.einsum('ij,ij->i', pts, pts)[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(d2, np.inf)
    return bool(np.min(d2) >= (1e-9) ** 2)


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _sobol_sphere(n, seed):
    try:
        sampler = qmc.Sobol(d=3, seed=seed)
        samples = sampler.random(n) - 0.5
        return _normalize(samples)
    except Exception:
        rng = np.random.default_rng(seed)
        return _normalize(rng.standard_normal((n, 3)))


def _spectral_init(n, rng, l_max=8):
    """Smooth initial configuration via low-frequency spherical perturbation."""
    theta = rng.uniform(0, 2 * np.pi, n)
    cos_phi = 2.0 * rng.uniform(0, 1, n) - 1.0
    sin_phi = np.sqrt(1.0 - cos_phi ** 2)
    pts = np.column_stack([sin_phi * np.cos(theta),
                           sin_phi * np.sin(theta),
                           cos_phi])
    for l in range(1, l_max + 1):
        for m in range(-l, l + 1):
            amp = rng.normal(0, 0.01 / (l + 1))
            phase = rng.uniform(0, 2 * np.pi)
            t = np.arctan2(pts[:, 1], pts[:, 0])
            c = pts[:, 2]
            mod = amp * np.cos(m * t + phase) * (c ** abs(m))
            pts[:, 2] += mod * 0.1
            pts[:, 0] += amp * 0.05 * np.sin(m * t + phase) * sin_phi
    return _normalize(pts)


def _exp_map(pts, tangent):
    """Exponential map on the unit sphere."""
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    mask_small = norms < 1e-8
    theta = norms_safe
    v_norm = tangent / norms_safe
    new_pts = pts * np.cos(theta) + v_norm * np.sin(theta)
    if np.any(mask_small):
        new_pts[mask_small] = pts[mask_small] + tangent[mask_small]
    return _normalize(new_pts)


def _riemannian_polish(pts, deadline, max_iter=2500, lr_init=1e-3):
    """Riemannian gradient descent with backtracking line search."""
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)
    lr = lr_init
    stall_count = 0
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        e, g = _energy_grad(best_pts)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-13:
            break
        alpha = lr
        accepted = False
        for _ in range(40):
            new_pts = _exp_map(best_pts, -alpha * g)
            new_e = _energy_fast(new_pts)
            if new_e < e - 1e-4 * alpha * g_norm * g_norm:
                accepted = True
                break
            alpha *= 0.5
        if accepted:
            new_pts = _exp_map(best_pts, -alpha * g)
            new_e_precise = _precise_energy(new_pts)
            if new_e_precise < best_energy:
                best_pts = new_pts
                best_energy = new_e_precise
                lr = min(lr * 1.05, 0.05)
                stall_count = 0
            else:
                lr *= 0.9
                stall_count += 1
        else:
            lr *= 0.5
            stall_count += 1
        if stall_count > 50:
            break
    return best_pts, best_energy


def _lbfgs(pts, deadline, maxiter, ftol, gtol, maxls=30):
    """Single L-BFGS-B run with normalization inside the objective."""
    n = pts.shape[0]

    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_grad(p)
        return e, g.reshape(-1)

    try:
        result = minimize(objective_and_grad, pts.reshape(-1),
                          method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'maxfun': maxiter * 5,
                                   'ftol': ftol, 'gtol': gtol, 'maxls': maxls})
        return _normalize(result.x.reshape(n, 3))
    except Exception:
        return pts


def _optimize_from_start(pts, deadline):
    """Multi-phase L-BFGS-B + Riemannian polish from one start."""
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)

    def _try(new_pts):
        nonlocal best_pts, best_energy
        if not _valid(new_pts):
            return
        e_p = _precise_energy(new_pts)
        if e_p < best_energy:
            best_pts = new_pts
            best_energy = e_p

    if time.time() < deadline - 3.0:
        _try(_lbfgs(best_pts, deadline, 300, 1e-10, 1e-6, 20))
    if time.time() < deadline - 3.0:
        _try(_lbfgs(best_pts, deadline, 500, 1e-13, 1e-9, 30))
    if time.time() < deadline - 10.0:
        try:
            r_pts, r_e = _riemannian_polish(best_pts, deadline, max_iter=1000, lr_init=5e-4)
            if r_e < best_energy:
                best_pts = r_pts
                best_energy = r_e
        except Exception:
            pass
    if time.time() < deadline - 5.0:
        _try(_lbfgs(best_pts, deadline, 600, 1e-15, 1e-11, 40))

    return best_pts, best_energy


def _final_polish(pts, deadline):
    """High-precision polish: tiny perturbations, coordinate descent, and tight L-BFGS."""
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)
    rng = np.random.default_rng(98765)

    def _try(new_pts):
        nonlocal best_pts, best_energy
        if not _valid(new_pts):
            return
        e_p = _precise_energy(new_pts)
        if e_p < best_energy:
            best_pts = new_pts
            best_energy = e_p

    # Phase 1: Coordinate-descent style polish (optimize each point fixing others)
    if time.time() < deadline - 5.0:
        for round_num in range(3):
            if time.time() > deadline - 5.0:
                break
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                # Fix all points except i, optimize point i
                fixed = best_pts.copy()
                # Use L-BFGS-B on just point i (3 variables)
                def obj_i(flat_i):
                    p_i = flat_i.reshape(3)
                    p_i = p_i / np.linalg.norm(p_i)
                    temp = fixed.copy()
                    temp[i] = p_i
                    e = _energy_fast(temp)
                    # Gradient w.r.t. p_i in tangent space
                    dot_products = temp @ temp.T
                    dist_sq = 2.0 - 2.0 * dot_products
                    np.fill_diagonal(dist_sq, 1.0)
                    dist_sq = np.maximum(dist_sq, 1e-16)
                    dist = np.sqrt(dist_sq)
                    W = 1.0 / dist_sq
                    np.fill_diagonal(W, 0.0)
                    row_sums = W.sum(axis=1, keepdims=True)
                    grad_all = (W @ temp) - row_sums * temp
                    grad_i = grad_all[i]
                    dot = np.sum(grad_i * p_i)
                    grad_i = grad_i - dot * p_i
                    return e, grad_i.flatten()
                try:
                    res = minimize(obj_i, best_pts[i].flatten(), method='L-BFGS-B',
                                   jac=True, options={'maxiter': 100, 'ftol': 1e-15,
                                                       'gtol': 1e-12, 'maxls': 30})
                    new_pt = res.x / np.linalg.norm(res.x)
                    new_pts = best_pts.copy()
                    new_pts[i] = new_pt
                    _try(new_pts)
                except Exception:
                    pass

    # Phase 2: Tiny perturbations with restarts
    scales = [0.0, 3e-4, 2e-4, 1e-4, 5e-5, 3e-5, 1e-5, 5e-6]
    for scale in scales:
        if time.time() > deadline - 3.0:
            break
        n_restarts = 1 if scale == 0.0 else 2
        for _ in range(n_restarts):
            if time.time() > deadline - 3.0:
                break
            if scale == 0.0:
                start = best_pts
            else:
                start = _normalize(best_pts + rng.standard_normal((n, 3)) * scale)
            _try(_lbfgs(start, deadline, 200, 1e-14, 1e-10, 30))

    # Phase 3: Final ultra-tight L-BFGS-B
    if time.time() < deadline - 2.0:
        _try(_lbfgs(best_pts, deadline, 1000, 1e-16, 1e-13, 50))

    # Phase 4: One more coordinate-descent pass on the final result
    if time.time() < deadline - 1.0:
        for i in range(n):
            if time.time() > deadline - 1.0:
                break
            fixed = best_pts.copy()
            def obj_i2(flat_i):
                p_i = flat_i.reshape(3)
                p_i = p_i / np.linalg.norm(p_i)
                temp = fixed.copy()
                temp[i] = p_i
                e = _energy_fast(temp)
                dot_products = temp @ temp.T
                dist_sq = 2.0 - 2.0 * dot_products
                np.fill_diagonal(dist_sq, 1.0)
                dist_sq = np.maximum(dist_sq, 1e-16)
                dist = np.sqrt(dist_sq)
                W = 1.0 / dist_sq
                np.fill_diagonal(W, 0.0)
                row_sums = W.sum(axis=1, keepdims=True)
                grad_all = (W @ temp) - row_sums * temp
                grad_i = grad_all[i]
                dot = np.sum(grad_i * p_i)
                grad_i = grad_i - dot * p_i
                return e, grad_i.flatten()
            try:
                res = minimize(obj_i2, best_pts[i].flatten(), method='L-BFGS-B',
                               jac=True, options={'maxiter': 50, 'ftol': 1e-15,
                                                   'gtol': 1e-13, 'maxls': 20})
                new_pt = res.x / np.linalg.norm(res.x)
                new_pts = best_pts.copy()
                new_pts[i] = new_pt
                _try(new_pts)
            except Exception:
                pass

    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + _SAFETY  # absolute guard (== _DEADLINE - _SAFETY)

    rng = np.random.default_rng(12345)

    initializations = []
    # Structured: Fibonacci (multiple offsets)
    for offset in [0.5, 0.0, 0.15, 0.3, 0.7]:
        initializations.append(_fibonacci_spiral(N, offset))
    # Sobol structured
    for seed in range(4):
        initializations.append(_sobol_sphere(N, seed * 13 + 3))
    # Smooth spectral
    for seed in range(4):
        initializations.append(_spectral_init(N, np.random.default_rng(seed * 100 + 7), l_max=6))
    # Random
    for i in range(6):
        initializations.append(_random_sphere(N, rng))
    # Perturbed Fibonacci
    base = _fibonacci_spiral(N)
    for scale in [0.005, 0.01, 0.02, 0.05]:
        initializations.append(_normalize(base + rng.standard_normal((N, 3)) * scale))

    # Sort by quick energy to prioritize good starts
    scored = sorted(((_energy_fast(init), init) for init in initializations),
                    key=lambda x: x[0])

    best_pts = None
    best_energy = float('inf')

    for _, init_pts in scored:
        if time.time() > deadline - 25.0:  # reserve time for final polish
            break
        opt_pts, e_precise = _optimize_from_start(init_pts, deadline - 25.0)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = opt_pts

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)

    # Final high-precision polish on the best configuration
    try:
        final_pts, final_e = _final_polish(best_pts, deadline)
        if final_e < best_energy:
            best_pts = final_pts
            best_energy = final_e
    except Exception:
        pass

    best_pts = _normalize(best_pts)
    if not _valid(best_pts) or not np.isfinite(_precise_energy(best_pts)):
        best_pts = _normalize(_fibonacci_spiral(N))

    return best_pts
