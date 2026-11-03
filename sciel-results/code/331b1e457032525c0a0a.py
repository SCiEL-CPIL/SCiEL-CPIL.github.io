"""Logarithmic energy minimization for N points on the unit sphere.
Combines multi-start BFGS optimization with Sobol/Fibonacci initializations,
gradient-flow annealing for broader exploration, spherical-harmonic potential
field learning, and ultra-precise final polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.special import lpmv

N = 100
_DEADLINE = 145.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx
    grad = -raw_grad
    
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence."""
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _optimize(pts, deadline, max_iter=500, ftol=1e-12, gtol=1e-10, method='L-BFGS-B'):
    """Run L-BFGS-B optimization from a starting point with projection."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_grad(p)
        return e, g.reshape(-1)
    
    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method=method,
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
                'maxfun': max_iter * 10,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _anneal(pts, deadline, n_steps=200, base_lr=0.01, min_lr=0.001):
    """Gradient-flow annealing with adaptive learning rate and noise schedule."""
    current = pts.copy()
    n = current.shape[0]
    best_e = _energy(current)
    best_current = current.copy()
    
    for step in range(n_steps):
        if time.time() > deadline - 5.0:
            break
        
        # Cosine annealing schedule with restarts
        progress = step / max(n_steps - 1, 1)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        lr = max(lr, min_lr)
        
        e, g = _energy_grad(current)
        
        # Adaptive noise: larger early, smaller late
        noise_scale = 0.002 * (1 - progress) + 0.0001
        perturbation = noise_scale * np.random.standard_normal((n, 3))
        step_vec = -lr * (g + perturbation)
        
        # Project step to tangent space
        dot = np.sum(step_vec * current, axis=1, keepdims=True)
        step_vec = step_vec - dot * current
        
        current = current + step_vec
        current = _project(current)
        
        # Track best
        e_new = _energy(current)
        if e_new < best_e:
            best_e = e_new
            best_current = current.copy()
    
    return best_current


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def _spherical_harmonics(theta, phi, l_max=8):
    """Compute real spherical harmonics Y_l^m for 0<=l<=l_max, -l<=m<=l.
    Returns array of shape (n_points, (l_max+1)^2).
    theta: colatitude (0..pi), phi: azimuth (0..2pi).
    """
    n = theta.shape[0]
    coeffs = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            abs_m = abs(m)
            # Associated Legendre P_l^m(cos(theta))
            ct = np.cos(theta)
            pt = lpmv(abs_m, l, ct)
            # Normalization: sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!)
            norm = math.sqrt((2 * l + 1) / (4.0 * math.pi) *
                             math.factorial(l - abs_m) / math.factorial(l + abs_m))
            if m == 0:
                val = norm * pt
            elif m > 0:
                val = math.sqrt(2.0) * norm * pt * np.cos(m * phi)
            else:
                val = math.sqrt(2.0) * norm * pt * np.sin(abs_m * phi)
            coeffs.append(val)
    return np.column_stack(coeffs)


def _potential_energy_from_coeffs(theta, phi, coeffs, l_max=8):
    """Evaluate potential and its gradient w.r.t. theta, phi from harmonic coefficients.
    V(theta,phi) = sum_k c_k * Y_k(theta,phi)
    Returns (V, dV_dtheta, dV_dphi) each of shape (n,).
    """
    n = theta.shape[0]
    Y = _spherical_harmonics(theta, phi, l_max)
    V = Y @ coeffs
    # Numerical gradient for simplicity (we only need this for finding minima)
    eps = 1e-6
    V_tp = _spherical_harmonics(theta + eps, phi, l_max) @ coeffs
    V_tm = _spherical_harmonics(theta - eps, phi, l_max) @ coeffs
    V_fp = _spherical_harmonics(theta, phi + eps, l_max) @ coeffs
    V_fm = _spherical_harmonics(theta, phi - eps, l_max) @ coeffs
    dV_dt = (V_tp - V_tm) / (2 * eps)
    dV_dp = (V_fp - V_fm) / (2 * eps)
    return V, dV_dt, dV_dp


def _learn_potential_field(rng, deadline, l_max=6, n_iters=800, lr_init=0.05, seed_pts=None):
    """Learn spherical-harmonic potential whose minima give low-energy point sets.
    
    Strategy:
    1. Start with initial point estimates (from seed_pts or random)
    2. Fit a potential that is low at those points and high elsewhere
    3. Find the 100 lowest minima of the potential
    4. Compute pairwise log-energy of minima
    5. Update coefficients to lower the energy
    
    This is a co-evolutionary approach: points define where minima should be,
    coefficients define the potential, and we iterate.
    """
    # Precompute basis functions on a grid for finding minima
    theta_grid = np.linspace(0.01, math.pi - 0.01, 50)
    phi_grid = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    theta_g, phi_g = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    theta_flat = theta_g.ravel()
    phi_flat = phi_g.ravel()
    Y_grid = _spherical_harmonics(theta_flat, phi_flat, l_max)
    n_basis = Y_grid.shape[1]
    
    if seed_pts is not None:
        # Extract theta, phi from 3D points
        pts = seed_pts.copy()
        pts = _project(pts)
        z = pts[:, 2]
        r_xy = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        r_xy = np.maximum(r_xy, 1e-12)
        theta_pts = np.arccos(np.clip(z, -1, 1))
        phi_pts = np.arctan2(pts[:, 1], pts[:, 0])
        phi_pts = np.mod(phi_pts, 2 * math.pi)
    else:
        theta_pts = np.arccos(1 - 2 * rng.random(N))
        phi_pts = 2 * math.pi * rng.random(N)
    
    Y_pts = _spherical_harmonics(theta_pts, phi_pts, l_max)
    
    # Initialize coefficients: small random
    coeffs = rng.standard_normal(n_basis) * 0.1
    
    best_energy = float('inf')
    best_theta = theta_pts.copy()
    best_phi = phi_pts.copy()
    
    lr = lr_init
    
    for it in range(n_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Find 100 lowest minima on the grid
        V_grid = Y_grid @ coeffs
        # Get indices of lowest 150 grid values (candidates)
        n_candidates = min(150, len(V_grid))
        candidate_idx = np.argsort(V_grid)[:n_candidates]
        cand_theta = theta_flat[candidate_idx].copy()
        cand_phi = phi_flat[candidate_idx].copy()
        
        # Refine candidates via local gradient descent
        refined_theta = cand_theta.copy()
        refined_phi = cand_phi.copy()
        for _ in range(10):
            V_c, dV_dt, dV_dp = _potential_energy_from_coeffs(
                refined_theta, refined_phi, coeffs, l_max)
            step_t = -0.01 * dV_dt
            step_p = -0.01 * dV_dp
            refined_theta = np.clip(refined_theta + step_t, 0.01, math.pi - 0.01)
            refined_phi = np.mod(refined_phi + step_p, 2 * math.pi)
        
        # Select 100 best refined candidates (lowest potential)
        V_refined = _potential_energy_from_coeffs(refined_theta, refined_phi, coeffs, l_max)[0]
        top_idx = np.argsort(V_refined)[:N]
        new_theta = refined_theta[top_idx].copy()
        new_phi = refined_phi[top_idx].copy()
        
        # Check separation (vectorized)
        min_sep = np.inf
        for i in range(0, N, 10):
            dt = np.abs(new_theta[i] - new_theta[i+1:i+11])
            dp = np.abs(new_phi[i] - new_phi[i+1:i+11])
            dp = np.minimum(dp, 2*math.pi - dp)
            sep = np.sqrt(dt**2 + dp**2)
            min_sep = min(min_sep, np.min(sep))
        
        if min_sep < 0.01:
            # Too close, add random kick
            coeffs += 0.05 * rng.standard_normal(n_basis)
            continue
        
        # Convert to 3D and compute actual energy
        z_new = np.cos(new_theta)
        r_new = np.sin(new_theta)
        x_new = r_new * np.cos(new_phi)
        y_new = r_new * np.sin(new_phi)
        pts_new = np.column_stack([x_new, y_new, z_new])
        pts_new = _project(pts_new)
        
        e = _energy(pts_new)
        
        if e < best_energy:
            best_energy = e
            best_theta = new_theta.copy()
            best_phi = new_phi.copy()
        
        # Update coefficients: want low potential at best points, high elsewhere
        Y_best = _spherical_harmonics(best_theta, best_phi, l_max)
        # Gradient: reduce potential at points, increase at grid
        target = -0.3 * Y_best.mean(axis=0) + 0.05 * Y_grid.mean(axis=0)
        
        t_norm = np.linalg.norm(target)
        if t_norm > 1e-12:
            target /= t_norm
        
        coeffs = coeffs - lr * target
        
        # Add small noise for exploration
        coeffs += 0.01 * rng.standard_normal(n_basis)
        
        # Decay learning rate
        lr *= 0.99
        
        # Update Y_pts periodically
        if it % 15 == 0:
            Y_pts = _spherical_harmonics(best_theta, best_phi, l_max)
    
    # Convert final best to 3D
    z_f = np.cos(best_theta)
    r_f = np.sin(best_theta)
    x_f = r_f * np.cos(best_phi)
    y_f = r_f * np.sin(best_phi)
    pts_f = np.column_stack([x_f, y_f, z_f])
    pts_f = _project(pts_f)
    
    # Validate
    if not _valid(pts_f):
        return _fibonacci_spiral(N, offset=0.5)
    
    return pts_f


def _newton_polish(pts, deadline, max_iter=50, tol=1e-14):
    """Newton-like local refinement using Hessian approximation."""
    current = pts.copy()
    n = current.shape[0]
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Approximate Hessian via finite differences in tangent space
        # Use a simpler momentum-based approach instead
        lr = 0.1 / (1 + it * 0.1)
        step_vec = -lr * g
        dot = np.sum(step_vec * current, axis=1, keepdims=True)
        step_vec = step_vec - dot * current
        
        current = current + step_vec
        current = _project(current)
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy)
    for s in range(15):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(8):
        jitter_scale = 0.05 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random points with different seeds
    for s in range(15):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    best_pts = None
    best_energy = float('inf')
    
    # Score all initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 25.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Phase 1: Optimize top candidates with BFGS
    top_candidates = scored_inits[:25]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 35.0:
            break
        
        optimized = _optimize(init_pts, deadline, max_iter=800, ftol=1e-14, gtol=1e-11)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Phase 1.5: Spherical harmonic potential field learning
    if time.time() < deadline - 25.0:
        # Try potential field with different seeds and l_max values
        for l_max in [4, 6, 8]:
            if time.time() > deadline - 20.0:
                break
            for seed_offset in range(3):
                if time.time() > deadline - 18.0:
                    break
                sub_deadline = min(deadline - 5.0, time.time() + 8.0)
                seed_pts = None
                if best_pts is not None and seed_offset == 0:
                    seed_pts = best_pts
                elif seed_offset == 1:
                    seed_pts = scored_inits[0][1] if scored_inits else None
                else:
                    seed_pts = _random_sphere(np.random.default_rng(999 + seed_offset))
                
                ph_pts = _learn_potential_field(
                    np.random.default_rng(500 + seed_offset),
                    sub_deadline, l_max=l_max, n_iters=400,
                    lr_init=0.03, seed_pts=seed_pts)
                
                if _valid(ph_pts):
                    # Polish the potential-field result
                    polished = _optimize(ph_pts, deadline, max_iter=1000, ftol=1e-14, gtol=1e-12)
                    e_precise = _precise_energy(polished)
                    if e_precise < best_energy:
                        best_energy = e_precise
                        best_pts = polished.copy()
    
    # Phase 2: Annealing + optimization on the best configuration
    if best_pts is not None and time.time() < deadline - 12.0:
        annealed = _anneal(best_pts, deadline, n_steps=150, base_lr=0.01, min_lr=0.0003)
        refined = _optimize(annealed, deadline, max_iter=1200, ftol=1e-14, gtol=1e-12)
        
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined.copy()
    
    # Phase 3: Final ultra-precise polish with multiple strategies
    if best_pts is not None and time.time() < deadline - 3.0:
        # Try multiple tiny perturbations around the best
        for attempt in range(5):
            if time.time() > deadline - 2.0:
                break
            perturbed = best_pts + 0.0003 * rng.standard_normal((N, 3))
            perturbed = _project(perturbed)
            polished = _optimize(perturbed, deadline, max_iter=2000, ftol=1e-16, gtol=1e-14)
            
            e_precise = _precise_energy(polished)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = polished.copy()
        
        # Newton-like polish
        if time.time() < deadline - 1.5:
            polished = _newton_polish(best_pts, deadline, max_iter=30, tol=1e-15)
            e_precise = _precise_energy(polished)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = polished.copy()
    
    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 1.0:
        refined = _optimize(best_pts, deadline, max_iter=3000, ftol=1e-16, gtol=1e-15)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
