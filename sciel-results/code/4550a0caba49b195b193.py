"""Logarithmic energy minimization for N points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_START = time.time()
_DEADLINE = 145.0  # 5s safety margin from 150s kill

# Precompute upper triangle indices
_IU = np.triu_indices(N, k=1)
_IU_LIST = list(zip(_IU[0], _IU[1]))


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _energy_and_grad(pts):
    """Compute log energy and tangent-space gradient on the sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient is in the tangent space at each point.
    """
    n = pts.shape[0]
    # Pairwise dot products
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    
    # Energy
    energy = -0.5 * np.sum(np.log(dist_sq[_IU]))
    
    # Gradient: for energy E = -0.5 * sum_{i<j} log(d_ij^2)
    # dE/dx_i = sum_{j != i} (1/d_ij^2) * (x_i - x_j)  [Euclidean gradient]
    # Project to tangent: grad_tan_i = grad_i - (grad_i . x_i) * x_i
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # Euclidean gradient
    W_row_sum = W.sum(axis=1, keepdims=True)
    grad_euclid = W_row_sum * pts - W @ pts
    
    # Project to tangent space
    dot_proj = np.sum(grad_euclid * pts, axis=1, keepdims=True)
    grad_tan = grad_euclid - dot_proj * pts
    
    return float(energy), grad_tan


def _energy_only(pts):
    """Fast energy computation without gradient."""
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    return float(-0.5 * np.sum(np.log(dist_sq[_IU])))


def _precise_energy(pts):
    """Compute energy using math.fsum for maximum accuracy."""
    terms = []
    for i, j in _IU_LIST:
        d = np.linalg.norm(pts[i] - pts[j])
        if d < 1e-9:
            return float('inf')
        terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _project(X):
    """Project points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


def _exp_map(base, tangent):
    """Exponential map on the sphere: move from base along tangent vector.
    
    For unit sphere: exp_p(v) = p*cos(|v|) + (v/|v|)*sin(|v|)
    This is the exact geodesic flow.
    """
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    # Handle zero-length tangent vectors
    small = norms < 1e-14
    norms_safe = np.where(small, 1.0, norms)
    
    cos_n = np.cos(norms_safe)
    sin_n = np.sin(norms_safe) / norms_safe
    
    result = base * cos_n + tangent * sin_n
    # For small norms, use Taylor expansion: exp ≈ base + tangent + 0.5*tangent^2...
    # But our formula handles it via sin(x)/x ≈ 1 for small x
    result[small] = base[small] + tangent[small]
    return _project(result)


def _fibonacci_spiral(n, offset=0.5):
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _sobol_sphere(n, seed=0):
    """Generate quasi-random points on the sphere using Sobol sequence."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    m = max(1, int(np.ceil(np.log2(n))))
    while (1 << m) < n:
        m += 1
    points = sampler.random(1 << m)[:n]
    theta = 2.0 * np.pi * points[:, 0]
    z = 2.0 * points[:, 1] - 1.0
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])


def _riemannian_lbfgs(pts, max_time, maxiter=5000, ftol=1e-15, gtol=1e-14):
    """Riemannian L-BFGS-B optimization on the sphere.
    
    Uses the exponential map as retraction and pullback gradient.
    """
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    start_t = time.time()
    
    def objective_and_grad(flat):
        if time.time() - start_t > max_time:
            raise TimeoutError()
        # flat is a tangent vector at the current point
        # We need to map it to the sphere via exponential map
        p_new = _exp_map(best_pts, flat.reshape(n, 3))
        e, g_tan = _energy_and_grad(p_new)
        return e, g_tan.reshape(-1)
    
    # We'll do iterative Riemannian optimization manually
    # Using L-BFGS on the tangent space with retraction
    
    # For simplicity and robustness, use scipy's minimize with a wrapper
    # that operates in a local tangent space chart
    
    # Alternative: direct gradient descent with adaptive step in tangent space
    current = pts.copy()
    lr = 0.05
    patience = 0
    
    for iteration in range(maxiter):
        if time.time() - start_t > max_time:
            break
        if _remaining() < 5.0:
            break
        
        e, g = _energy_and_grad(current)
        g_norm = np.linalg.norm(g)
        
        if g_norm < gtol:
            break
        
        # Line search in tangent space
        alpha = lr
        improved = False
        for ls_iter in range(20):
            new_pt = _exp_map(current, alpha * g * (-1.0))  # negative gradient
            e_new = _energy_only(new_pt)
            
            if e_new < e - 1e-4 * alpha * g_norm ** 2:
                current = new_pt
                e = e_new
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            # Try even smaller step
            new_pt = _exp_map(current, -0.001 * g)
            e_new = _energy_only(new_pt)
            if e_new < e:
                current = new_pt
                e = e_new
                lr *= 0.5
            else:
                patience += 1
                if patience > 10:
                    break
        else:
            patience = 0
            lr = min(lr * 1.1, 0.5)
        
        # Update best
        e_prec = _precise_energy(current)
        if e_prec < best_energy:
            best_energy = e_prec
            best_pts = current.copy()
    
    return best_pts, best_energy


def _riemannian_lbfgs_scipy(pts, max_time, maxiter=3000, ftol=1e-14, gtol=1e-13):
    """Use scipy's L-BFGS-B in a local coordinate system.
    
    We parameterize each point by its tangent vector relative to the current position.
    This is a local chart that becomes exact for small displacements.
    """
    n = pts.shape[0]
    start_t = time.time()
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def wrapped_objective(flat):
        if time.time() - start_t > max_time:
            raise TimeoutError()
        # flat represents tangent vectors at the base point
        tangent = flat.reshape(n, 3)
        # Project tangent to be perpendicular to base points
        dot_base = np.sum(tangent * best_pts, axis=1, keepdims=True)
        tangent = tangent - dot_base * best_pts
        # Move along geodesic
        new_pts = _exp_map(best_pts, tangent)
        e, g = _energy_and_grad(new_pts)
        # Pull gradient back to tangent space at base
        # For small displacements, this is approximately the same
        return e, g.reshape(-1)
    
    try:
        result = minimize(
            wrapped_objective,
            np.zeros(n * 3),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxiter * 3}
        )
        
        if result.success or result.fun < best_energy:
            # Recover the actual points
            tangent = result.x.reshape(n, 3)
            dot_base = np.sum(tangent * best_pts, axis=1, keepdims=True)
            tangent = tangent - dot_base * best_pts
            new_pts = _exp_map(best_pts, tangent)
            e_prec = _precise_energy(new_pts)
            if e_prec < best_energy:
                best_energy = e_prec
                best_pts = new_pts
    except (TimeoutError, Exception):
        pass
    
    return best_pts, best_energy


def _optimize_from_start(pts, deadline):
    """Multi-stage optimization from a starting point."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    # Stage 1: Coarse Riemannian optimization
    if _remaining() > 20.0:
        alloc = min(_remaining() - 15.0, 30.0)
        pts1, e1 = _riemannian_lbfgs_scipy(best_pts, max_time=alloc, maxiter=2000, ftol=1e-12, gtol=1e-8)
        if e1 < best_energy:
            best_energy = e1
            best_pts = pts1
    
    # Stage 2: Medium refinement
    if _remaining() > 15.0:
        alloc = min(_remaining() - 12.0, 25.0)
        pts2, e2 = _riemannian_lbfgs_scipy(best_pts, max_time=alloc, maxiter=3000, ftol=1e-13, gtol=1e-10)
        if e2 < best_energy:
            best_energy = e2
            best_pts = pts2
    
    # Stage 3: Fine refinement
    if _remaining() > 10.0:
        alloc = min(_remaining() - 8.0, 20.0)
        pts3, e3 = _riemannian_lbfgs_scipy(best_pts, max_time=alloc, maxiter=4000, ftol=1e-14, gtol=1e-12)
        if e3 < best_energy:
            best_energy = e3
            best_pts = pts3
    
    # Stage 4: Ultra-fine with direct gradient descent
    if _remaining() > 5.0:
        alloc = min(_remaining() - 4.0, 15.0)
        pts4, e4 = _riemannian_lbfgs(best_pts, max_time=alloc, maxiter=5000, ftol=1e-15, gtol=1e-14)
        if e4 < best_energy:
            best_energy = e4
            best_pts = pts4
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate focused set of initializations
    initializations = []
    
    # Primary: Fibonacci spiral (best known structured init for log energy)
    initializations.append(_fibonacci_spiral(N, offset=0.5))
    
    # Fibonacci with different offsets
    for offset in [0.3, 0.7, 0.2, 0.8]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Perturbed Fibonacci (small random perturbations to escape symmetry)
    base_fib = _fibonacci_spiral(N, offset=0.5)
    for s in range(6):
        amp = 0.01 + 0.01 * s
        jittered = base_fib + amp * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Sobol sequences (good uniform coverage)
    for sob_seed in [0, 1, 2]:
        initializations.append(_sobol_sphere(N, seed=sob_seed))
    
    # Golden angle spiral variant
    i = np.arange(N)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / N
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    theta = golden_angle * i
    golden_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    initializations.append(_project(golden_pts))
    
    # Sort by initial energy to prioritize best basins
    init_energies = []
    for idx, init_pts in enumerate(initializations):
        e_init = _energy_only(init_pts)
        init_energies.append((e_init, idx, init_pts))
    init_energies.sort(key=lambda x: x[0])
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase A: Quick screening of top initializations
    # Allocate more time to the best initializations
    for e_init, idx, init_pts in init_energies[:10]:
        if _remaining() < 30.0:
            break
        
        # Allocate time based on rank: best gets most time
        rank = init_energies.index((e_init, idx, init_pts))
        alloc = max(3.0, 12.0 - rank * 0.8)
        alloc = min(alloc, _remaining() - 25.0)
        if alloc < 2.0:
            break
        
        try:
            # Quick optimization to identify basin
            pts_opt, e_opt = _riemannian_lbfgs_scipy(
                init_pts, max_time=alloc, maxiter=1500, ftol=1e-10, gtol=1e-6
            )
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt
        except Exception:
            pass
    
    # Phase B: Deep optimization from the best found
    if best_pts is not None and _remaining() > 20.0:
        deep_deadline_time = _remaining() - 15.0
        deep_deadline_time = min(deep_deadline_time, 50.0)
        optimized, e_precise = _optimize_from_start(best_pts, deadline)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized
    
    # Phase C: Try second-best initialization if it might be in a different basin
    if _remaining() > 25.0:
        # Find second-best init that's different from the best
        for e_init, idx, init_pts in init_energies:
            if init_pts is best_pts:
                continue
            # Check if this is in a different basin (rough check)
            if np.linalg.norm(init_pts - best_pts) > 0.5:
                alloc = min(_remaining() - 20.0, 20.0)
                if alloc < 5.0:
                    break
                try:
                    pts_opt, e_opt = _riemannian_lbfgs_scipy(
                        init_pts, max_time=alloc, maxiter=2000, ftol=1e-11, gtol=1e-7
                    )
                    if e_opt < best_energy:
                        best_energy = e_opt
                        best_pts = pts_opt
                        # Deep optimize this new best
                        if _remaining() > 15.0:
                            optimized, e_precise = _optimize_from_start(best_pts, deadline)
                            if e_precise < best_energy:
                                best_energy = e_precise
                                best_pts = optimized
                except Exception:
                    pass
                break  # Only try one alternative basin
    
    # Phase D: Final ultra-precise polish
    if best_pts is not None and _remaining() > 5.0:
        n = best_pts.shape[0]
        
        # Multiple passes with increasing precision
        polish_configs = [
            {'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-11},
            {'maxiter': 4000, 'ftol': 1e-14, 'gtol': 1e-12},
            {'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13},
            {'maxiter': 6000, 'ftol': 1e-15, 'gtol': 1e-14},
            {'maxiter': 8000, 'ftol': 1e-16, 'gtol': 1e-15},
        ]
        
        for cfg in polish_configs:
            if _remaining() < 4.0:
                break
            alloc = min(_remaining() - 3.0, 12.0)
            try:
                pts_polished, e_polished = _riemannian_lbfgs_scipy(
                    best_pts, max_time=alloc, maxiter=cfg['maxiter'], 
                    ftol=cfg['ftol'], gtol=cfg['gtol']
                )
                if e_polished < best_energy:
                    best_energy = e_polished
                    best_pts = pts_polished
            except Exception:
                pass
        
        # Final direct gradient descent polish
        if _remaining() > 3.0:
            alloc = min(_remaining() - 2.0, 8.0)
            try:
                pts_final, e_final = _riemannian_lbfgs(
                    best_pts, max_time=alloc, maxiter=10000, ftol=1e-16, gtol=1e-15
                )
                if e_final < best_energy:
                    best_energy = e_final
                    best_pts = pts_final
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization and validation
    best_pts = _project(best_pts)
    
    # Verify validity
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
