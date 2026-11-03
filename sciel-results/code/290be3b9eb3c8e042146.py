"""Improved logarithmic energy minimizer for 100 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its tangent-space gradient for points on the sphere.
    Energy E = sum_{i<j} log(1 / ||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # Ambient gradient: grad_ambient_i = -pts[i] * W[i].sum() + (W @ pts)[i]
    row_sums = W.sum(axis=1, keepdims=True)
    grad_ambient = (W @ pts) - row_sums * pts
    
    # Project to tangent space
    dot = np.sum(grad_ambient * pts, axis=1, keepdims=True)
    grad_tangent = grad_ambient - dot * pts
    
    return energy, grad_tangent

def _fast_energy(pts):
    """Compute energy quickly for sorting and comparison."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))

def _precise_energy(pts):
    """Compute energy using math.fsum for final scoring accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return pts / norms

def _generate_fibonacci(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_sobol(n, seed=42):
    """Generate points on the sphere using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed)
    samples = sampler.random(n)
    samples = samples - 0.5
    return _normalize(samples)

def _generate_halton(n, skip=0):
    """Generate points on the sphere using Halton sequence."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, 3))
    primes = [2, 3, 5]
    for i in range(n):
        idx = i + skip + 1
        for d, p in enumerate(primes):
            points[i, d] = halton(idx, p)
    points -= 0.5
    return _normalize(points)

def _generate_random(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def _generate_jacobi_spherical(n, rng):
    """Generate points using Jacobi polynomial roots for better initial distribution."""
    # Use roots of Jacobi polynomials for latitude, uniform for longitude
    from scipy.special import roots_jacobi
    # a=0, b=0 gives Legendre polynomial roots (uniform on sphere)
    z_vals, _ = roots_jacobi(n, 0, 0)
    z_vals = np.sort(z_vals)
    phi_vals = rng.uniform(0, 2*np.pi, n)
    r_vals = np.sqrt(np.clip(1 - z_vals**2, 0, 1))
    return np.column_stack([r_vals * np.cos(phi_vals), r_vals * np.sin(phi_vals), z_vals])

def _gradient_flow(pts, rng, n_steps, lr_start, lr_end, deadline, noise_scale_max=0.01):
    """
    Perform gradient flow with annealing to explore configuration space.
    Uses tangent-space gradient descent with adaptive learning rate.
    """
    n = pts.shape[0]
    current = pts.copy()
    best = pts.copy()
    best_energy = _fast_energy(pts)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(n_steps - 1, 1)
        lr = lr_start * (lr_end / lr_start) ** progress
        
        e, g = _energy_and_grad(current)
        
        # Add small noise for exploration (annealed)
        noise_scale = noise_scale_max * (1.0 - progress) ** 1.5
        noise = rng.standard_normal((n, 3)) * noise_scale
        
        # Update in ambient space, then project back to sphere
        current = current - lr * g + noise
        current = _normalize(current)
        
        e_new = _fast_energy(current)
        if e_new < best_energy:
            best_energy = e_new
            best = current.copy()
    
    return best, best_energy

def _lbfgs_optimize(pts, deadline, maxiter=10000, ftol=1e-14, gtol=1e-11):
    """Run L-BFGS-B optimization from a given start with given tolerances."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_and_grad(p)
        return e, g.reshape(-1)
    
    x0 = pts.reshape(-1)
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': maxiter,
                'maxfun': maxiter * 5,
                'ftol': ftol,
                'gtol': gtol,
            }
        )
        if result.success or result.fun < _fast_energy(pts):
            new_pts = _normalize(result.x.reshape(n, 3))
            return new_pts
    except Exception:
        pass
    
    return pts

def _optimize_single(pts, deadline):
    """Multi-phase L-BFGS optimization from a given start."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    # Phase 1: Standard L-BFGS-B
    if time.time() < deadline - 5:
        new_pts = _lbfgs_optimize(pts, deadline, maxiter=5000, ftol=1e-14, gtol=1e-11)
        e_new = _precise_energy(new_pts)
        if e_new < best_energy:
            best_pts = new_pts
            best_energy = e_new
    
    # Phase 2: Polish with tighter tolerances
    if time.time() < deadline - 3:
        new_pts = _lbfgs_optimize(best_pts, deadline, maxiter=8000, ftol=1e-16, gtol=1e-13)
        e_new = _precise_energy(new_pts)
        if e_new < best_energy:
            best_pts = new_pts
            best_energy = e_new
    
    return best_pts, best_energy

def _final_polish(pts, deadline):
    """
    Final ultra-precise polish pass with very tight tolerances.
    Uses multiple L-BFGS restarts from slightly perturbed versions.
    """
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    # Try several tiny perturbations and polish each
    rng = np.random.default_rng(12345)
    perturbations = [0.0, 5e-4, 3e-4, 2e-4, 1.5e-4, 1e-4, 7e-5, 5e-5, 3e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
    
    for scale in perturbations:
        if time.time() > deadline:
            break
        
        if scale == 0.0:
            start = best_pts
        else:
            start = _normalize(best_pts + rng.standard_normal((n, 3)) * scale)
        
        new_pts = _lbfgs_optimize(start, deadline, maxiter=30000, ftol=1e-18, gtol=1e-15)
        e_new = _precise_energy(new_pts)
        if e_new < best_energy:
            best_pts = new_pts
            best_energy = e_new
    
    # One more final pass on the best
    if time.time() < deadline - 2.0:
        new_pts = _lbfgs_optimize(best_pts, deadline, maxiter=50000, ftol=1e-18, gtol=1e-15)
        e_new = _precise_energy(new_pts)
        if e_new < best_energy:
            best_pts = new_pts
            best_energy = e_new
    
    return best_pts, best_energy

def _simulated_annealing(pts, rng, deadline, n_iter=2000):
    """
    Simulated annealing to escape local minima.
    Proposes random perturbations and accepts with probability based on energy change.
    """
    n = pts.shape[0]
    current = pts.copy()
    best = pts.copy()
    best_energy = _fast_energy(pts)
    
    T_start = 0.1
    T_end = 0.001
    n_steps = n_iter
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(n_steps - 1, 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Propose a random perturbation
        scale = 0.05 * (1.0 - progress) + 0.005
        proposal = _normalize(current + rng.standard_normal((n, 3)) * scale)
        
        e_current = _fast_energy(current)
        e_proposal = _fast_energy(proposal)
        delta_e = e_proposal - e_current
        
        # Accept if better or with probability exp(-delta_e / T)
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current = proposal
        
        if e_proposal < best_energy:
            best_energy = e_proposal
            best = proposal.copy()
    
    return best, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Spiral (multiple offsets)
    for offset in [0.5, 0.0, 0.25, 0.75, 0.1, 0.9]:
        initializations.append(_generate_fibonacci(N, offset=offset))
    
    # 2. Sobol Sequences - better uniform coverage
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        initializations.append(_generate_sobol(N, seed=seed))
    
    # 3. Halton sequences with different skips
    for skip in [0, 10, 50, 100, 500, 1000]:
        initializations.append(_generate_halton(N, skip=skip))
    
    # 4. Random Points
    for _ in range(10):
        initializations.append(_generate_random(N, rng))
        
    # 5. Perturbed Fibonacci
    base = _generate_fibonacci(N)
    for scale in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))
    
    # 6. More structured initializations - golden ratio with different offsets
    for offset in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        i = np.arange(N) + 0.5 + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        initializations.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 7. Jacobi polynomial roots
    for _ in range(3):
        initializations.append(_generate_jacobi_spherical(N, rng))
    
    # Sort initializations by quick energy to prioritize good starts
    scored_inits = []
    for init in initializations:
        e_quick = _fast_energy(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase A: Gradient flow exploration on top initializations
    flow_deadline = start_time + 25.0
    for e_quick, init_pts in scored_inits[:20]:
        if time.time() > flow_deadline:
            break
        
        flowed_pts, e_precise = _gradient_flow(init_pts, rng, 800, 0.05, 0.0003, flow_deadline)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = flowed_pts
    
    # Phase B: L-BFGS optimization on all initializations (including flowed ones)
    if best_pts is not None:
        scored_inits.insert(0, (best_energy, best_pts))
    
    lbfgs_deadline = deadline - 40.0  # Reserve 40s for final polish and simulated annealing
    for e_quick, init_pts in scored_inits[:30]:
        if time.time() > lbfgs_deadline:
            break
            
        optimized_pts, e_precise = _optimize_single(init_pts, lbfgs_deadline)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized_pts
    
    if best_pts is None:
        best_pts = _generate_fibonacci(N)
    
    # Phase C: Simulated annealing to potentially escape local minima
    sa_deadline = deadline - 35.0
    if time.time() < sa_deadline:
        sa_pts, e_sa = _simulated_annealing(best_pts, rng, sa_deadline, n_iter=1500)
        e_sa_precise = _precise_energy(sa_pts)
        if e_sa_precise < best_energy:
            best_energy = e_sa_precise
            best_pts = sa_pts
    
    # Phase D: Final ultra-precise polish
    polish_deadline = deadline - 3.0  # 3s safety margin
    best_pts, best_energy = _final_polish(best_pts, polish_deadline)
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _generate_fibonacci(N)
        best_pts = _normalize(best_pts)
        
    return best_pts

# EVOLVE-BLOCK-END
