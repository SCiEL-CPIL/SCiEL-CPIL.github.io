"""Optimized logarithmic energy minimization for 100 points on the sphere."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import scipy.optimize as so

_DEADLINE = 142.0  # seconds; safety margin before 150s kill

def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return 1e18
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_fast(pts):
    """Vectorized energy computation."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-18):
        return 1e18
    d = np.sqrt(d_sq)
    return float(np.sum(np.log(1.0 / d)))

def _energy_grad(pts):
    """Compute energy and ambient gradient.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    dE/dx_i = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n,n)
    np.fill_diagonal(dist_sq, 1.0)
    
    dist_sq_inv = 1.0 / dist_sq
    grad = -np.sum(diff * dist_sq_inv[:, :, np.newaxis], axis=1)  # (n,3)
    
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-18):
        return 1e18, np.zeros_like(pts)
    E = float(np.sum(np.log(1.0 / np.sqrt(d_sq))))
    
    return E, grad

def _tangent_project(pts, vec):
    """Project vector onto tangent space at each point."""
    dot = np.sum(vec * pts, axis=1, keepdims=True)
    return vec - dot * pts

def _retract(pts, v, alpha=1.0):
    """Retraction on the sphere."""
    new_pts = pts + alpha * v
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return new_pts / norms

def _fibonacci_start():
    """Golden-angle Fibonacci spiral on the sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(pts)

def _random_start(seed):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _project(pts)

def _jittered_fibonacci(jitter, seed):
    """Fibonacci with random jitter."""
    pts = _fibonacci_start()
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((N, 3)) * jitter
    noise = _tangent_project(pts, noise)
    return _retract(pts, noise, 1.0)

def _project(pts):
    """Project onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _lbfgs_optimize(pts, max_iter=5000, tol=1e-12):
    """L-BFGS-B optimization with Riemannian gradient."""
    pts = pts.copy()
    
    def objective(x_flat):
        x = x_flat.reshape(N, 3)
        E, _ = _energy_grad(x)
        return E
    
    def gradient(x_flat):
        x = x_flat.reshape(N, 3)
        _, grad = _energy_grad(x)
        # Project gradient to tangent space
        grad_tan = _tangent_project(x, grad)
        return grad_tan.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = so.minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol,
                                    'maxls': 30, 'maxfun': max_iter * 3})
        opt_pts = result.x.reshape(N, 3)
        opt_pts = _project(opt_pts)
        opt_E = _energy_fast(opt_pts)
        return opt_pts, opt_E
    except Exception:
        return pts, _energy_fast(pts)

def _cg_optimize(pts, max_iter=5000, tol=1e-12):
    """Conjugate gradient on the sphere with line search."""
    pts = pts.copy()
    E, grad = _energy_grad(pts)
    g_tan = _tangent_project(pts, grad)
    s = -g_tan
    g_old = g_tan.copy()
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        # Backtracking line search
        alpha = 0.1
        best_alpha = None
        best_E = E
        directional_deriv = np.sum(g_tan * s)
        
        for _ in range(60):
            new_pts = _retract(pts, s, alpha)
            new_E = _energy_fast(new_pts)
            if new_E < best_E:
                best_E = new_E
                best_alpha = alpha
            # Armijo condition
            if new_E < E + 1e-4 * alpha * directional_deriv:
                break
            alpha *= 0.5
        
        if best_alpha is None:
            break
        
        pts = _retract(pts, s, best_alpha)
        E = best_E
        
        E_new, grad_new = _energy_grad(pts)
        g_new = _tangent_project(pts, grad_new)
        
        if np.linalg.norm(g_new) < tol:
            break
        
        # Polak-Ribiere beta
        dot_new_old = np.sum(g_new * (g_new - g_old))
        dot_old_old = np.sum(g_old * g_old)
        if dot_old_old > 1e-15:
            beta = max(0.0, dot_new_old / dot_old_old)
        else:
            beta = 0.0
        
        s = -g_new + beta * s
        g_old = g_new
    
    return pts, E

def _differential_evolution_global(pts_init, max_iter=50, pop_size=20, tol=1e-10):
    """Differential evolution in tangent space for global exploration.
    
    Uses a simplified DE approach: maintain a population of configurations,
    mutate using DE/rand/1/bin, and select based on energy.
    """
    rng = np.random.default_rng(777)
    
    # Initialize population around the initial configuration
    population = [pts_init.copy()]
    for _ in range(pop_size - 1):
        noise = rng.standard_normal((N, 3)) * 0.3
        noise = _tangent_project(pts_init, noise)
        population.append(_retract(pts_init, noise, 1.0))
    
    energies = [_energy_fast(p) for p in population]
    best_idx = int(np.argmin(energies))
    best_pts = population[best_idx].copy()
    best_E = energies[best_idx]
    
    F = 0.8  # scale factor
    CR = 0.9  # crossover rate
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 15:
            break
        
        new_population = []
        new_energies = []
        
        for i in range(pop_size):
            # DE/rand/1/bin mutation
            r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
            while r1 == i or r2 == i or r3 == i:
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
            
            # Mutant vector in tangent space
            mutant = population[r1] + F * (population[r2] - population[r3])
            # Project back to sphere
            mutant = _project(mutant)
            
            # Crossover
            trial = population[i].copy()
            for j in range(N):
                if rng.random() < CR:
                    trial[j] = mutant[j]
            trial = _project(trial)
            
            trial_E = _energy_fast(trial)
            
            # Selection
            if trial_E < energies[i]:
                new_population.append(trial)
                new_energies.append(trial_E)
            else:
                new_population.append(population[i])
                new_energies.append(energies[i])
        
        population = new_population
        energies = new_energies
        
        new_best_idx = int(np.argmin(energies))
        if energies[new_best_idx] < best_E:
            best_E = energies[new_best_idx]
            best_pts = population[new_best_idx].copy()
        
        # Check convergence
        if np.std(energies) < 1e-10:
            break
    
    return best_pts, best_E

def _simulated_annealing(pts, max_iter=2000, T_start=0.1, T_end=1e-6):
    """Simulated annealing on the sphere."""
    rng = np.random.default_rng(42)
    pts = pts.copy()
    E = _energy_fast(pts)
    best_pts = pts.copy()
    best_E = E
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        # Random perturbation in tangent space
        step_size = 0.01 * (1.0 + T * 10)
        noise = rng.standard_normal((N, 3)) * step_size
        noise = _tangent_project(pts, noise)
        new_pts = _retract(pts, noise, 1.0)
        new_E = _energy_fast(new_pts)
        
        delta_E = new_E - E
        if delta_E < 0 or rng.random() < math.exp(-delta_E / max(T, 1e-10)):
            pts = new_pts
            E = new_E
            if E < best_E:
                best_E = E
                best_pts = pts.copy()
    
    return best_pts, best_E

def _local_polish(pts, max_iter=10000, tol=1e-14):
    """High-precision local polish using L-BFGS-B with very tight tolerance."""
    return _lbfgs_optimize(pts, max_iter=max_iter, tol=tol)

def _multi_start_optimize(initializations, time_budget=100.0):
    """Optimize from multiple starting points, allocating time adaptively."""
    best_pts = None
    best_E = 1e18
    start_time = time.time()
    
    for i, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 10:
            break
        
        # Allocate time based on remaining budget
        n_remaining = len(initializations) - i
        time_per_init = min(remaining / max(1, n_remaining), 30.0)
        
        if time_per_init < 5:
            break
        
        # Phase 1: CG optimization
        opt_pts, E = _cg_optimize(init_pts, max_iter=3000, tol=1e-12)
        if E < best_E:
            best_E = E
            best_pts = opt_pts
        
        # Phase 2: L-BFGS polish if time permits
        if time.time() - start_time < time_budget - 15:
            polished_pts, polished_E = _lbfgs_optimize(best_pts, max_iter=2000, tol=1e-13)
            if polished_E < best_E:
                best_E = polished_E
                best_pts = polished_pts
    
    return best_pts, best_E

def construct():
    start_time = time.time()
    
    best_pts = None
    best_E = 1e18
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_start())
    
    # 2. Random starts
    for seed in [42, 123, 456, 789, 1000, 2024, 31415, 27182, 16180, 82718]:
        initializations.append(_random_start(seed))
    
    # 3. Jittered Fibonacci
    for jitter in [0.01, 0.05, 0.1, 0.2, 0.3]:
        initializations.append(_jittered_fibonacci(jitter, seed=100 + int(jitter * 1000)))
    
    # Phase 1: Multi-start optimization
    best_pts, best_E = _multi_start_optimize(initializations, time_budget=80.0)
    
    # Phase 2: Differential evolution for global exploration
    if best_pts is not None and time.time() < _DEADLINE - 30:
        de_pts, de_E = _differential_evolution_global(best_pts, max_iter=30, pop_size=15)
        if de_E < best_E:
            best_E = de_E
            best_pts = de_pts
        # Polish the DE result
        if time.time() < _DEADLINE - 15:
            polished_pts, polished_E = _lbfgs_optimize(de_pts, max_iter=3000, tol=1e-13)
            if polished_E < best_E:
                best_E = polished_E
                best_pts = polished_pts
    
    # Phase 3: Simulated annealing to escape local minima
    if best_pts is not None and time.time() < _DEADLINE - 20:
        sa_pts, sa_E = _simulated_annealing(best_pts, max_iter=1000, T_start=0.05, T_end=1e-7)
        if sa_E < best_E:
            best_E = sa_E
            best_pts = sa_pts
        # Polish SA result
        if time.time() < _DEADLINE - 10:
            polished_pts, polished_E = _lbfgs_optimize(sa_pts, max_iter=2000, tol=1e-13)
            if polished_E < best_E:
                best_E = polished_E
                best_pts = polished_pts
    
    # Phase 4: Final high-precision polish
    if best_pts is not None and time.time() < _DEADLINE - 5:
        final_pts, final_E = _local_polish(best_pts, max_iter=5000, tol=1e-14)
        if final_E < best_E:
            best_E = final_E
            best_pts = final_pts
    
    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_start()
    
    return _project(best_pts)
# EVOLVE-BLOCK-END
