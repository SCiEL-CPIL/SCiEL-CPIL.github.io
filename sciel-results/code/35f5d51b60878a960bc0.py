"""Optimize logarithmic energy of 150 points on the unit sphere using Riemannian optimization."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient on the sphere.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|xi-xj|
    
    Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Compute pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # inv_dist_sq[i,j] = 1/|xi-xj|^2, with diagonal = 0
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # Gradient: grad[i] = sum_j (pts[i] - pts[j]) / |pts[i]-pts[j]|^2
    # = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    # Use upper triangle only
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    # Filter out any very small distances (shouldn't happen)
    mask = d > 1e-15
    d = d[mask]
    if len(d) == 0:
        energy = 0.0
    else:
        energy = -np.sum(np.log(d))
    
    return energy, grad


def _energy_only(pts):
    """Compute just the energy (for speed during optimization)."""
    energy, _ = _energy_and_grad(pts)
    return energy


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_sphere(n, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    pts = rng.randn(n, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _local_minimize(pts, time_limit=10.0, tol=1e-10):
    """Local minimization using L-BFGS on the sphere."""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    start_time = time.time()
    
    x0 = pts.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def gradient(x):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': tol, 'gtol': 1e-8, 'maxfun': 10000}
        )
        best_pts = _project_to_sphere(result.x.reshape(n, 3))
    except Exception:
        best_pts = _project_to_sphere(pts.copy())
    
    return best_pts


def _basin_hopping_single(pts, time_limit=30.0, n_trials=50, step_size=0.2):
    """Basin hopping: local minimize, perturb, repeat."""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    start_time = time.time()
    
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    rng = np.random.RandomState(12345)
    
    for trial in range(n_trials):
        if time.time() - start_time > time_limit:
            break
        
        # Local minimize current point
        local_pts = _local_minimize(pts, time_limit=min(2.0, time_limit - (time.time() - start_time)))
        local_energy = _energy_only(local_pts)
        
        if local_energy < best_energy:
            best_energy = local_energy
            best_pts = local_pts.copy()
        
        # Perturb
        perturb = rng.randn(n, 3) * step_size
        pts = _project_to_sphere(best_pts + perturb)
    
    return best_pts


def _simulated_annealing(pts, time_limit=20.0, t_start=0.5, t_end=0.01, n_steps=2000):
    """Simulated annealing on the sphere."""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    start_time = time.time()
    
    current_energy = _energy_only(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    rng = np.random.RandomState(67890)
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Linear cooling schedule
        progress = step / n_steps
        T = t_start + (t_end - t_start) * progress
        
        # Generate perturbation
        scale = 0.05 + 0.1 * (1.0 - progress)
        perturb = rng.randn(n, 3) * scale
        candidate = _project_to_sphere(pts + perturb)
        
        candidate_energy = _energy_only(candidate)
        delta = candidate_energy - current_energy
        
        if delta < 0 or rng.rand() < math.exp(-delta / max(T, 1e-10)):
            pts = candidate
            current_energy = candidate_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
    
    return best_pts


def _parallel_tempering(initial_pts, time_limit=60.0, n_replicas=5):
    """
    Parallel Tempering (Replica Exchange) optimization.
    Uses multiple replicas at different temperatures to escape local minima.
    """
    n = len(initial_pts)
    start_time = time.time()
    
    # Initialize replicas
    temps = [0.0, 0.03, 0.08, 0.15, 0.3]
    if n_replicas > len(temps):
        extra = [0.3 * (1 + i) for i in range(n_replicas - len(temps))]
        temps.extend(extra)
    temps = temps[:n_replicas]
    
    replicas = []
    for i in range(n_replicas):
        if i == 0:
            replicas.append(_project_to_sphere(initial_pts.copy()))
        else:
            rng = np.random.RandomState(42 + i)
            perturb = rng.randn(n, 3) * 0.15
            replicas.append(_project_to_sphere(initial_pts + perturb))
            
    energies = [_energy_only(r) for r in replicas]
    best_pts = replicas[0].copy()
    best_energy = min(energies)
    
    base_perturb_scale = 0.08
    
    steps = 0
    while time.time() - start_time < time_limit:
        # 1. Perform a Metropolis step for each replica
        new_energies = energies[:]
        for i in range(n_replicas):
            T = temps[i]
            scale = base_perturb_scale * (1.0 + 0.5 * T) 
            rng = np.random.RandomState(steps * 100 + i)
            perturb = rng.randn(n, 3) * scale
            candidate = _project_to_sphere(replicas[i] + perturb)
            
            new_e = _energy_only(candidate)
            delta_e = new_e - energies[i]
            
            if T == 0.0:
                if new_e < energies[i]:
                    replicas[i] = candidate
                    energies[i] = new_e
                    new_energies[i] = new_e
            else:
                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                    replicas[i] = candidate
                    energies[i] = new_e
                    new_energies[i] = new_e
        
        # 2. Attempt swaps between adjacent replicas
        for i in range(n_replicas - 1):
            T1 = temps[i]
            T2 = temps[i+1]
            E1 = new_energies[i]
            E2 = new_energies[i+1]
            
            if T1 == 0.0 or T2 == 0.0:
                T1_eff = 1e-6
                T2_eff = T2 if T2 > 0 else 1e-6
            else:
                T1_eff = T1
                T2_eff = T2
                
            delta_swap = (E2 - E1) * (1.0/T2_eff - 1.0/T1_eff)
            
            if delta_swap < 0 or np.random.rand() < np.exp(-delta_swap):
                replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
                energies[i], energies[i+1] = energies[i+1], energies[i]
                new_energies[i], new_energies[i+1] = new_energies[i+1], new_energies[i]
        
        # Update best
        for i in range(n_replicas):
            if energies[i] < best_energy:
                best_energy = energies[i]
                best_pts = replicas[i].copy()
        
        steps += 1
        
        if steps % 50 == 0 and time.time() - start_time > time_limit:
            break

    return best_pts


def _final_polish(pts, time_limit=15.0):
    """Final high-precision polish with very tight tolerance."""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    
    start_time = time.time()
    prev_energy = _energy_only(pts)
    
    x0 = pts.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def gradient(x):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 20000}
        )
        polished = _project_to_sphere(result.x.reshape(n, 3))
    except Exception:
        polished = pts
    
    return polished


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_pts = None
    best_energy = float('inf')
    
    def try_candidate(pts):
        nonlocal best_pts, best_energy
        if time.time() - start_time > deadline - 5.0:
            return
        pts = _project_to_sphere(pts)
        if not np.all(np.isfinite(pts)):
            return
        # Check min distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 1e-9:
            return
        e = _energy_only(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()

    # Strategy:
    # 1. Generate good initial guesses
    # 2. Run multiple optimization methods in parallel (sequentially but with time budgets)
    # 3. Polish the best result
    
    initializations = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=42),
        _random_sphere(N, seed=123),
        _random_sphere(N, seed=456)
    ]
    
    # Allocate time:
    # 4 initializations.
    # Give each ~ 25s for parallel tempering. Total 100s.
    # Remaining 40s for polishing and basin hopping.
    
    pt_time_budget = 25.0
    
    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > deadline - 30.0:
            break
            
        remaining_for_pt = min(pt_time_budget, deadline - 30.0 - (time.time() - start_time))
        if remaining_for_pt < 5.0:
            continue
            
        try:
            # Run Parallel Tempering
            pt_result = _parallel_tempering(init_pts, time_limit=remaining_for_pt, n_replicas=5)
            try_candidate(pt_result)
            
            # Quick local polish on the PT result
            if time.time() - start_time < deadline - 20.0:
                polish_t = min(3.0, deadline - 20.0 - (time.time() - start_time))
                if polish_t > 1.0:
                    polished = _local_minimize(pt_result, time_limit=polish_t)
                    try_candidate(polished)
        except Exception:
            pass

    # Basin hopping on the best found so far
    if best_pts is not None and time.time() - start_time < deadline - 15.0:
        bh_time = min(15.0, deadline - 15.0 - (time.time() - start_time))
        if bh_time > 5.0:
            try:
                bh_result = _basin_hopping_single(best_pts, time_limit=bh_time, n_trials=30, step_size=0.15)
                try_candidate(bh_result)
            except Exception:
                pass

    # Simulated annealing on the best found so far
    if best_pts is not None and time.time() - start_time < deadline - 10.0:
        sa_time = min(10.0, deadline - 10.0 - (time.time() - start_time))
        if sa_time > 3.0:
            try:
                sa_result = _simulated_annealing(best_pts, time_limit=sa_time, t_start=0.3, t_end=0.01, n_steps=1500)
                try_candidate(sa_result)
            except Exception:
                pass

    # Final Global Polish on the absolute best found
    if best_pts is not None and time.time() - start_time < deadline - 5.0:
        polish_time = min(10.0, deadline - 5.0 - (time.time() - start_time))
        if polish_time > 1.0:
            try:
                final_polished = _final_polish(best_pts, time_limit=polish_time)
                try_candidate(final_polished)
            except Exception:
                pass

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    best_pts = _project_to_sphere(best_pts)
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
        best_pts = _project_to_sphere(best_pts)
        
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    if np.min(dist) < 1e-9:
        best_pts = _fibonacci_sphere(N)
        best_pts = _project_to_sphere(best_pts)

    return best_pts
# EVOLVE-BLOCK-END
