"""Dzugutov cluster global minima search with Langevin dynamics self-assembly."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
_TOTAL_BUDGET = 310.0  # 330s limit with 20s safety margin
_HARD_LIMIT = 325.0
_START_TIME = None
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]
_INSTANCE_RESULTS = {}

A_DZ = 5.82
B_DZ = 1.28
a_DZ = 1.87
b_DZ = 1.94
c_DZ = 1.10
d_DZ = 0.27
m_DZ = 16


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return _TOTAL_BUDGET - _elapsed()


def _hard_remaining():
    return _HARD_LIMIT - _elapsed()


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy using vectorized operations."""
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        diffs = positions[i+1:] - positions[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        mask_a = dists < a_DZ
        mask_b = (dists >= a_DZ) & (dists < b_DZ)
        
        if np.any(mask_a):
            da = dists[mask_a]
            da = np.maximum(da, 1e-10)
            v_a = A_DZ * (da ** (-m_DZ) - B_DZ) * np.exp(c_DZ / (da - a_DZ))
            total += np.sum(v_a)
        
        if np.any(mask_b):
            db = dists[mask_b]
            db = np.maximum(db, 1e-10)
            v_b = B_DZ * np.exp(d_DZ / (db - b_DZ))
            total += np.sum(v_b)
    
    return total


def _dzugutov_gradient(positions):
    """Compute gradient of Dzugutov potential energy."""
    try:
        return dzugutov_energy_gradient(positions)
    except Exception:
        # Finite difference fallback
        n = len(positions)
        grad = np.zeros_like(positions)
        eps = 1e-5
        E0 = _dzugutov_energy(positions)
        for i in range(n):
            for j in range(3):
                pos_p = positions.copy()
                pos_p[i, j] += eps
                E_p = _dzugutov_energy(pos_p)
                grad[i, j] = (E_p - E0) / eps
        return grad


def _lbgfgs_polish(positions, time_limit=5.0):
    """Polish a local minimum using L-BFGS-B with gradient information."""
    start_time = time.time()
    try:
        flat = positions.flatten()
        
        def objective(x):
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            return _dzugutov_energy(x.reshape(-1, 3))
        
        def gradient(x):
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            return _dzugutov_gradient(x.reshape(-1, 3)).flatten()
        
        result = minimize(
            objective,
            flat,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        
        if result.success or result.fun < _dzugutov_energy(positions):
            polished = result.x.reshape(-1, 3)
            # Ensure minimum distance constraint
            valid = True
            for i in range(len(polished)):
                for j in range(i+1, len(polished)):
                    if np.linalg.norm(polished[i] - polished[j]) < 0.5:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return polished
    except Exception:
        pass
    return positions


def _langevin_step(positions, velocities, dt, temperature, rng):
    """
    Perform one Langevin dynamics step.
    Uses velocity Verlet-like integration with friction and random forces.
    """
    n = len(positions)
    grad = _dzugutov_gradient(positions)
    
    # Friction coefficient (overdamped limit)
    gamma = 0.5
    
    # Update velocities: v_new = (v - grad*dt/m) * exp(-gamma*dt) + sqrt(2*gamma*kT*dt/m) * noise
    # For unit mass:
    noise = rng.normal(0, np.sqrt(2 * gamma * temperature * dt), positions.shape)
    velocities = (velocities - grad * dt) * np.exp(-gamma * dt) + noise
    
    # Update positions
    positions = positions + velocities * dt
    
    return positions, velocities


def _langevin_anneal(positions, rng, time_limit=10.0, t_start=2.0, t_end=0.01):
    """
    Anneal positions using Langevin dynamics with cooling schedule.
    """
    start_time = time.time()
    current = positions.copy()
    n = len(current)
    
    # Initial velocities from Maxwell-Boltzmann distribution
    velocities = rng.normal(0, np.sqrt(t_start), current.shape)
    
    # Time-stepping parameters
    dt = 0.01
    max_steps = int(time_limit / (dt * 5))  # estimate
    
    # Annealing schedule
    n_steps = max_steps
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Linear cooling
        frac = step / max(1, n_steps - 1)
        temperature = t_start * (t_end / t_start) ** frac
        
        current, velocities = _langevin_step(current, velocities, dt, temperature, rng)
        
        # Ensure minimum distance by pushing apart overlapping atoms
        for i in range(n):
            for j in range(i+1, n):
                d = current[i] - current[j]
                dist = np.linalg.norm(d)
                if dist < 0.5 and dist > 1e-10:
                    push = d / dist * (0.5 - dist) * 0.5
                    current[i] += push
                    current[j] -= push
    
    return current


def _generate_random_compact(n_atoms, spacing=1.1, rng=None):
    """Generate a compact random cluster with minimum distance constraint."""
    if rng is None:
        rng = np.random.default_rng(123)
    
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.9
    points = []
    min_dist = 0.52
    
    # Start with a central atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    attempts = 0
    max_attempts = 20000
    while len(points) < n_atoms and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        if valid:
            points.append(p)
    
    # If we couldn't place all atoms, fill remaining randomly
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n_atoms])


def _self_assemble_langevin(n_atoms, rng, time_limit=8.0):
    """
    Physics-driven self-assembly using Langevin dynamics.
    Initialize atoms at random positions and use Langevin dynamics with
    annealing to let the structure naturally evolve into a low-energy
    polytetrahedral configuration.
    """
    start_time = time.time()
    
    # Generate initial random compact configuration
    initial = _generate_random_compact(n_atoms, spacing=1.1, rng=rng)
    
    # Run Langevin annealing
    annealed = _langevin_anneal(
        initial, rng,
        time_limit=time_limit * 0.6,
        t_start=3.0,
        t_end=0.005
    )
    
    # Relax to local minimum
    try:
        energy, relaxed = relax(annealed, 'dzugutov', None)
    except Exception:
        relaxed = annealed
        energy = _dzugutov_energy(relaxed)
    
    # Polish with L-BFGS-B
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 2.0:
        polished = _lbgfgs_polish(relaxed, time_limit=remaining_time * 0.8)
        if _dzugutov_energy(polished) < energy:
            relaxed = polished
            energy = _dzugutov_energy(polished)
    
    return relaxed[:n_atoms]


def _basin_hopping(positions, rng, max_iter=50, step_size=0.3, time_limit=10.0):
    """
    Basin hopping: perturb, relax, polish, accept if lower energy.
    """
    start_time = time.time()
    current = positions.copy()
    current_energy = _dzugutov_energy(current)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb
        perturbed = current + rng.normal(0, step_size, current.shape)
        
        # Relax
        try:
            e_new, relaxed = relax(perturbed, 'dzugutov', None)
        except Exception:
            continue
        
        # Polish with L-BFGS-B if we have time
        remaining = time_limit - (time.time() - start_time)
        if remaining > 3.0:
            polished = _lbgfgs_polish(relaxed, time_limit=min(3.0, remaining * 0.5))
            if _dzugutov_energy(polished) < e_new:
                relaxed = polished
                e_new = _dzugutov_energy(relaxed)
        
        # Accept if lower (or with small probability for exploration)
        if e_new < current_energy or rng.random() < 0.08:
            current = relaxed
            current_energy = e_new
        
        # Reduce step size slowly
        step_size *= 0.97
    
    return current, current_energy


def _relax_and_score(positions, potential, parameter):
    """Relax positions and return energy, positions."""
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return energy, relaxed
    except Exception:
        return np.inf, positions


def _process_instance(size, potential, parameter, time_budget):
    """Process a single instance using Langevin self-assembly and basin hopping."""
    rng = np.random.default_rng(size * 1000 + 7)
    
    best_energy = np.inf
    best_positions = None
    
    # Time allocation: split between multiple Langevin self-assembly runs and basin hopping
    n_assembly_runs = 3
    assembly_time_each = time_budget * 0.4 / n_assembly_runs
    bh_time = time_budget * 0.5
    
    # Run multiple Langevin self-assembly attempts with different seeds
    for run in range(n_assembly_runs):
        if _hard_remaining() < 10.0:
            break
        
        run_rng = np.random.default_rng(size * 7000 + run * 1000 + 3)
        
        try:
            assembled = _self_assemble_langevin(
                size, run_rng, 
                time_limit=assembly_time_each
            )
            
            if len(assembled) == size:
                energy = _dzugutov_energy(assembled)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = assembled
        except Exception:
            pass
    
    # Also try a quick random + relax as fallback
    if _hard_remaining() > 20.0:
        try:
            rand = _generate_random_compact(size, spacing=1.1, rng=np.random.default_rng(size * 2000 + 13))
            energy, positions = _relax_and_score(rand, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # Basin hopping on the best found so far
    if best_positions is not None and _hard_remaining() > 15.0:
        bh_actual_time = min(bh_time, _hard_remaining() - 8.0)
        if bh_actual_time > 5.0:
            try:
                bh_pos, bh_energy = _basin_hopping(
                    best_positions, 
                    np.random.default_rng(size * 9000 + 5),
                    max_iter=40,
                    step_size=0.35,
                    time_limit=bh_actual_time
                )
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_pos
            except Exception:
                pass
    
    # Final polish with L-BFGS-B
    if best_positions is not None and _hard_remaining() > 5.0:
        try:
            polished = _lbgfgs_polish(
                best_positions,
                time_limit=min(5.0, _hard_remaining() - 3.0)
            )
            if _dzugutov_energy(polished) < best_energy:
                best_energy = _dzugutov_energy(polished)
                best_positions = polished
        except Exception:
            pass
    
    # Final safety: ensure valid output
    if best_positions is None or len(best_positions) != size:
        radius = 1.1 * size ** (1.0 / 3.0)
        best_positions = rng.uniform(-radius, radius, (size, 3))
    
    return best_positions


def construct(instance):
    """Find low-energy Dzugutov cluster using self-assembly and basin hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget for this instance
    remaining_instances = [s for s in _INSTANCE_ORDER if s >= size]
    time_per_instance = _remaining() / max(1, len(remaining_instances))
    time_per_instance = min(time_per_instance, 45.0)
    
    # Check if we already have a result
    if size in _INSTANCE_RESULTS:
        return _INSTANCE_RESULTS[size]
    
    # Process this instance
    result = _process_instance(size, potential, parameter, time_per_instance)
    
    # Cache the result
    _INSTANCE_RESULTS[size] = result
    
    return result
# EVOLVE-BLOCK-END
