"""New program: Multi-Start Basin Hopping for Morse clusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r))(exp(rho(1-r)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    x = rho * (1.0 - r)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    xij = x[iu]
    e = np.exp(xij) * (np.exp(xij) - 2.0)
    return float(np.sum(e))


def _initial_guess(n, rho, rng):
    """Generate a compact random initial guess and relax it."""
    radius = 0.8 * n ** (1.0 / 3.0)
    pos = rng.uniform(-radius, radius, (n, 3))
    
    # Enforce minimum distance roughly
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < 0.5:
                diff = pos[i] - pos[j]
                if np.linalg.norm(diff) < 1e-6:
                    diff = rng.normal(0, 1, 3)
                diff /= np.linalg.norm(diff)
                pos[i] += diff * 0.26
                pos[j] -= diff * 0.26
                
    energy, pos = relax(pos, 'morse', rho)
    return energy, pos


def _perturb(pos, step_size, rng):
    """Apply Gaussian perturbation and enforce minimum distance."""
    perturbed = pos + rng.normal(0, step_size, pos.shape)
    n = len(perturbed)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(perturbed[i] - perturbed[j])
            if d < 0.5:
                mid = (perturbed[i] + perturbed[j]) / 2
                diff = perturbed[i] - perturbed[j]
                if np.linalg.norm(diff) < 1e-10:
                    diff = rng.normal(0, 1, 3)
                diff /= np.linalg.norm(diff)
                perturbed[i] = mid + diff * 0.25
                perturbed[j] = mid - diff * 0.25
    return perturbed


def construct(instance):
    """Multi-Start Basin Hopping optimization for Morse clusters."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter
    
    start_time = time.time()
    time_budget = 48.0  # seconds per instance
    
    rng = np.random.default_rng(42 + size)
    
    best_energy = np.inf
    best_pos = None
    
    # Strategy: Multiple independent basin hopping runs
    # Each run: Initial guess -> Relax -> N steps of BH
    # If a run finds a better global minimum, update best.
    
    # Estimate number of BH steps per run based on size
    # Smaller N: more steps. Larger N: fewer steps.
    if size <= 40:
        steps_per_run = 30
    elif size <= 60:
        steps_per_run = 20
    else:
        steps_per_run = 15
        
    step_size = 0.8
    
    run_id = 0
    while time.time() - start_time < time_budget - 2.0:
        run_id += 1
        run_start = time.time()
        
        # 1. Initial Guess
        try:
            energy, pos = _initial_guess(size, rho, rng)
        except Exception:
            continue
            
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
        # 2. Basin Hopping Steps
        current_energy = energy
        current_pos = pos
        
        for step in range(steps_per_run):
            if time.time() - start_time > time_budget - 1.0:
                break
                
            # Perturb
            perturbed = _perturb(current_pos, step_size, rng)
            
            # Relax
            try:
                new_energy, new_pos = relax(perturbed, 'morse', rho)
            except Exception:
                continue
                
            # Accept if better (Hill Climbing within the run)
            # This is a "First Improvement" or "Best Improvement" strategy.
            # For short runs, Hill Climbing is often sufficient to find the basin floor.
            if new_energy < current_energy:
                current_energy = new_energy
                current_pos = new_pos
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
                    
            # If no improvement, we could try a larger step or break.
            # For simplicity, we continue with the same step size.
            
    # Final refinement of the best found
    if best_pos is not None:
        try:
            final_energy, final_pos = relax(best_pos, 'morse', rho)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pos = final_pos
        except Exception:
            pass
            
    return best_pos
# EVOLVE-BLOCK-END
