"""Morse cluster global minima: Langevin dynamics with simulated annealing cooling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 280.0  # 20s safety margin before 300s kill
_instance_call_count = [0]


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(positions, rho):
    """Compute Morse energy for a configuration."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            x = rho * (1.0 - r)
            ex = np.exp(x)
            energy += ex * (ex - 2.0)
    return energy


def _langevin_anneal(positions, rho, dt, T0, T_final, n_steps, rng, deadline):
    """Run Langevin dynamics with linear cooling from T0 to T_final.
    
    Returns (final_positions, best_positions, best_energy).
    """
    n_atoms = len(positions)
    pos = positions.copy()
    vel = np.zeros_like(pos)
    
    best_energy = _morse_energy(pos, rho)
    best_pos = pos.copy()
    
    # Adaptive step size
    step_scale = 1.0
    consecutive_rejects = 0
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Linear cooling schedule
        alpha = step / max(n_steps - 1, 1)
        T = T0 + (T_final - T0) * alpha
        
        # Get gradient
        try:
            grad = morse_energy_gradient(pos, rho)
        except Exception:
            break
        
        # Langevin update: pos += dt * (-grad + noise)
        # Friction term implicit in dt scaling
        noise = rng.standard_normal(pos.shape) * np.sqrt(2.0 * T * dt)
        pos_new = pos + dt * (-grad + noise)
        
        # Compute energy difference
        E_old = _morse_energy(pos, rho)
        E_new = _morse_energy(pos_new, rho)
        dE = E_new - E_old
        
        # Metropolis acceptance
        if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-10)):
            pos = pos_new
            consecutive_rejects = 0
            # Adapt step size upward slightly
            step_scale = min(step_scale * 1.02, 3.0)
        else:
            consecutive_rejects += 1
            # Adapt step size downward if many rejections
            if consecutive_rejects > 10:
                step_scale = max(step_scale * 0.9, 0.1)
                consecutive_rejects = 0
        
        # Update best
        if E_new < best_energy:
            best_energy = E_new
            best_pos = pos_new.copy()
    
    return pos, best_pos, best_energy


def _ensure_min_distance(positions, min_dist=0.5):
    """Ensure all pairs are at least min_dist apart."""
    n = len(positions)
    for _ in range(50):  # Iterate a few times
        fixed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    fixed = True
                    mid = 0.5 * (positions[i] + positions[j])
                    direction = positions[i] - positions[j]
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        direction = np.array([1.0, 0.0, 0.0])
                        norm = 1.0
                    direction /= norm
                    offset = 0.51 * direction
                    positions[i] = mid + offset
                    positions[j] = mid - offset
        if not fixed:
            break
    return positions


def construct(instance):
    """Find low-energy Morse cluster via Langevin annealing from ideal gas init."""
    _instance_call_count[0] += 1
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Per-instance time budget
    per_instance_budget = 44.0
    if _instance_call_count[0] <= 2:
        per_instance_budget = 48.0
    elif _instance_call_count[0] >= 5:
        per_instance_budget = 40.0
    
    instance_deadline = time.time() + per_instance_budget
    
    # Ideal gas initialization: random positions in a sphere
    # Radius chosen so that average inter-particle spacing ~ 1.0
    # Volume ~ n * (4/3)*pi*(0.5)^3 for close packing, but we want spread out
    # Use radius such that density gives avg distance ~ 1
    radius = 1.2 * (size / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)
    radius = max(radius, 2.0)
    
    rng = np.random.default_rng(42 + _instance_call_count[0])
    
    # Multiple restarts with different seeds and parameters
    best_energy = np.inf
    best_positions = None
    
    n_restarts = 4
    time_per_restart = (per_instance_budget - 5.0) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > instance_deadline - 3.0:
            break
        
        # Initialize from ideal gas (random in sphere)
        pos_init = rng.uniform(-radius, radius, (size, 3))
        # Reject points outside sphere
        for i in range(size):
            while np.linalg.norm(pos_init[i]) > radius:
                pos_init[i] = rng.uniform(-radius, radius, 3)
        
        # Annealing parameters
        # Higher rho means steeper potential, need smaller steps
        dt_base = 0.002 / max(parameter, 1.0)
        dt = dt_base
        
        # Temperature: start high enough to explore, cool to near zero
        T0 = 2.0 / max(parameter, 1.0)
        T_final = 0.01 / max(parameter, 1.0)
        
        # Number of steps: use time budget
        restart_deadline = time.time() + time_per_restart - 1.0
        n_steps = 2000  # Will be limited by deadline
        
        try:
            final_pos, best_pos, best_e = _langevin_anneal(
                pos_init, parameter, dt, T0, T_final, n_steps, rng, restart_deadline
            )
            
            if best_e < best_energy:
                best_energy = best_e
                best_positions = best_pos.copy()
        except Exception:
            continue
    
    # Final polish: run relax on the best found configuration
    if best_positions is not None and time.time() < instance_deadline - 2.0:
        try:
            energy, positions = relax(best_positions, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # If still nothing, fallback to simple random + relax
    if best_positions is None:
        pos_init = rng.uniform(-radius, radius, (size, 3))
        for i in range(size):
            while np.linalg.norm(pos_init[i]) > radius:
                pos_init[i] = rng.uniform(-radius, radius, 3)
        try:
            energy, positions = relax(pos_init, potential, parameter)
            best_energy = energy
            best_positions = positions
        except Exception:
            best_positions = pos_init
    
    # Ensure minimum distance constraint
    best_positions = _ensure_min_distance(best_positions, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
