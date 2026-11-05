"""New program using Basin Hopping global optimization for Morse cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax
import time


def _initial_positions(n, seed):
    """Generate initial compact random positions."""
    rng = np.random.default_rng(seed)
    radius = (n / 10.0) ** 0.33 * 1.5
    positions = []
    attempts = 0
    max_attempts = n * 100
    while len(positions) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if len(positions) == 0:
            positions.append(p)
        else:
            dists = np.linalg.norm(np.array(positions) - p, axis=1)
            if np.min(dists) > 0.55:
                positions.append(p)
        attempts += 1
    
    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        positions.append(p)
        
    return np.array(positions)


def construct(instance):
    """Find low-energy Morse cluster configuration using Basin Hopping."""
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Determine iterations based on size
    if n <= 40:
        num_attempts = 300
    elif n <= 60:
        num_attempts = 150
    else:
        num_attempts = 80

    best_positions = None
    best_energy = np.inf
    
    # Initial guess
    current_pos = _initial_positions(n, seed=42)
    try:
        current_energy, current_pos = relax(current_pos, potential, rho)
    except Exception:
        current_energy = np.inf
        current_pos = _initial_positions(n, seed=43)

    best_energy = current_energy
    best_positions = current_pos.copy()

    # Temperature for Metropolis acceptance
    T_start = 2.0
    T_end = 0.05
    cooling_rate = (T_end / T_start) ** (1.0 / max(num_attempts, 1))
    
    # Perturbation amplitude
    perturb_scale = 0.4
    
    rng = np.random.default_rng(12345)
    
    start_time = time.time()
    time_limit = 45.0 # Leave buffer for 6 instances (total 300s)
    
    for i in range(num_attempts):
        if time.time() - start_time > time_limit:
            break
            
        T = T_start * (cooling_rate ** i)
        
        # 1. Perturb
        noise = rng.normal(0, perturb_scale, size=(n, 3))
        perturbed_pos = current_pos + noise
        
        # 2. Relax perturbed state
        try:
            new_energy, new_pos = relax(perturbed_pos, potential, rho)
            if not np.isfinite(new_energy):
                continue
        except Exception:
            continue
            
        # 3. Metropolis Acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / T)):
            current_pos = new_pos
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_pos.copy()
                
    return best_positions
# EVOLVE-BLOCK-END
