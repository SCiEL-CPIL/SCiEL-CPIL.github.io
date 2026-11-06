"""Morse cluster global minima: Basin hopping with robust relax and lattice seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 285.0  # 15s safety margin for 300s limit
_INSTANCE_COUNT = 6
_INSTANCE_INDEX = [0]

def _enforce_min_distance(positions, min_dist=0.5):
    """Ensure all pairwise distances are at least min_dist."""
    n = len(positions)
    if n < 2:
        return positions
        
    for _ in range(20):
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        mask = (dist < min_dist) & (~np.eye(n, dtype=bool))
        if not np.any(mask):
            break
            
        i_idx, j_idx = np.where(mask)
        
        moved = False
        for i, j in zip(i_idx, j_idx):
            if i >= j: continue
            d = dist[i, j]
            if d < 1e-8:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = (positions[i] - positions[j]) / d
                
            push = (min_dist - d) / 2.0 + 0.01
            positions[i] += push * direction
            positions[j] -= push * direction
            moved = True
            
        if not moved:
            break
            
    return positions

def _morse_energy(positions, rho):
    """Compute Morse energy: E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)"""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Only sum upper triangle
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e = np.sum(ex * (ex - 2.0))
    return float(e)

def _simulated_annealing(n, rho, time_budget, seed=None):
    """
    Run Simulated Annealing using Langevin dynamics.
    Returns (best_energy, best_positions).
    """
    rng = np.random.default_rng(seed)
    
    # Initialize with random positions (Ideal Gas)
    # Scale radius to ensure some overlap/interaction but not too dense
    radius = 1.5 * n ** (1.0/3.0)
    positions = rng.uniform(-radius, radius, (n, 3))
    velocities = rng.normal(0.0, 1.0, (n, 3))
    
    # Parameters
    # Morse potential has characteristic frequency ~ sqrt(2*rho)
    omega = np.sqrt(2.0 * rho)
    dt = 0.005 / omega  # Small time step
    gamma = 0.1 * omega # Friction coefficient
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.01
    
    # Estimate max steps based on time budget
    # Each step involves gradient calculation (O(n^2)) and updates
    # We measure time per step dynamically in the first few iterations
    start_time = time.time()
    best_energy = np.inf
    best_positions = None
    
    # Initial energy
    try:
        current_energy = _morse_energy(positions, rho)
    except Exception:
        current_energy = 1e10
        
    if current_energy < best_energy:
        best_energy = current_energy
        best_positions = positions.copy()
        
    # Warm up to estimate step time
    steps_per_sec = 0
    warmup_steps = 10
    for _ in range(warmup_steps):
        try:
            grad = morse_energy_gradient(positions, rho)
            # Langevin update
            force = -grad
            # v_new = v + dt * (F/m - gamma * v) + sqrt(2 * gamma * T * dt) * noise
            # m=1
            noise = rng.normal(0.0, np.sqrt(2.0 * gamma * T_start * dt), (n, 3))
            velocities = velocities + dt * (force - gamma * velocities) + noise
            positions = positions + dt * velocities
            
            # Clip velocities to prevent explosion
            v_norms = np.linalg.norm(velocities, axis=1)
            max_v = 10.0 / dt
            mask = v_norms > max_v
            if np.any(mask):
                velocities[mask] *= max_v / v_norms[mask]
                
            # Check min distance
            positions = _enforce_min_distance(positions, 0.5)
            
            current_energy = _morse_energy(positions, rho)
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = positions.copy()
        except Exception:
            pass
            
    elapsed_warmup = time.time() - start_time
    if elapsed_warmup > 0:
        steps_per_sec = warmup_steps / elapsed_warmup
    else:
        steps_per_sec = 100.0 # Fallback
        
    max_steps = int(time_budget * steps_per_sec * 0.8) # Leave 20% for overhead
    max_steps = max(100, min(max_steps, 10000)) # Cap steps
    
    # Main loop
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
            
        # Linear cooling
        progress = step / max_steps
        T = T_start + (T_end - T_start) * progress
        
        try:
            grad = morse_energy_gradient(positions, rho)
            force = -grad
            
            # Langevin update
            noise = rng.normal(0.0, np.sqrt(2.0 * gamma * T * dt), (n, 3))
            velocities = velocities + dt * (force - gamma * velocities) + noise
            positions = positions + dt * velocities
            
            # Clip velocities
            v_norms = np.linalg.norm(velocities, axis=1)
            max_v = 10.0 / dt
            mask = v_norms > max_v
            if np.any(mask):
                velocities[mask] *= max_v / v_norms[mask]
                
            # Enforce min distance
            positions = _enforce_min_distance(positions, 0.5)
            
            current_energy = _morse_energy(positions, rho)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = positions.copy()
                
        except Exception:
            # If gradient fails, restart with random positions
            positions = rng.uniform(-radius, radius, (n, 3))
            velocities = rng.normal(0.0, 1.0, (n, 3))
            continue
            
    return best_energy, best_positions

def construct(instance):
    """Construct low-energy Morse cluster configuration."""
    _INSTANCE_INDEX[0] += 1
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Time management
    elapsed = time.time() - _START_TIME
    remaining_total = _HARD_DEADLINE - time.time()
    instances_left = _INSTANCE_COUNT - _INSTANCE_INDEX[0] + 1
    
    # Allocate time budget for this instance
    time_budget = max(5.0, min(48.0, remaining_total / instances_left))
    
    start_time = time.time()
    best_energy = np.inf
    best_positions = None
    
    # Run multiple independent SA chains
    # Use 2 chains to balance exploration and exploitation
    n_chains = 2
    chain_budget = time_budget / n_chains - 1.0 # Leave 1s for final polish
    
    for c in range(n_chains):
        if time.time() - start_time > time_budget - 2.0:
            break
            
        # Different seeds for diversity
        seed_val = 42 + c * 1000
        
        try:
            e, pos = _simulated_annealing(size, rho, chain_budget, seed=seed_val)
            if e < best_energy:
                best_energy = e
                best_positions = pos.copy()
        except Exception:
            continue
            
    # Fallback if SA failed
    if best_positions is None:
        best_positions = np.random.uniform(-2, 2, (size, 3))
        best_positions = _enforce_min_distance(best_positions, 0.5)
        try:
            best_energy, best_positions = relax(best_positions, potential, rho)
        except Exception:
            best_energy = _morse_energy(best_positions, rho)

    # Final polish: High-precision relaxation
    if best_positions is not None and time.time() - start_time < time_budget - 0.5:
        try:
            e, pos = relax(best_positions, potential, rho)
            if e < best_energy:
                best_energy = e
                best_positions = pos.copy()
        except Exception:
            pass

    best_positions = _enforce_min_distance(best_positions, 0.5)
    return best_positions
# EVOLVE-BLOCK-END
