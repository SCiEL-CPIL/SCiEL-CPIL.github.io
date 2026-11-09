"""Physics-driven self-assembly for Dzugutov cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _enforce_min_distance(positions, min_dist=0.5, max_iter=20):
    """Enforce minimum distance between all atom pairs."""
    pos = positions.copy()
    n = len(pos)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_dist and dist > 1e-10:
                    push = (min_dist - dist) / dist * d
                    pos[i] += 0.5 * push
                    pos[j] -= 0.5 * push
                    moved = True
        if not moved:
            break
    return pos


def _random_langevin_anneal(n, seed, time_budget=15):
    """Single random start with simulated annealing via Langevin dynamics."""
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    # Random initial positions in a sphere
    radius = 1.2 * n ** (1.0 / 3.0)
    positions = rng.uniform(-radius, radius, size=(n, 3))
    positions = _enforce_min_distance(positions)
    velocities = rng.normal(0, 0.1, size=(n, 3))
    
    # Adaptive step size and temperature schedule
    dt = 0.005
    damping = 1.0
    t0_temp = 0.8
    t1_temp = 0.005
    total_steps = 500
    
    best_energy = np.inf
    best_positions = positions.copy()
    
    for step in range(total_steps):
        elapsed = time.time() - start_time
        if elapsed > time_budget - 2:
            break
        
        progress = step / max(1, total_steps)
        # Exponential cooling
        temp = t1_temp + (t0_temp - t1_temp) * np.exp(-5.0 * progress)
        
        # Compute energy and gradient using the provided function
        energy, grad = dzugutov_energy_gradient(positions)
        forces = -grad
        
        # Langevin step
        sigma = np.sqrt(2 * temp * damping)
        noise = rng.normal(0, 1, size=positions.shape)
        new_vel = velocities + (forces - damping * velocities) * dt + sigma * np.sqrt(dt) * noise
        new_pos = positions + new_vel * dt
        new_pos = _enforce_min_distance(new_pos)
        
        positions = new_pos
        velocities = new_vel
        
        # Periodic relaxation with scipy
        if step % 100 == 99 or step == total_steps - 1:
            energy, pos = relax(positions, 'dzugutov', None)
            if energy < best_energy:
                best_energy = energy
                best_positions = pos.copy()
                positions = pos.copy()
                velocities = np.zeros_like(positions)
    
    # Final relaxation
    energy, pos = relax(positions, 'dzugutov', None)
    if energy < best_energy:
        best_energy = energy
        best_positions = pos.copy()
    
    return best_positions, best_energy


def _polish(positions, seed, time_budget=5):
    """Polish with small perturbations and relaxation."""
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    energy, _ = dzugutov_energy_gradient(positions)
    best_energy = energy
    best_pos = positions.copy()
    
    n_attempts = max(2, min(10, int(time_budget * 2)))
    
    for _ in range(n_attempts):
        if time.time() - start_time > time_budget - 1:
            break
        
        # Small random perturbation
        sigma = 0.05
        pos = positions + rng.normal(0, sigma, size=positions.shape)
        pos = _enforce_min_distance(pos)
        
        energy, pos = relax(pos, 'dzugutov', None)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            positions = pos.copy()
    
    return best_pos


def construct(instance):
    """Physics-driven self-assembly for Dzugutov cluster global minima."""
    size = instance["n"]
    potential = instance["potential"]
    
    if potential != 'dzugutov':
        spacing = {"morse": 1.0, "sutton_chen": 0.72}.get(potential, 1.0)
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), potential, instance.get("rho", instance.get("kind")))
        return positions
    
    # 300s total / 6 instances = 50s per instance, use 45s
    total_budget = 45
    t0 = time.time()
    
    # Multiple independent random Langevin annealing runs
    n_starts = 4
    per_start_budget = (total_budget - 5) / n_starts  # Reserve 5s for polishing
    
    best_energy = np.inf
    best_pos = None
    
    for start in range(n_starts):
        elapsed = time.time() - t0
        remaining = total_budget - 5 - elapsed  # Leave 5s for polish
        if remaining < 2:
            break
        
        seed = size * 7919 + start * 1000 + 1
        pos, energy = _random_langevin_anneal(size, seed, time_budget=min(per_start_budget, remaining))
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # If we still have time, try one more start with tighter constraints
    elapsed = time.time() - t0
    remaining = total_budget - 5 - elapsed
    if remaining > 3 and best_pos is not None:
        seed = size * 7919 + 5000
        pos, energy = _random_langevin_anneal(size, seed, time_budget=min(remaining - 1, 8))
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # Final polish
    remaining = total_budget - (time.time() - t0)
    if best_pos is not None and remaining > 2:
        seed3 = size * 7919 + 3
        best_pos = _polish(best_pos, seed3, time_budget=min(5, remaining - 1))
    
    if best_pos is None:
        # Fallback: simple random + relax
        rng = np.random.default_rng(size)
        radius = 1.2 * size ** (1.0 / 3.0)
        positions = rng.uniform(-radius, radius, size=(size, 3))
        positions = _enforce_min_distance(positions)
        energy, best_pos = relax(positions, 'dzugutov', None)
    
    return best_pos
# EVOLVE-BLOCK-END
