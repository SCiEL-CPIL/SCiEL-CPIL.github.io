"""Morse cluster global minima: Core-Shell and Basin Hopping search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 280.0  # 20s safety margin before 300s kill
_instance_call_count = [0]


def _time_left():
    return _HARD_DEADLINE - time.time()


def _fcc_points(n, spacing=1.0):
    """Generate n points from an FCC lattice centered at origin."""
    a = spacing * np.sqrt(2.0)
    density = 4.0 / (a ** 3)
    vol_needed = n / density
    r_est = (3.0 * vol_needed / (4.0 * np.pi)) ** (1.0 / 3.0)
    r_est *= 1.2
    
    points = []
    # Iterate over a cubic grid
    limit = int(np.ceil(r_est / a)) + 2
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                # FCC lattice points: (0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)
                for offset in [(0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    p = np.array([i*a + offset[0], j*a + offset[1], k*a + offset[2]])
                    if np.linalg.norm(p) <= r_est:
                        points.append(p)
    
    if len(points) < n:
        # Fill with random points if lattice is too sparse or estimation failed
        rng = np.random.default_rng(0)
        while len(points) < n:
            p = rng.uniform(-r_est, r_est, 3)
            if np.linalg.norm(p) <= r_est:
                points.append(p)
    else:
        # Randomly select n points
        indices = np.random.default_rng(1).choice(len(points), n, replace=False)
        points = [points[i] for i in indices]
        
    return np.array(points)

def _bcc_points(n, spacing=1.0):
    """Generate n points from a BCC lattice centered at origin."""
    a = spacing * np.sqrt(3.0) / 2.0 # Nearest neighbor is a*sqrt(3)/2
    density = 2.0 / (a ** 3)
    vol_needed = n / density
    r_est = (3.0 * vol_needed / (4.0 * np.pi)) ** (1.0 / 3.0)
    r_est *= 1.2
    
    points = []
    limit = int(np.ceil(r_est / a)) + 2
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                # BCC lattice points: (0,0,0), (a/2, a/2, a/2)
                for offset in [(0,0,0), (a/2, a/2, a/2)]:
                    p = np.array([i*a + offset[0], j*a + offset[1], k*a + offset[2]])
                    if np.linalg.norm(p) <= r_est:
                        points.append(p)
                        
    if len(points) < n:
        rng = np.random.default_rng(2)
        while len(points) < n:
            p = rng.uniform(-r_est, r_est, 3)
            if np.linalg.norm(p) <= r_est:
                points.append(p)
    else:
        indices = np.random.default_rng(3).choice(len(points), n, replace=False)
        points = [points[i] for i in indices]
        
    return np.array(points)

def _sc_points(n, spacing=1.0):
    """Generate n points from a SC lattice centered at origin."""
    a = spacing
    density = 1.0 / (a ** 3)
    vol_needed = n / density
    r_est = (3.0 * vol_needed / (4.0 * np.pi)) ** (1.0 / 3.0)
    r_est *= 1.2
    
    points = []
    limit = int(np.ceil(r_est / a)) + 2
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                p = np.array([i*a, j*a, k*a])
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
                    
    if len(points) < n:
        rng = np.random.default_rng(4)
        while len(points) < n:
            p = rng.uniform(-r_est, r_est, 3)
            if np.linalg.norm(p) <= r_est:
                points.append(p)
    else:
        indices = np.random.default_rng(5).choice(len(points), n, replace=False)
        points = [points[i] for i in indices]
        
    return np.array(points)

def _basin_hopping(seed, potential, rho, max_kicks, time_limit):
    """
    Perform basin hopping with adaptive kick scale.
    """
    best_pos = seed.copy()
    best_energy = np.inf
    current_pos = seed.copy()
    
    kick_scale = 0.2
    
    for k in range(max_kicks):
        if time.time() > time_limit:
            break
        if _time_left() < 5.0:
            break
            
        try:
            energy, pos = relax(current_pos, potential, rho)
        except Exception:
            break
            
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
        # Adaptive kick: if we found a better energy, reduce kick scale slightly
        # Otherwise, increase it to explore more
        if energy < best_energy:
            kick_scale *= 0.9
        else:
            kick_scale *= 1.1
        kick_scale = np.clip(kick_scale, 0.05, 0.5)
        
        rng = np.random.default_rng(k + 1000)
        noise = rng.normal(0, kick_scale, pos.shape)
        current_pos = pos + noise
        
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Morse cluster configurations via Lattice Seeds and Basin Hopping."""
    _instance_call_count[0] += 1
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Estimate per-instance time budget
    per_instance_budget = 44.0
    if _instance_call_count[0] <= 2:
        per_instance_budget = 46.0
    elif _instance_call_count[0] >= 5:
        per_instance_budget = 40.0
        
    instance_deadline = time.time() + per_instance_budget
    
    # Define seeds from lattices
    seeds = []
    seeds.append(_fcc_points(size))
    seeds.append(_bcc_points(size))
    seeds.append(_sc_points(size))
    
    # Add a random seed for diversity
    rng = np.random.default_rng(42)
    random_seed = rng.uniform(-size**(1/3), size**(1/3), (size, 3))
    seeds.append(random_seed)
    
    best_energy = np.inf
    best_positions = None
    
    # Distribute time among seeds
    n_seeds = len(seeds)
    time_per_seed = (instance_deadline - time.time()) / n_seeds
    
    for i, seed in enumerate(seeds):
        if time.time() > instance_deadline - 2.0:
            break
            
        # Adjust max_kicks based on time available for this seed
        # Assume relax takes ~1-2s for large clusters, so limit kicks
        max_kicks = int(time_per_seed / 2.0)
        max_kicks = max(1, min(max_kicks, 10))
        
        pos, energy = _basin_hopping(seed, potential, parameter, max_kicks, instance_deadline - 1.0)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
            
    # If no valid position found (shouldn't happen), fallback
    if best_positions is None:
        best_positions = _fcc_points(size)
        
    # Safety check: minimum distance
    n = len(best_positions)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_positions[i] - best_positions[j])
            if d < 0.5:
                mid = 0.5 * (best_positions[i] + best_positions[j])
                direction = best_positions[i] - best_positions[j]
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    direction = np.array([1.0, 0.0, 0.0])
                    norm = 1.0
                direction /= norm
                offset = 0.26 * direction
                best_positions[i] = mid + offset
                best_positions[j] = mid - offset

    return best_positions
# EVOLVE-BLOCK-END
