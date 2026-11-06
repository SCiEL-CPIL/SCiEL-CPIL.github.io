"""Morse cluster global minima: Basin hopping with diverse structural seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 290.0  # 10s safety margin before 300s kill
_instance_call_count = [0]


def _time_left():
    return _HARD_DEADLINE - time.time()


def _fast_morse_energy(positions, rho):
    """
    Fast vectorized Morse energy calculation.
    E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)
    """
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    u = rho * (1.0 - dists)
    # Clip to avoid overflow
    u = np.clip(u, -500, 500)
    exp_u = np.exp(u)
    v = exp_u * (exp_u - 2.0)
    
    # Sum over i<j (half of symmetric matrix, excluding diagonal)
    # Set inf to 0
    v[np.isinf(v)] = 0.0
    
    total_energy = 0.5 * np.sum(v)
    return total_energy


def _enforce_min_dist(positions, min_dist=0.501):
    """
    Ensure all pairwise distances are at least min_dist.
    Uses iterative repulsion for pairs that are too close.
    """
    n = len(positions)
    if n < 2:
        return positions.copy()
    
    pos = positions.copy()
    
    for _ in range(50):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Find pairs too close
        close_mask = dists < min_dist
        if not np.any(close_mask):
            break
        
        # Repel close pairs
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < min_dist:
                    d = pos[i] - pos[j]
                    norm = np.linalg.norm(d)
                    if norm < 1e-10:
                        d = np.array([1.0, 0.0, 0.0])
                        norm = 1.0
                    else:
                        d = d / norm
                    # Push apart by half the deficit each
                    deficit = min_dist - dists[i, j]
                    pos[i] += 0.5 * deficit * d
                    pos[j] -= 0.5 * deficit * d
    
    return pos


def _fcc_points(n, seed=0):
    """Generate n points from an FCC lattice."""
    rng = np.random.default_rng(seed)
    a = 1.0  # lattice constant
    points = []
    side = int(np.ceil(n ** (1/3))) + 2
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                # FCC basis: (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
                base = np.array([i * a, j * a, k * a])
                for bx, by, bz in [(0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)]:
                    points.append(base + np.array([bx, by, bz]))
    points = np.array(points)
    
    if len(points) >= n:
        idx = rng.choice(len(points), n, replace=False)
        return points[idx]
    else:
        # Pad with random
        extra = _random_compact(n - len(points), seed=seed + 100)
        return np.vstack([points, extra])


def _bcc_points(n, seed=0):
    """Generate n points from a BCC lattice."""
    rng = np.random.default_rng(seed)
    a = 1.0
    points = []
    side = int(np.ceil(n ** (1/3))) + 2
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                base = np.array([i * a, j * a, k * a])
                for bx, by, bz in [(0,0,0), (a/2,a/2,a/2)]:
                    points.append(base + np.array([bx, by, bz]))
    points = np.array(points)
    
    if len(points) >= n:
        idx = rng.choice(len(points), n, replace=False)
        return points[idx]
    else:
        extra = _random_compact(n - len(points), seed=seed + 100)
        return np.vstack([points, extra])


def _sc_points(n, seed=0):
    """Generate n points from a simple cubic lattice."""
    rng = np.random.default_rng(seed)
    a = 1.0
    points = []
    side = int(np.ceil(n ** (1/3))) + 2
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                points.append([i * a, j * a, k * a])
    points = np.array(points)
    
    if len(points) >= n:
        idx = rng.choice(len(points), n, replace=False)
        return points[idx]
    else:
        extra = _random_compact(n - len(points), seed=seed + 100)
        return np.vstack([points, extra])


def _icosahedral_shells(n, seed=0):
    """
    Generate points on concentric icosahedral shells.
    """
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]  # center
    shell_radii = [1.0, 1.6, 2.2, 2.8, 3.4, 4.0, 4.6, 5.2]
    shell_counts = [12, 30, 60, 90, 120, 150, 180, 210]
    
    for radius, count in zip(shell_radii, shell_counts):
        if len(points) >= n:
            break
        # Place points on icosahedron scaled to radius
        for _ in range(count):
            v = verts[rng.integers(0, 12)] * radius
            # Add small jitter to avoid exact overlaps
            v += rng.normal(0, 0.05, 3)
            points.append(v)
    
    points = np.array(points)
    if len(points) >= n:
        idx = rng.choice(len(points), n, replace=False)
        return points[idx]
    else:
        extra = _random_compact(n - len(points), seed=seed + 100)
        return np.vstack([points, extra])


def _random_compact(n, seed=0):
    """
    Generate n random points in a sphere, ensuring min distance.
    """
    rng = np.random.default_rng(seed)
    radius = max(2.0, n ** (1.0/3.0) * 1.2)
    
    points = []
    min_dist = 0.55
    
    attempts = 0
    max_attempts = n * 500
    while len(points) < n and attempts < max_attempts:
        vec = rng.normal(size=3)
        norm = np.linalg.norm(vec)
        if norm > 0:
            p = vec / norm * radius * (rng.uniform(0, 1) ** (1/3))
            if len(points) == 0:
                points.append(p)
            else:
                pts = np.array(points)
                dists = np.linalg.norm(pts - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
        attempts += 1
        
    if len(points) < n:
        while len(points) < n:
            vec = rng.normal(size=3)
            norm = np.linalg.norm(vec)
            if norm > 0:
                p = vec / norm * radius * (rng.uniform(0, 1) ** (1/3))
            else:
                p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points)


def _perturb(positions, scale, seed=0):
    """
    Perturb positions by adding random noise.
    """
    rng = np.random.default_rng(seed)
    return positions + rng.normal(0, scale, positions.shape)


def _relax_one(positions, rho):
    """
    Relax a single configuration and return (energy, positions).
    """
    try:
        energy, pos = relax(positions, 'morse', rho)
        return energy, pos
    except:
        return _fast_morse_energy(positions, rho), positions.copy()


def _basin_hop(positions, rho, time_limit, seed=0):
    """
    Run basin hopping on initial positions for time_limit seconds.
    """
    rng = np.random.default_rng(seed)
    n = len(positions)
    
    # Initial relaxation
    current_energy, current_pos = _relax_one(positions, rho)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    start_time = time.time()
    step_count = 0
    
    # Adaptive kick scale
    kick_scale = 0.3
    min_kick = 0.05
    max_kick = 1.0
    
    while time.time() - start_time < time_limit:
        if time.time() > _HARD_DEADLINE - 5.0:
            break
        
        # Perturb
        perturbed = _perturb(current_pos, kick_scale, seed=seed + step_count)
        perturbed = _enforce_min_dist(perturbed)
        
        # Relax
        new_energy, new_pos = _relax_one(perturbed, rho)
        
        # Acceptance: always accept if better, else with probability
        delta_e = new_energy - current_energy
        if delta_e < 0 or (rng.random() < np.exp(-delta_e / max(0.1, kick_scale * 10))):
            current_energy = new_energy
            current_pos = new_pos
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
            # Reduce kick scale on success
            kick_scale = max(min_kick, kick_scale * 0.95)
        else:
            # Increase kick scale on failure
            kick_scale = min(max_kick, kick_scale * 1.1)
        
        step_count += 1
        
        # Every 20 steps, try a larger perturbation
        if step_count % 20 == 0:
            large_perturbed = _perturb(best_pos, kick_scale * 2, seed=seed + step_count + 1000)
            large_perturbed = _enforce_min_dist(large_perturbed)
            large_energy, large_pos = _relax_one(large_perturbed, rho)
            if large_energy < best_energy:
                best_energy = large_energy
                best_pos = large_pos.copy()
                current_energy = large_energy
                current_pos = large_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """
    Find low-energy Morse cluster configurations.
    """
    _instance_call_count[0] += 1
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Time budget
    per_instance_budget = 42.0
    if _instance_call_count[0] == 1:
        per_instance_budget = 45.0
    elif _instance_call_count[0] == 6:
        per_instance_budget = 38.0
        
    instance_deadline = time.time() + per_instance_budget
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. FCC
    for seed in range(3):
        initial_configs.append(_fcc_points(n, seed=seed))
        
    # 2. BCC
    for seed in range(3):
        initial_configs.append(_bcc_points(n, seed=seed + 10))
        
    # 3. Simple cubic
    for seed in range(2):
        initial_configs.append(_sc_points(n, seed=seed + 20))
        
    # 4. Icosahedral shells
    for seed in range(3):
        initial_configs.append(_icosahedral_shells(n, seed=seed + 30))
        
    # 5. Random compact
    for seed in range(4):
        initial_configs.append(_random_compact(n, seed=seed + 40))
        
    # Run basin hopping on each, keeping the best
    best_energy = np.inf
    best_positions = None
    
    num_runs = len(initial_configs)
    time_per_run = (per_instance_budget - 5.0) / num_runs
    
    for i, init_pos in enumerate(initial_configs):
        if time.time() > instance_deadline - 2.0:
            break
            
        try:
            e, pos = _basin_hop(init_pos, rho, time_per_run, seed=i + 42)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except Exception:
            pass
            
    # If no run succeeded, fallback
    if best_positions is None:
        init_pos = _random_compact(n, seed=0)
        best_energy, best_positions = _relax_one(init_pos, rho)
        
    # Final refinement
    if time.time() < instance_deadline - 1.0:
        try:
            final_energy, final_pos = relax(best_positions, 'morse', rho)
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
        except:
            pass
            
    # Ensure minimum distance
    best_positions = _enforce_min_dist(best_positions, min_dist=0.501)
    
    return best_positions
# EVOLVE-BLOCK-END
