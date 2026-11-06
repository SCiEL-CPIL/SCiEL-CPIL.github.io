"""Morse cluster global minima: Basin hopping with robust relax and lattice seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 285.0  # 15s safety margin for 300s limit
_INSTANCE_COUNT = 6
_INSTANCE_INDEX = [0]

def _generate_fcc_lattice(n, spacing=1.0):
    """Generate n points from an FCC lattice centered at origin."""
    r_max = spacing * (n ** (1.0/3.0) + 2)
    limit = int(r_max / spacing) + 2
    
    candidates = []
    # FCC points: (i, j, k) where i,j,k integers and i+j+k even
    for i in range(-2*limit, 2*limit + 1):
        for j in range(-2*limit, 2*limit + 1):
            for k in range(-2*limit, 2*limit + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (spacing / 2.0)
                    candidates.append(p)
    
    if not candidates:
        return np.random.uniform(-r_max, r_max, (n, 3))
        
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    sorted_idx = np.argsort(dists)
    selected = candidates[sorted_idx[:n]]
    
    selected -= np.mean(selected, axis=0)
    return selected

def _generate_bcc_lattice(n, spacing=1.0):
    """Generate n points from a BCC lattice centered at origin."""
    r_max = spacing * (n ** (1.0/3.0) + 2)
    limit = int(r_max / spacing) + 2
    
    candidates = []
    # BCC points: (i,j,k) * (a/2) where i,j,k have same parity
    for i in range(-2*limit, 2*limit + 1):
        for j in range(-2*limit, 2*limit + 1):
            for k in range(-2*limit, 2*limit + 1):
                if (i % 2) == (j % 2) and (j % 2) == (k % 2):
                    p = np.array([i, j, k]) * (spacing / 2.0)
                    candidates.append(p)
    
    if not candidates:
        return np.random.uniform(-r_max, r_max, (n, 3))
        
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    sorted_idx = np.argsort(dists)
    selected = candidates[sorted_idx[:n]]
    
    selected -= np.mean(selected, axis=0)
    return selected

def _generate_random(n, spacing=1.0, seed=None):
    """Generate n random points."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = rng.uniform(-radius, radius, (n, 3))
    return points

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
    time_budget = max(5.0, min(45.0, remaining_total / instances_left))
    
    start_time = time.time()
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse seeds
    seeds = []
    
    # 1. FCC Lattice
    try:
        seeds.append(_generate_fcc_lattice(size, spacing=1.0))
    except Exception:
        pass
        
    # 2. BCC Lattice
    try:
        seeds.append(_generate_bcc_lattice(size, spacing=1.0))
    except Exception:
        pass
        
    # 3. Random structures
    for s in [42, size, size * 7]:
        try:
            seeds.append(_generate_random(size, spacing=1.0, seed=s))
        except Exception:
            pass

    if not seeds:
        seeds.append(_generate_random(size, 1.0, 123))

    # Phase 1: Relax all seeds
    for seed_idx, seed_pos in enumerate(seeds):
        if time.time() - start_time > time_budget * 0.5:
            break
            
        jittered = seed_pos + np.random.randn(*seed_pos.shape) * 0.01
        jittered = _enforce_min_distance(jittered, 0.5)
        
        try:
            energy, positions = relax(jittered, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue

    if best_positions is None:
        best_positions = seeds[0].copy()
        best_energy = _morse_energy(best_positions, rho)

    # Phase 2: Basin Hopping with adaptive kick scale
    time_elapsed = time.time() - start_time
    remaining_time = time_budget - time_elapsed
    
    if remaining_time > 2.0 and best_positions is not None:
        kick_scale = 0.3
        # Use a cooling schedule: start large, gradually reduce
        # But keep it simple: constant kick, but try more iterations
        
        # Estimate how many kicks we can do
        # Measure average relax time
        t0 = time.time()
        try:
            test_kick = best_positions + np.random.randn(*best_positions.shape) * kick_scale
            test_kick = _enforce_min_distance(test_kick, 0.5)
            relax(test_kick, potential, rho)
            avg_relax_time = time.time() - t0
        except Exception:
            avg_relax_time = 0.5
            
        if avg_relax_time < 0.1:
            max_kicks = int(remaining_time / (avg_relax_time + 0.05))
        else:
            max_kicks = int(remaining_time / (avg_relax_time * 1.5))
            
        max_kicks = max(5, min(max_kicks, 200))
        
        for kick_iter in range(max_kicks):
            if time.time() - start_time > time_budget - 1.0:
                break
                
            # Adaptive kick: slightly reduce as we go
            current_kick = kick_scale * (1.0 - 0.3 * (kick_iter / max_kicks))
            
            kicked = best_positions + np.random.randn(*best_positions.shape) * current_kick
            kicked = _enforce_min_distance(kicked, 0.5)
            
            try:
                energy, positions = relax(kicked, potential, rho)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue

    # Final polish: One more relax on the best position
    if best_positions is not None:
        try:
            final_energy, final_pos = relax(best_positions, potential, rho)
            if final_energy < best_energy:
                best_positions = final_pos
        except Exception:
            pass

    best_positions = _enforce_min_distance(best_positions, 0.5)
    return best_positions
# EVOLVE-BLOCK-END
