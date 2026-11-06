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
    time_budget = max(5.0, min(48.0, remaining_total / instances_left))
    
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

    # Phase 2: Basin Hopping with multiple parallel chains
    time_elapsed = time.time() - start_time
    remaining_time = time_budget - time_elapsed
    
    if remaining_time > 3.0 and best_positions is not None:
        # Use multiple independent chains to explore different basins
        n_chains = min(4, max(2, int(remaining_time / 8.0)))
        chain_budget = remaining_time / n_chains
        
        # Initialize chains from best position with different kick scales
        chain_starts = []
        for c in range(n_chains):
            kick = 0.2 + 0.15 * c
            kicked = best_positions + np.random.randn(*best_positions.shape) * kick
            kicked = _enforce_min_distance(kicked, 0.5)
            try:
                e, pos = relax(kicked, potential, rho)
                if e < best_energy:
                    best_energy = e
                    best_positions = pos.copy()
                chain_starts.append(pos.copy())
            except Exception:
                chain_starts.append(best_positions.copy())
        
        # Run basin hopping on each chain
        for chain_idx, chain_pos in enumerate(chain_starts):
            if time.time() - start_time > time_budget - 2.0:
                break
            
            chain_start = time.time()
            kick_scale = 0.25 + 0.1 * chain_idx
            
            # Adaptive kick: reduce over time within this chain
            max_iters = int(chain_budget / 0.3)  # estimate ~0.3s per iteration
            max_iters = max(3, min(max_iters, 50))
            
            for it in range(max_iters):
                if time.time() - start_time > time_budget - 2.0:
                    break
                if time.time() - chain_start > chain_budget:
                    break
                
                # Gradually reduce kick size
                current_kick = kick_scale * (1.0 - 0.4 * (it / max(max_iters - 1, 1)))
                
                kicked = chain_pos + np.random.randn(*best_positions.shape) * current_kick
                kicked = _enforce_min_distance(kicked, 0.5)
                
                try:
                    energy, positions = relax(kicked, potential, rho)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
                    # Accept if energy is lower or with probability (Metropolis)
                    if energy < chain_energy or np.random.random() < np.exp(-(energy - chain_energy) / (0.05 + 0.1 * it)):
                        chain_pos = positions.copy()
                        chain_energy = energy
                except Exception:
                    continue

    # Final polish: High-precision relaxation with tighter tolerance
    if best_positions is not None:
        # Try multiple small perturbations and take the best
        polish_candidates = [best_positions.copy()]
        
        # Add small random perturbations
        for _ in range(3):
            perturbed = best_positions + np.random.randn(*best_positions.shape) * 0.02
            perturbed = _enforce_min_distance(perturbed, 0.5)
            polish_candidates.append(perturbed)
        
        best_polish_energy = np.inf
        best_polish_pos = best_positions.copy()
        
        for cand in polish_candidates:
            try:
                e, pos = relax(cand, potential, rho)
                if e < best_polish_energy:
                    best_polish_energy = e
                    best_polish_pos = pos.copy()
            except Exception:
                continue
        
        if best_polish_energy < best_energy:
            best_energy = best_polish_energy
            best_positions = best_polish_pos

    best_positions = _enforce_min_distance(best_positions, 0.5)
    return best_positions
# EVOLVE-BLOCK-END
