"""Morse cluster global minima search with high-throughput multi-start and structured initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline: evaluator kills after 330s wall-clock.
# We use a shared deadline across all six construct() calls.
# The first call sets the deadline; subsequent calls check against it.
_DEADLINE = time.time() + 310.0  # 20s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _morse_energy(pos, rho):
    """Compute total Morse energy: sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij))-2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    u = rho * (1.0 - dist)
    e = np.exp(u)
    pair_e = e * (e - 2.0)
    return 0.5 * np.sum(pair_e)

def _generate_fcc_cluster(n, spacing=1.0, rng=None):
    """
    Generate a compact FCC-like cluster.
    Much faster than rejection sampling.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    # Estimate side length of cubic lattice needed
    # Density of FCC is 4 atoms per unit cell (volume 1).
    # Approximate volume V = n / 4. Side ~ (V)^(1/3)
    # We generate a slightly larger box and crop
    side = int(np.ceil((n / 4.0) ** (1.0/3.0))) + 2
    
    coords = []
    # Generate points in a cubic grid
    # To ensure compactness, we generate a sphere of lattice points
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                # Actually, simpler to just use a simple cubic lattice with spacing 0.707 (1/sqrt(2))
                # or just use a simple cubic grid with spacing 1.0 and let relax fix it.
                # Simple cubic is fine for initialization.
                coords.append([i * spacing, j * spacing, k * spacing])
                
                if len(coords) >= n:
                    break
            if len(coords) >= n:
                break
        if len(coords) >= n:
            break
    
    points = np.array(coords[:n])
    
    # Center
    points -= np.mean(points, axis=0)
    
    # Perturb to break symmetry and avoid exact degeneracies
    points += rng.normal(0, 0.1, points.shape)
    
    return points

def _generate_bcc_cluster(n, spacing=1.0, rng=None):
    """Generate a BCC-like cluster."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    side = int(np.ceil((n / 2.0) ** (1.0/3.0))) + 2
    coords = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                # BCC basis: (0,0,0) and (0.5,0.5,0.5)
                coords.append([i * spacing, j * spacing, k * spacing])
                coords.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
                
                if len(coords) >= n:
                    break
            if len(coords) >= n:
                break
        if len(coords) >= n:
            break
            
    points = np.array(coords[:n])
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.1, points.shape)
    return points

def _generate_random_compact(n, radius, rng=None):
    """
    Generate n random points within a sphere, ensuring they are not overlapping.
    Uses a vectorized rejection sampling approach for speed.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    min_dist = 0.51 # Slightly more than constraint 0.5
    
    # Estimate a good radius. For n points in a sphere, typical radius R ~ (3n/4pi)^(1/3) * spacing
    # But we are given radius. Let's stick to the provided radius but ensure it's large enough.
    
    points = np.empty((0, 3))
    max_attempts = 50000
    attempts = 0
    
    while len(points) < n and attempts < max_attempts:
        # Generate a batch of candidates
        batch_size = max(10, n - len(points))
        candidates = rng.uniform(-radius, radius, (batch_size, 3))
        
        # Filter by radius
        norms = np.linalg.norm(candidates, axis=1)
        candidates = candidates[norms <= radius]
        
        if len(candidates) == 0:
            attempts += 1
            continue
            
        # Check against existing points
        if len(points) > 0:
            # Compute distances between candidates and existing points
            # dists shape: (batch, existing)
            dists = np.linalg.norm(candidates[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
            min_dists = np.min(dists, axis=1)
            valid_mask = min_dists >= min_dist
        else:
            valid_mask = np.ones(len(candidates), dtype=bool)
            
        valid_candidates = candidates[valid_mask]
        
        # If we have valid candidates, we need to check them against each other
        # This is the expensive part. For small batches, it's fast.
        if len(valid_candidates) > 0:
             # Greedy selection from valid_candidates to maintain min_dist
             selected = []
             for p in valid_candidates:
                 if len(selected) == 0:
                     selected.append(p)
                 else:
                     d = np.linalg.norm(np.array(selected) - p, axis=1)
                     if np.min(d) >= min_dist:
                         selected.append(p)
                 
                 if len(points) + len(selected) >= n:
                     break
             
             points = np.vstack([points, np.array(selected)])
        
        attempts += 1
        # If we are stuck (no progress), increase radius slightly or break
        if attempts > 100 and len(points) < n:
             # Fallback: just add random points, relax will fix overlaps
             # This is dangerous but better than timeout
             while len(points) < n:
                 p = rng.uniform(-radius, radius, 3)
                 if np.linalg.norm(p) <= radius * 1.5: # Allow slightly larger radius
                     points = np.vstack([points, p])
             break
             
    return points[:n]

def _generate_icosahedral_shell(n, radius=1.0, rng=None):
    """Generate n points distributed on a spherical shell using Fibonacci sphere."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * i  # Golden angle
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    return np.column_stack([x, y, z])

def construct(instance):
    """Find low-energy arrangement of n atoms with Morse potential."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget management
    # We want to spend as much time as possible on relax.
    # Estimate relax time: ~0.1s to 0.5s for n=80.
    # We will loop until time runs out.
    
    start_time = time.time()
    total_time_allowed = _DEADLINE - start_time
    
    # If this is the first call, we have ~310s.
    # If it's later, we have less.
    # We don't know which call we are, so we just use what's left.
    
    rng = np.random.default_rng(size * 100 + 42) # Deterministic but varied
    
    best_energy = np.inf
    best_positions = None
    
    # Estimate a reasonable radius for random starts
    # Typical cluster radius R ~ n^(1/3)
    base_radius = size ** (1.0/3.0)
    
    # Strategy:
    # 1. Try a few high-quality structured starts (FCC, BCC, Icosa)
    # 2. Then do many random compact starts
    
    structured_starts = []
    
    # FCC
    structured_starts.append(_generate_fcc_cluster(size, 1.0, rng))
    # BCC
    structured_starts.append(_generate_bcc_cluster(size, 1.0, rng))
    # Icosahedral shell (might be too sparse, but good for surface)
    structured_starts.append(_generate_icosahedral_shell(size, base_radius * 1.5, rng))
    
    # Relax structured starts
    for i, pos in enumerate(structured_starts):
        if time.time() > _DEADLINE - 5.0:
            break
        try:
            energy, relaxed_pos = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
        except Exception:
            pass
    
    # Random restarts
    # We will generate starts in a loop.
    # We want to maximize the number of relax calls.
    
    while time.time() < _DEADLINE - 2.0:
        # Generate a random compact start
        # Vary the radius slightly to explore different densities
        radius_factor = 0.8 + 0.4 * rng.random()
        radius = base_radius * radius_factor
        
        try:
            pos = _generate_random_compact(size, radius, rng)
        except Exception:
            # If generation fails or is too slow, use a simple random ball
            pos = rng.uniform(-base_radius, base_radius, (size, 3))
        
        try:
            energy, relaxed_pos = relax(pos, potential, parameter)
            
            # Check validity (min separation)
            if len(relaxed_pos) >= 2:
                diff = relaxed_pos[:, np.newaxis, :] - relaxed_pos[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(dist, np.inf)
                if np.min(dist) < 0.49:
                    continue
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                
        except Exception:
            continue
    
    # Fallback if no valid result
    if best_positions is None:
        best_positions = _generate_fcc_cluster(size, 1.0, rng)
        
    return best_positions
# EVOLVE-BLOCK-END
