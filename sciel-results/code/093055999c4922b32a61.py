import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import scipy.optimize

def _min_distance_check(pos):
    """Check if all pairwise distances are at least 0.5."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _generate_fcc_cluster(n, rng):
    """Generate an FCC cluster of approximately n atoms, centered."""
    # FCC lattice points
    # Basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # We generate a cubic block of FCC and take the n closest to center
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    pts = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                for b in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i, j, k]) + np.array(b)
                    pts.append(p)
    
    pts = np.array(pts)
    # Center
    pts -= np.mean(pts, axis=0)
    # Sort by distance from center
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices[:n]]
    # Add small noise to break symmetry
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def _generate_hcp_cluster(n, rng):
    """Generate an HCP-like cluster."""
    # HCP layers are hexagonal
    # This is a rough approximation. We use a hexagonal grid and stack layers.
    pts = []
    # Layer spacing ~ sqrt(2/3) for ideal HCP, but let's use 1.0 for simplicity in generation
    # Actually, for Morse, equilibrium distance is 1. 
    # Hexagonal packing in plane: distance 1.
    # Layer height: sqrt(2/3) * 1 ~ 0.816
    layer_height = np.sqrt(2.0/3.0)
    
    # Generate hexagonal grid points
    # Radius R such that approx n/2 atoms per layer (if 2 layers) or more
    # Let's just generate a large hex grid and pick closest to center
    max_r = int(np.ceil(np.sqrt(n))) + 2
    for i in range(-max_r, max_r+1):
        for j in range(-max_r, max_r+1):
            # Hex coords to Cartesian
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2.0
            # Check if within hexagon radius approx
            if np.linalg.norm([x, y]) <= max_r:
                pts.append([x, y, 0])
    
    pts = np.array(pts)
    # If we have enough, we can just take a slice. 
    # For a more structured approach, let's just use FCC for now as it's often competitive 
    # or generate a specific known structure if n matches.
    # For now, fallback to FCC if HCP generation is too complex or slow.
    # Actually, let's just return FCC for simplicity in this "different strategy" 
    # but ensure we have *more* diverse structured starts than the seed.
    return _generate_fcc_cluster(n, rng)

def _generate_icosahedron(n, rng):
    """Generate a Mackay icosahedron-like structure if n is close to a magic number."""
    # Magic numbers for icosahedra: 13, 55, 147, 309...
    # For n not matching, we can take a truncated icosahedron or just use FCC.
    # Let's implement a simple generator for the 13-atom icosahedron and scale/extend?
    # Too complex. Let's stick to FCC and BCC-like (cubic) which are robust.
    # BCC
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    pts = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                pts.append([i, j, k])
                # BCC center
                pts.append([i+0.5, j+0.5, k+0.5])
                
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices[:n]]
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def _generate_random_packed(n, rng):
    """Generate a random packed cluster using rejection sampling."""
    radius = n ** (1.0/3.0) * 1.2
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in pts):
                pts.append(p)
        attempts += 1
        
    if len(pts) < n:
        # Fill remaining with random
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p)
            
    return np.array(pts)

def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Strategy: Multi-start optimization using diverse, physics-informed structured seeds
    (FCC, BCC, Random Packed) followed by local relaxation.
    """
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance["potential"]
    
    start_time = time.time()
    # Safety margin: 300s limit. 
    # We have 6 instances. If called once per instance, we have ~300s total for all 6? 
    # "called once for each of six instances... all six calls together must finish within 300 s"
    # This implies the *sum* of execution times for all 6 calls is < 300s.
    # So each call has ~50s on average.
    # However, the evaluator calls construct() for each instance. 
    # We must assume each call has a time budget. 
    # If the evaluator runs them sequentially, and the total is 300s, 
    # we should aim for < 45s per call to be safe, or manage a global timer if possible?
    # The prompt says "called once for each... all six calls together must finish within 300s".
    # This suggests the harness measures the total time of the 6 calls.
    # To be safe, we will limit each individual call to ~45s, assuming they are run in parallel or sequentially with a global budget.
    # If they are sequential, 45s * 6 = 270s < 300s. This is safe.
    
    deadline = start_time + 45.0
    
    rng = np.random.default_rng(42 + n)
    
    # Generate a pool of diverse initial configurations
    # 1. FCC
    # 2. BCC
    # 3. Random Packed
    # 4. FCC with different noise
    # 5. Random Packed with different seed
    
    initial_configs = []
    
    # FCC
    try:
        cfg = _generate_fcc_cluster(n, rng)
        if _min_distance_check(cfg):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # BCC
    try:
        cfg = _generate_icosahedron(n, rng) # This is actually BCC in my impl
        if _min_distance_check(cfg):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # Random Packed
    try:
        cfg = _generate_random_packed(n, rng)
        if _min_distance_check(cfg):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # More FCC with larger noise
    try:
        cfg = _generate_fcc_cluster(n, rng)
        cfg += rng.normal(0, 0.2, size=cfg.shape)
        if _min_distance_check(cfg):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # More Random Packed
    try:
        cfg = _generate_random_packed(n, rng)
        if _min_distance_check(cfg):
            initial_configs.append(cfg)
    except Exception:
        pass

    if not initial_configs:
        # Fallback
        cfg = rng.uniform(-n**0.33, n**0.33, size=(n, 3))
        initial_configs.append(cfg)

    # Relax all initial configurations
    best_energy = np.inf
    best_positions = initial_configs[0]
    
    for cfg in initial_configs:
        if time.time() > deadline - 5.0:
            break
        try:
            energy, positions = relax(cfg, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
            
    # Basin Hopping / Local Search from the best found
    # We use a smaller kick size now because we are likely in a good basin.
    kick_sigma = 0.05
    current_energy = best_energy
    current_positions = best_positions
    
    # Time remaining for local search
    while time.time() < deadline - 2.0:
        # Perturb
        new_positions = current_positions + rng.normal(0, kick_sigma, size=(n, 3))
        
        # Check min distance
        if not _min_distance_check(new_positions):
            # Try to repair by scaling? Or just reject.
            # Reject is safer.
            continue
            
        # Relax
        try:
            energy, positions = relax(new_positions, potential, rho)
        except Exception:
            continue
            
        # Acceptance: Greedy (since we want global min and have limited time, 
        # and our starts are good, we can be greedy to converge faster)
        # But a little stochasticity helps escape shallow local minima.
        if energy < current_energy:
            current_energy = energy
            current_positions = positions
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        else:
            # Metropolis with low T
            T = 0.05
            delta_e = energy - current_energy
            if np.random.rand() < np.exp(-delta_e / T):
                current_energy = energy
                current_positions = positions

    return best_positions
