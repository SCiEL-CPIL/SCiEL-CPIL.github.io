import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import scipy.optimize

def _min_distance_check(pos):
    """Check if all pairwise distances are at least 0.5."""
    n = len(pos)
    # Use a simple loop for small n, or vectorized for larger
    # Vectorized distance matrix is O(N^2) memory, fine for N=80
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _generate_initial_config(n, rng, mode='sphere'):
    """Generate an initial configuration of n points."""
    if mode == 'sphere':
        # Uniform in sphere
        radius = n ** (1.0/3.0) * 0.8
        pts = []
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                # Simple rejection for min distance during generation to avoid bad starts
                if all(np.linalg.norm(p - q) >= 0.5 for q in pts):
                    pts.append(p)
        return np.array(pts)
    elif mode == 'fcc':
        # Generate FCC lattice points
        # FCC basis: (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
        # Scale to fit n points
        # Approximate volume per atom in FCC is 1.0 (if a=1, 4 atoms in cell of vol 1, so 0.25/atom)
        # Let's just generate a grid and take first n
        side = int(np.ceil(n ** (1.0/3.0))) + 2
        pts = []
        for i in range(-side, side+1):
            for j in range(-side, side+1):
                for k in range(-side, side+1):
                    for b in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                        p = np.array([i, j, k]) + np.array(b)
                        pts.append(p)
                    if len(pts) >= n:
                        break
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        pts = np.array(pts[:n])
        # Center it
        pts -= np.mean(pts, axis=0)
        # Add small noise
        pts += rng.normal(0, 0.1, size=pts.shape)
        return pts
    else:
        return _generate_initial_config(n, rng, 'sphere')

def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Uses a basin-hopping-like strategy with diverse initializations and local relaxation.
    """
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance["potential"]
    
    start_time = time.time()
    # Safety margin: 300s limit, use 45s per instance
    deadline = start_time + 45.0
    
    rng = np.random.default_rng(42 + n)
    
    # 1. Generate diverse initial configurations
    # Modes: sphere, fcc, bcc (approx), random
    modes = ['sphere', 'fcc', 'sphere', 'fcc', 'sphere', 'fcc', 'sphere', 'fcc']
    initial_configs = []
    for mode in modes:
        if time.time() > deadline - 10.0:
            break
        try:
            cfg = _generate_initial_config(n, rng, mode)
            if len(cfg) == n and _min_distance_check(cfg):
                initial_configs.append(cfg)
        except Exception:
            pass
            
    if not initial_configs:
        # Fallback: simple random
        cfg = rng.uniform(-n**0.33, n**0.33, size=(n, 3))
        initial_configs.append(cfg)

    # 2. Relax all initial configurations
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
            
    # 3. Basin Hopping Loop
    # Perturb, check, relax, accept
    kick_sigma = 0.1
    current_energy = best_energy
    current_positions = best_positions
    
    while time.time() < deadline - 2.0:
        # Perturb
        # Random displacement
        new_positions = current_positions + rng.normal(0, kick_sigma, size=(n, 3))
        
        # Check min distance
        if not _min_distance_check(new_positions):
            # Try to repair by pushing apart or just reject
            # Rejection is safer and faster
            continue
            
        # Relax
        try:
            energy, positions = relax(new_positions, potential, rho)
        except Exception:
            continue
            
        # Acceptance Criterion
        # Use Metropolis: accept if E_new < E_old, or with probability exp(-(E_new-E_old)/T)
        # Since we want global min and have limited time, we can use a simple greedy or slight stochasticity
        # Let's use a simple Metropolis with T=0.01
        if energy < current_energy:
            current_energy = energy
            current_positions = positions
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                # Optionally reduce kick size or keep it
        else:
            delta_e = energy - current_energy
            # T decays slightly? Or fixed? Fixed is simpler.
            T = 0.01
            if np.random.rand() < np.exp(-delta_e / T):
                current_energy = energy
                current_positions = positions

    return best_positions
