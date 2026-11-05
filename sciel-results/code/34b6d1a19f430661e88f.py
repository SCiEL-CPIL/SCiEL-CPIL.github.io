"""Program for Dzugutov cluster global minima using Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def _enforce_min_distance(positions, min_dist=0.5, max_iter=100):
    """
    Simple iterative method to ensure all pairs are at least min_dist apart.
    Returns True if successful, False if stuck (though unlikely with small clusters).
    """
    n = len(positions)
    positions = positions.copy()
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                if dist < min_dist:
                    if dist < 1e-10:
                        # Random push
                        d = np.random.normal(size=3)
                        dist = np.linalg.norm(d)
                    # Push apart to exactly min_dist
                    correction = (d / dist) * (min_dist - dist) / 2.0
                    positions[i] += correction
                    positions[j] -= correction
                    moved = True
        if not moved:
            break
    return positions

def construct(instance):
    """
    Construct low-energy Dzugutov cluster using Basin Hopping.
    """
    n = instance["n"]
    potential = instance["potential"]
    
    # Time budget per instance (approx 45s to be safe, total 300s / 6)
    time_limit = 45.0
    start_time = time.time()
    
    # Initial guess: Scaled icosahedron or Dense Random
    # Dzugutov optimal pair distance is around 1.0-1.1? 
    # Let's estimate. V(r) min.
    # A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16
    # At r=1.0: A(1 - 1.28)exp(1.1/(-0.87)) = 5.82(-0.28)exp(-1.26) ~ -1.6 * 0.28 ~ -0.45
    # At r=1.1: A(1/1.1^16 - 1.28)exp(1.1/(-0.77)) = 5.82(0.167 - 1.28)exp(-1.42) ~ 5.82(-1.11)0.24 ~ -1.55
    # At r=1.2: A(1/1.2^16 - 1.28)exp(1.1/(-0.67)) = 5.82(0.104 - 1.28)exp(-1.64) ~ 5.82(-1.17)0.19 ~ -1.3
    # So optimal pair distance is around 1.1.
    # Cluster radius for n atoms with d~1.1 is roughly (n * 1.1^3)^(1/3) ~ n^(1/3) * 1.1
    
    radius = 1.2 * n ** (1.0/3.0)
    
    # Generate initial positions: Dense random packing
    rng = np.random.default_rng(42)
    init_pos = rng.uniform(-radius, radius, (n, 3))
    init_pos = _enforce_min_distance(init_pos)
    
    # Relax initial
    try:
        E_curr, pos_curr = relax(init_pos, potential)
    except Exception:
        E_curr, pos_curr = relax(init_pos, "dzugutov")
        
    E_best = E_curr
    pos_best = pos_curr.copy()
    
    # Basin Hopping Parameters
    T_start = 10.0
    T_end = 0.1
    T_cooling = 0.995
    
    # We will run until time limit or convergence
    T = T_start
    sigma = 0.5
    
    while time.time() - start_time < time_limit:
        # Perturb
        noise = np.random.normal(0, sigma, (n, 3))
        pos_new = pos_curr + noise
        pos_new = _enforce_min_distance(pos_new)
        
        # Relax
        try:
            E_new, pos_relaxed = relax(pos_new, potential)
        except Exception:
            E_new, pos_relaxed = relax(pos_new, "dzugutov")
            
        # Acceptance criterion (Metropolis)
        delta_E = E_new - E_curr
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            pos_curr = pos_relaxed
            E_curr = E_new
            
            if E_new < E_best:
                E_best = E_new
                pos_best = pos_relaxed.copy()
        
        # Cooling
        T *= T_cooling
        if T < T_end:
            T = T_end
            # Restart with new random init if stuck or to explore more
            # Simple restart:
            init_pos = rng.uniform(-radius, radius, (n, 3))
            init_pos = _enforce_min_distance(init_pos)
            try:
                E_curr, pos_curr = relax(init_pos, potential)
            except Exception:
                E_curr, pos_curr = relax(init_pos, "dzugutov")
            T = T_start * 0.5 # Restart with lower T to focus
            sigma = 0.3
            
        # Adjust sigma based on T
        sigma = 0.1 + 0.5 * (T - T_end) / (T_start - T_end)
        
        # Check for convergence (optional, but time limit is primary)
        # If E_best hasn't improved in a long time, we could restart, 
        # but the time limit handles the overall stop.
        
    return pos_best
# EVOLVE-BLOCK-END
