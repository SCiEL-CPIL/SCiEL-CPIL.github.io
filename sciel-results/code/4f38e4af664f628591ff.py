import numpy as np
import science_chemistry

N = 55

def _calc_energy(pos):
    """Calculate LJ energy for positions pos (N,3)."""
    # Compute pairwise squared distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Set diagonal to 1 to avoid division by zero, term becomes 0
    np.fill_diagonal(dist_sq, 1.0)
    
    r6 = dist_sq ** (-3.0)
    r12 = r6 ** 2.0
    
    # Sum over all i,j. Diagonal is 0.
    # Sum_{i!=j} = 2 * Sum_{i<j}
    # E = 4 * Sum_{i<j} (...) = 2 * Sum_{i!=j} (...)
    return 2.0 * np.sum(r12 - r6)

def construct():
    best_pos = None
    best_energy = np.inf
    num_restarts = 25
    max_attempts = 200
    
    for _ in range(num_restarts):
        for _ in range(max_attempts):
            # Generate random points in a cube [-3, 3]^3
            pos = np.random.uniform(-3.0, 3.0, size=(N, 3))
            
            # Check min distance >= 0.5
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dist_sq, np.inf)
            
            if np.min(dist_sq) >= 0.25: # 0.5^2
                try:
                    relaxed_pos = science_chemistry.relax(pos)
                    e = _calc_energy(relaxed_pos)
                    if e < best_energy:
                        best_energy = e
                        best_pos = relaxed_pos
                except Exception:
                    pass
                break # Try next restart if this one succeeded in generating valid start
    
    if best_pos is None:
        # Fallback: Simple BCC or FCC fragment
        # Let's use a simple grid and relax
        coords = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if len(coords) < N:
                        coords.append([x, y, z])
        pos = np.array(coords, dtype=float)
        # Scale to min dist ~ 1.0 (grid spacing)
        # LJ min dist is 2^(1/6) ~ 1.12
        pos *= 2.0 ** (1.0/6.0)
        try:
            pos = science_chemistry.relax(pos)
        except Exception:
            pass
        return pos
        
    return best_pos
