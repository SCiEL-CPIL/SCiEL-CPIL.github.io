import numpy as np
import time
import sys

# Try to import science_chemistry, but handle if it's not available in the test environment 
# (though the prompt says it is available)
try:
    import science_chemistry
    HAS_RELAX = True
except ImportError:
    HAS_RELAX = False

N = 38

def _lj_energy(pos):
    """Compute Lennard-Jones energy for a set of positions."""
    # pos shape (N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Mask out self-interactions and very close pairs to avoid inf
    mask = dist_sq > 1e-12
    dist_sq = dist_sq[mask]
    
    if len(dist_sq) == 0:
        return 0.0
        
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    return energy

def _generate_fcc_cluster(n_atoms):
    """Generate the n_atoms closest points in an FCC lattice to the origin."""
    # FCC lattice constant a such that nearest neighbor distance is 2^(1/6)
    # Nearest neighbor distance in FCC is a / sqrt(2)
    # So a = 2^(1/6) * sqrt(2) = 2^(1/6) * 2^(1/2) = 2^(2/3)
    a = 2**(2.0/3.0)
    
    # We need to generate enough points.
    # The 38th atom is in the 3rd shell (distance a*sqrt(2)).
    # Max coordinate needed is roughly 2*a.
    max_coord = int(np.ceil(3.0 * a))
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC condition: i+j+k is even
                if (i + j + k) % 2 == 0:
                    points.append((i * a, j * a, k * a))
    
    points = np.array(points)
    dists = np.sum(points**2, axis=1)
    
    # Sort by distance from origin
    # If distances are equal, the order depends on the initial array order.
    # To make it deterministic and symmetric, we might want a better tie-breaker,
    # but for relaxation, any FCC-like start should work.
    sorted_indices = np.argsort(dists)
    
    return points[sorted_indices[:n_atoms]]

def construct():
    start_time = time.time()
    # 150s limit. Use 140s to be safe.
    deadline = start_time + 140.0
    
    # Initial guess: FCC cluster
    initial_pos = _generate_fcc_cluster(N)
    best_pos = initial_pos
    best_energy = _lj_energy(initial_pos)
    
    if not HAS_RELAX:
        return best_pos

    # Strategy: Relax the initial FCC guess
    try:
        relaxed_pos = science_chemistry.relax(initial_pos)
        e = _lj_energy(relaxed_pos)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed_pos
    except Exception:
        pass

    if time.time() > deadline:
        return best_pos

    # Strategy 2: Multi-start from a larger FCC cluster (43 atoms)
    # Remove 5 atoms randomly from the outer shell and relax.
    # This helps if the initial "first 38" selection was biased into a local minimum
    # (though the FCC global minimum is robust).
    
    # Generate 43-atom cluster (1+12+6+24)
    pos43 = _generate_fcc_cluster(43)
    
    # The outer shell is indices 19 to 42 (24 atoms)
    outer_indices = np.arange(19, 43)
    
    # Try a few random removals
    num_tries = 10
    for i in range(num_tries):
        if time.time() > deadline - 5.0: # Stop 5s before deadline
            break
            
        # Remove 5 random atoms from the outer shell
        remove_idx = np.random.choice(outer_indices, size=5, replace=False)
        keep_mask = np.ones(43, dtype=bool)
        keep_mask[remove_idx] = False
        pos_candidate = pos43[keep_mask]
        
        # Optional: Add small perturbation to help escape shallow local mins
        # But FCC is usually a deep basin.
        
        try:
            relaxed_pos = science_chemistry.relax(pos_candidate)
            e = _lj_energy(relaxed_pos)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed_pos
        except Exception:
            pass

    return best_pos
