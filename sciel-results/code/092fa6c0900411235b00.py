"""LJ38 cluster optimization using symmetry-adapted parameterization of the FCC truncated octahedron."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

DEADLINE = 145.0  # Safety margin before 150s kill
START_TIME = time.time()

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= (min_dist * min_dist)

def make_fcc_truncated_octahedron():
    """
    Create an FCC truncated octahedron cluster of 38 atoms.
    We generate points in the FCC lattice and select the 38 closest to origin.
    """
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC condition: i+j+k is even
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    if r2 <= 30:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    return np.array(selected, dtype=float)

def get_fcc_shell_structure():
    """
    Return the FCC lattice points for the truncated octahedron, grouped by shell.
    Shell 0: center (1 atom)
    Shell 1: 12 atoms at r2=2 (FCC nearest neighbors)
    Shell 2: 6 atoms at r2=4 (FCC next-nearest)
    Shell 3: 12 atoms at r2=6 (FCC 3rd neighbors)
    Shell 4: 7 atoms at r2=8 (FCC 4th neighbors) -- wait, let's verify counts.
    
    FCC points (i,j,k) with i+j+k even:
    r2=0: (0,0,0) -> 1 atom
    r2=2: (1,1,0) and perms/signs -> 12 atoms
    r2=4: (2,0,0) and perms/signs -> 6 atoms
    r2=6: (2,1,1) and perms/signs -> 12 atoms
    r2=8: (2,2,0) and perms/signs -> 12 atoms
    r2=12: (2,2,2) -> 8 atoms
    r2=14: (3,2,1) and perms/signs -> 24 atoms
    
    Total so far: 1+12+6+12 = 31. Need 7 more.
    Next shell is r2=8 (12 atoms). We need 7 of them.
    Which 7? The truncated octahedron has 8 vertices of type (2,2,0) and 6 of type (2,0,0)? No.
    Actually, the standard FCC 38-atom cluster is 1+12+6+12+7.
    The 7 atoms from the r2=8 shell are typically chosen to maintain high symmetry.
    The r2=8 points are (±2, ±2, 0) and perms. There are 12 such points.
    A common selection is to take all 6 of (±2,0,0) type? No, those are r2=4.
    The points at r2=8 are (±2, ±2, 0), (±2, 0, ±2), (0, ±2, ±2).
    To get 7, one might take all 12 and then remove 5? Or maybe the structure is different.
    
    Let's just generate all 38 points by sorting by distance and taking the first 38.
    This ensures we get the most compact cluster.
    """
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    if r2 <= 30:
                        points.append((i, j, k, r2))
    
    points.sort(key=lambda x: x[3])
    # Take first 38
    selected_int = [p[:3] for p in points[:38]]
    
    # Group by r2
    from collections import defaultdict
    shells = defaultdict(list)
    for p in selected_int:
        r2 = p[0]**2 + p[1]**2 + p[2]**2
        shells[r2].append(p)
        
    return shells, selected_int

def parameterize_fcc_cluster(params):
    """
    params: array of shape (4,)
    params[0]: global scale factor
    params[1]: strain in x direction
    params[2]: strain in y direction
    params[3]: strain in z direction
    
    We apply an affine transformation to the ideal FCC lattice points.
    """
    scale = params[0]
    sx = params[1]
    sy = params[2]
    sz = params[3]
    
    # Get the integer lattice points for the 38-atom cluster
    # We precompute this to avoid overhead
    if not hasattr(parameterize_fcc_cluster, 'int_points'):
        max_coord = 4
        points = []
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                for k in range(-max_coord, max_coord + 1):
                    if (i + j + k) % 2 == 0:
                        r2 = i*i + j*j + k*k
                        if r2 <= 30:
                            points.append((i, j, k, r2))
        points.sort(key=lambda x: x[3])
        selected_int = [p[:3] for p in points[:38]]
        parameterize_fcc_cluster.int_points = np.array(selected_int, dtype=float)
        
    int_points = parameterize_fcc_cluster.int_points
    
    # Ideal scale: 1/sqrt(2)
    base_scale = 1.0 / math.sqrt(2.0)
    
    # Apply scale and strain
    positions = int_points * (base_scale * scale)
    positions[:, 0] *= sx
    positions[:, 1] *= sy
    positions[:, 2] *= sz
    
    return positions

def energy_from_params(params):
    """Compute LJ energy from parameterized FCC cluster."""
    positions = parameterize_fcc_cluster(params)
    if not check_validity(positions, min_dist=0.5):
        return 1e20
    return lj_energy_vectorized(positions)

def construct():
    """
    Main entry point: Optimize the symmetry-adapted FCC parameterization.
    """
    global START_TIME
    START_TIME = time.time()
    
    # Initial guess: scale=1, strains=1
    x0 = np.array([1.0, 1.0, 1.0, 1.0])
    
    # Optimize using L-BFGS-B
    try:
        result = minimize(energy_from_params, x0, method='L-BFGS-B', 
                         options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10})
        best_params = result.x
        best_energy = result.fun
    except Exception:
        best_params = x0
        best_energy = energy_from_params(x0)
        
    # Get the positions
    best_pos = parameterize_fcc_cluster(best_params)
    
    # Final relaxation in Cartesian space to break any remaining symmetry constraints
    # This allows the cluster to relax into the exact global minimum
    def energy_func(pos):
        pos = pos.reshape(N, 3)
        return lj_energy_vectorized(pos)
        
    try:
        result2 = minimize(energy_func, best_pos.flatten(), method='L-BFGS-B', 
                          options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12})
        final_pos = result2.x.reshape(N, 3)
        final_energy = lj_energy_vectorized(final_pos)
        
        if final_energy < best_energy and check_validity(final_pos):
            best_pos = final_pos
            best_energy = final_energy
    except Exception:
        pass
        
    # Fallback if something went wrong
    if not check_validity(best_pos):
        best_pos = make_fcc_truncated_octahedron()
        
    return best_pos

# EVOLVE-BLOCK-END
