"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 38

# EVOLVE-BLOCK-START
def _lj_energy(pos):
    """Calculate total Lennard-Jones energy."""
    # pos shape (N, 3)
    # Compute pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    # Avoid division by zero (self-interaction)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # LJ potential 4 * (r^-12 - r^-6)
    # Sum over all pairs (i, j) where i < j
    # dist matrix is symmetric, diagonal is 0 (set to 1)
    # We can sum the upper triangle
    iu = np.triu_indices(N, k=1)
    r = dist[iu]
    r6 = r ** -6
    energy = 4.0 * np.sum(r6 * r6 - r6)
    return energy

def _construct_fcc_start():
    """Generate FCC truncated cluster start."""
    # Generate FCC lattice points in units of a/2
    # Condition: nx + ny + nz is even
    points = []
    # Range needs to be large enough to get 38 points
    # Closest points:
    # d2=0: (0,0,0) -> 1
    # d2=2: (1,1,0) perms -> 12
    # d2=4: (2,0,0) perms -> 6
    # d2=6: (2,1,1) perms -> 24
    # Total 1+12+6+24 = 43. We need 38.
    # So we take the 38 closest.
    
    for nx in range(-3, 4):
        for ny in range(-3, 4):
            for nz in range(-3, 4):
                if (nx + ny + nz) % 2 == 0:
                    d2 = nx*nx + ny*ny + nz*nz
                    points.append((d2, nx, ny, nz))
    
    # Sort by distance squared
    points.sort(key=lambda x: x[0])
    
    # Take first 38
    selected = points[:N]
    
    # Scale factor:
    # Nearest neighbor distance in lattice units (a/2) is sqrt(2).
    # We want physical distance r_min = 2^(1/6).
    # So sqrt(2) * scale = 2^(1/6)
    # scale = 2^(1/6) / sqrt(2) = 2^(1/6) / 2^(1/2) = 2^(1/6 - 3/6) = 2^(-2/6) = 2^(-1/3)
    scale = 2.0 ** (-1.0 / 3.0)
    
    coords = np.array([[nx, ny, nz] for _, nx, ny, nz in selected], dtype=float)
    coords *= scale
    
    return coords

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Initial guess: FCC truncated octahedron
    pos0 = _construct_fcc_start()
    
    # Relaxation
    # L-BFGS-B is robust and handles the smooth LJ potential well
    res = minimize(
        _lj_energy, 
        pos0, 
        method='L-BFGS-B', 
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
    )
    
    elapsed = time.time() - start_time
    if elapsed < deadline and res.success:
        return res.x
    
    # If relaxation failed or took too long, return the initial FCC structure
    # which is already very close to the global minimum
    return pos0
# EVOLVE-BLOCK-END
