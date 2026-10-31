"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _lj_energy(positions):
    """Compute total Lennard-Jones energy for a set of positions."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    # Subtract self-interaction (diagonal was set to 1, contributing 4*(1-1)=0, so no correction needed)
    return energy


def _generate_fcc_truncated_octahedron():
    """Generate 38 atoms in FCC truncated octahedron geometry."""
    # Generate FCC lattice points
    # FCC primitive vectors with a=1 (we'll scale later)
    a = 1.0
    v1 = np.array([a/2, a/2, 0.0])
    v2 = np.array([a/2, 0.0, a/2])
    v3 = np.array([0.0, a/2, a/2])
    
    points = []
    # Generate a range of integer combinations
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                p = i * v1 + j * v2 + k * v3
                points.append(p)
    
    points = np.array(points)
    
    # Sort by distance from origin
    dists = np.sum(points ** 2, axis=1)
    order = np.argsort(dists)
    points = points[order]
    
    # Take the first 38
    points = points[:N]
    
    return points


def construct():
    # Generate FCC truncated octahedron
    positions = _generate_fcc_truncated_octahedron()
    
    # The natural scale: nearest neighbor distance in FCC with a=1 is a/sqrt(2) = 0.7071
    # We want nearest neighbor ~1.1225, so scale by 1.1225 / 0.7071 ≈ 1.5874
    # But let's optimize the scale
    scale = 1.5874
    positions = positions * scale
    
    # Do a local relaxation to fine-tune
    # Use L-BFGS to minimize the LJ energy
    def energy_func(x):
        pos = x.reshape(-1, 3)
        # Check minimum distance constraint
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, 1e10)
        min_r2 = np.min(r2)
        if min_r2 < 0.25:  # r < 0.5
            return 1e10
        return _lj_energy(pos)
    
    x0 = positions.flatten()
    result = minimize(energy_func, x0, method='L-BFGS-B', 
                      options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
    
    positions = result.x.reshape(-1, 3)
    
    # Final check: ensure minimum distance >= 0.5
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1e10)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        # Fallback: scale up slightly
        positions = positions * (0.5 / min_r)
    
    return positions
# EVOLVE-BLOCK-END
