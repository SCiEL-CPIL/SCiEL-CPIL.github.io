"""Construct LJ38 global minimum: FCC truncated octahedron with optimized scale and shell relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar, minimize


def _generate_fcc_core():
    """
    Generate the 27 core atoms of the FCC cluster (shells 1 and 2).
    Shell 1: 12 atoms at dist a/sqrt(2)
    Shell 2: 14 atoms (8 at a*sqrt(3)/2, 6 at a)
    Total: 27 atoms.
    """
    a = np.sqrt(2)  # Lattice constant such that NN distance = 1
    
    points = []
    # Center
    points.append([0.0, 0.0, 0.0])
    
    # Shell 1: 12 atoms at (±0.5, ±0.5, 0) * a
    for i in [-0.5, 0.5]:
        for j in [-0.5, 0.5]:
            points.append([i * a, j * a, 0.0])
            points.append([i * a, 0.0, j * a])
            points.append([0.0, i * a, j * a])
    
    # Shell 2: 14 atoms
    # 8 atoms at (±0.5, ±0.5, ±0.5) * a
    for i in [-0.5, 0.5]:
        for j in [-0.5, 0.5]:
            for k in [-0.5, 0.5]:
                points.append([i * a, j * a, k * a])
                
    # 6 atoms at (±1, 0, 0) * a
    for i in [-1.0, 1.0]:
        points.append([i * a, 0.0, 0.0])
        points.append([0.0, i * a, 0.0])
        points.append([0.0, 0.0, i * a])
        
    return np.array(points)


def _generate_fcc_shell3_candidates():
    """
    Generate the 24 candidate atoms for the 3rd shell.
    These are at (±1, ±0.5, ±0.5) * a and permutations.
    """
    a = np.sqrt(2)
    candidates = []
    
    # Permutations of (1, 0.5, 0.5)
    # x=±1, y=±0.5, z=±0.5
    for sx in [-1.0, 1.0]:
        for sy in [-0.5, 0.5]:
            for sz in [-0.5, 0.5]:
                candidates.append([sx * a, sy * a, sz * a])
                candidates.append([sy * a, sx * a, sz * a])
                candidates.append([sx * a, sz * a, sy * a])
                
    return np.array(candidates)


def _lj_energy(positions):
    """Compute total LJ energy for a set of positions."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    # Avoid division by zero
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    r12 = r6 ** 2
    # Sum over all pairs (each pair counted twice, so divide by 2)
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6) / 2.0
    return energy


def construct():
    # 1. Generate core (27 atoms)
    core = _generate_fcc_core()
    
    # 2. Generate candidates for shell 3 (24 atoms)
    candidates = _generate_fcc_shell3_candidates()
    
    # 3. Select 11 candidates to form the initial 38-atom cluster.
    # We pick the first 11. The optimization will relax them into the correct symmetric positions.
    initial_outer = candidates[:11]
    
    # 4. Combine core and initial outer
    positions = np.vstack([core, initial_outer])
    
    # 5. Optimize the positions of the 11 outer atoms to minimize energy,
    #    keeping the 27 core atoms fixed.
    #    This fixes the symmetry breaking caused by the arbitrary selection of 11 from 24.
    
    def energy_fn_outer(x):
        # x is a flattened array of 33 coordinates (11 atoms * 3)
        outer = x.reshape(11, 3)
        full_pos = np.vstack([core, outer])
        return _lj_energy(full_pos)
    
    # Initial guess for outer atoms
    x0 = initial_outer.flatten()
    
    # Use L-BFGS for local optimization
    # This should be fast (33 variables)
    res = minimize(energy_fn_outer, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
    
    optimal_outer = res.x.reshape(11, 3)
    base_points = np.vstack([core, optimal_outer])
    
    # 6. Optimize the global scale factor
    def energy_fn_scale(s):
        return _lj_energy(base_points * s)
    
    result = minimize_scalar(energy_fn_scale, bounds=(0.9, 1.3), method='bounded', 
                             options={'xatol': 1e-10, 'maxiter': 1000})
    
    optimal_scale = result.x
    final_positions = base_points * optimal_scale
    
    # Verify minimum separation >= 0.5
    diff = final_positions[:, np.newaxis, :] - final_positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        # Fallback: use the seed program's approach
        side = int(np.ceil(N ** (1 / 3)))
        pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        pts.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        final_positions = np.array(pts[:N], dtype=float) * 1.12
    
    return final_positions
# EVOLVE-BLOCK-END
