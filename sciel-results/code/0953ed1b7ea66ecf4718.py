"""Construct LJ38 global minimum: FCC truncated octahedron with local relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar, minimize


def _fcc_truncated_octahedron_param(scale=1.0, dist=1.0):
    """
    Generate 38 FCC lattice points forming a truncated octahedron.
    'scale' controls the overall size.
    'dist' controls the cubic distortion (ratio of z-axis to x/y axis).
    """
    a = np.sqrt(2) * scale  # base lattice constant
    
    # Apply distortion: stretch z-axis by 'dist'
    # We generate points in a standard FCC grid then transform.
    # Standard FCC points are integer combinations of basis vectors.
    # Basis vectors for FCC with lattice constant a:
    # v1 = (a, 0, 0)
    # v2 = (0, a, 0)
    # v3 = (0, 0, a)
    # But we need the conventional cubic cell.
    # Let's generate points in a cubic box of side L = a * max_coord.
    # FCC points are at (i,j,k) where i,j,k are integers and i+j+k is even, 
    # OR more simply, the 4 basis points in the cell (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0).
    
    max_coord = 3
    points_list = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for offset in [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    x = i * a + offset[0]
                    y = j * a + offset[1]
                    z = (k * a + offset[2]) * dist
                    points_list.append([x, y, z])
    
    points = np.array(points_list)
    
    # Compute distances from origin
    dists = np.linalg.norm(points, axis=1)
    
    # Sort by distance and take the 38 closest
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    return points


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


def _lj_energy_and_grad(positions):
    """Compute total LJ energy and gradient for a set of positions."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    
    r = np.sqrt(r2)
    r6 = r2 ** 3
    r12 = r6 ** 2
    
    # Energy
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6) / 2.0
    
    # Gradient
    # dE/dr = 4 * (-12 * r^-13 + 6 * r^-7)
    # dE/dx_i = sum_j dE/dr_ij * dr_ij/dx_i
    # dr_ij/dx_i = (x_i - x_j) / r_ij
    # So dE/dx_i = sum_j 4 * (-12 * r^-13 + 6 * r^-7) * (x_i - x_j) / r
    #            = sum_j 4 * (-12 * r^-14 + 6 * r^-8) * (x_i - x_j)
    #            = sum_j 4 * r^-8 * (-12 * r^-6 + 6) * (x_i - x_j)
    
    # Let's compute the force term F_ij = -dE/dr_ij * (x_i - x_j)/r_ij
    # Force on i due to j is proportional to (x_i - x_j) * (12/r^13 - 6/r^7) * 4
    # Gradient is the negative of force? No, gradient is dE/dx.
    # dE/dx_i = sum_{j != i} 4 * (-12 * r_ij^-13 + 6 * r_ij^-7) * (x_i - x_j) / r_ij
    #         = sum_{j != i} 4 * (-12 * r_ij^-14 + 6 * r_ij^-8) * (x_i - x_j)
    
    # Vectorized calculation
    # coeff[i,j] = 4 * (-12 * r^-14 + 6 * r^-8)
    r_inv_8 = 1.0 / r6
    r_inv_14 = r_inv_8 / (r2 * r2) # r^-14 = r^-8 * r^-6
    
    coeff = 4.0 * (-12.0 * r_inv_14 + 6.0 * r_inv_8)
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_j coeff[i,j] * (x_i - x_j)
    # grad = coeff @ diff
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def construct():
    # Optimize rigid-body parameters: scale and distortion
    def energy_params(params):
        s, d = params
        pts = _fcc_truncated_octahedron_param(scale=s, dist=d)
        return _lj_energy(pts)
    
    # Grid search for robust initialization to avoid local minima in parameter space
    best_energy = np.inf
    best_params = [1.0, 1.0]
    
    # Coarse grid around expected values
    scales = np.linspace(0.95, 1.05, 7)
    dists = np.linspace(0.95, 1.05, 7)
    
    for s in scales:
        for d in dists:
            e = energy_params([s, d])
            if e < best_energy:
                best_energy = e
                best_params = [s, d]
                
    # Refine with Nelder-Mead
    res_params = minimize(energy_params, best_params, method='Nelder-Mead', 
                          options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
    
    optimal_scale = res_params.x[0]
    optimal_dist = res_params.x[1]
    
    # Generate the optimized rigid body
    positions = _fcc_truncated_octahedron_param(scale=optimal_scale, dist=optimal_dist)
    
    # Local relaxation using L-BFGS-B
    # Flatten positions for scipy.optimize.minimize
    x0 = positions.flatten()
    
    def objective(x):
        pos = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g.flatten()
    
    # Use L-BFGS-B for robustness
    res = minimize(objective, x0, method='L-BFGS-B', jac=True, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if res.success:
        positions = res.x.reshape(-1, 3)
        
    # Verify minimum separation >= 0.5
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    
    if min_r < 0.5:
        # If relaxation caused overlap (unlikely with LJ repulsion), revert to scaled FCC
        positions = _fcc_truncated_octahedron_param(scale=optimal_scale, dist=optimal_dist)
        # Check again
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(np.min(r2))
        if min_r < 0.5:
            # Fallback: use the seed program's approach
            side = int(np.ceil(N ** (1 / 3)))
            pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            pts.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
            positions = np.array(pts[:N], dtype=float) * 1.12
    
    return positions
# EVOLVE-BLOCK-END
