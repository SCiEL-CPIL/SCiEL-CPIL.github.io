# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _generate_fcc_38(scale=1.1225, dist=1.0):
    """
    Generate 38 FCC lattice points forming a truncated octahedron.
    'scale' controls the nearest-neighbor distance.
    'dist' controls the anisotropic distortion along the z-axis.
    """
    a = np.sqrt(2) * scale
    
    # Integer grid indices
    # The 38 atoms are the closest to the origin in an FCC lattice.
    # We generate a sufficient range and pick the closest.
    # Max coord in units of a is ~2.5. Range -3 to 3 is safe.
    i_range = np.arange(-3, 4)
    j_range = np.arange(-3, 4)
    k_range = np.arange(-3, 4)
    
    # Meshgrid
    I, J, K = np.meshgrid(i_range, j_range, k_range, indexing='ij')
    
    # Basis vectors (in units of a/2? No, let's use a)
    # Basis offsets: (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5) * a
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]) * a
    
    # We need to expand the grid to include basis offsets.
    # Instead of looping, we can create a large array.
    # Total points: 7*7*7*4 = 1372. This is small.
    points_list = []
    for b in basis:
        pts = np.stack([
            I.ravel() * a + b[0],
            J.ravel() * a + b[1],
            K.ravel() * a + b[2]
        ], axis=-1)
        points_list.append(pts)
        
    points = np.vstack(points_list)
    
    # Apply z-distortion
    points[:, 2] *= dist
    
    # Compute distances from origin
    dists = np.linalg.norm(points, axis=1)
    
    # Sort by distance and take the 38 closest
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    # Center
    points = points - np.mean(points, axis=0)
    
    return points


def _lj_energy_and_grad(positions):
    """Compute total LJ energy and gradient for a set of positions."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    
    r6 = r2 ** 3
    r12 = r6 ** 2
    
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6) / 2.0
    
    r_inv_8 = 1.0 / r6
    r_inv_14 = r_inv_8 / (r2 * r2)
    
    coeff = 4.0 * (-12.0 * r_inv_14 + 6.0 * r_inv_8)
    np.fill_diagonal(coeff, 0.0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def construct():
    """
    Optimize rigid-body parameters (scale, dist) using gradient-based methods.
    """
    
    def energy_params(params):
        s, d = params
        if s < 0.5 or d < 0.5: # Avoid singularities/overlaps
            return 1e10
        pts = _generate_fcc_38(scale=s, dist=d)
        e, _ = _lj_energy_and_grad(pts)
        return e

    # Initial guess
    # Optimal LJ distance is 2^(1/6) ~ 1.1225.
    # In our generator, 'scale' is the nearest neighbor distance.
    # So start at 1.1225.
    # Dist should be 1.0.
    x0 = np.array([1.1225, 1.0])
    
    # Optimize rigid body parameters
    # Nelder-Mead is robust for 2D non-smooth-ish landscapes (though this is smooth)
    # L-BFGS-B is also good.
    res_params = minimize(energy_params, x0, method='Nelder-Mead', 
                          options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000, 'adaptive': True})
    
    optimal_scale = res_params.x[0]
    optimal_dist = res_params.x[1]
    
    # Generate optimized rigid body
    positions = _generate_fcc_38(scale=optimal_scale, dist=optimal_dist)
    
    # Verify minimum separation
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    
    if min_r < 0.5:
        # If rigid body causes overlap, revert to standard scale
        positions = _generate_fcc_38(scale=1.1225, dist=1.0)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(np.min(r2))
        if min_r < 0.5:
             # Fallback
             side = int(np.ceil(N ** (1 / 3)))
             pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
             pts.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
             positions = np.array(pts[:N], dtype=float) * 1.12
             return positions

    # Optional: Short local relaxation to polish
    # The rigid body is likely very close to the minimum.
    # We do a short L-BFGS run to ensure we are at a local minimum of the full system.
    x0 = positions.flatten()
    
    def objective(x):
        pos = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g.flatten()
        
    # Short run, tight tolerance
    res = minimize(objective, x0, method='L-BFGS-B', jac=True, 
                   options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if res.success and res.fun < res_params.fun:
        positions = res.x.reshape(-1, 3)
        
        # Check separation again
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(np.min(r2))
        
        if min_r < 0.5:
            positions = _generate_fcc_38(scale=optimal_scale, dist=optimal_dist)
            
    return positions
# EVOLVE-BLOCK-END
