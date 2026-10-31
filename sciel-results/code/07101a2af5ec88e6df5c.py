"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def lj_energy_and_grad(pos):
    """
    Calculate Lennard-Jones energy and gradient.
    pos: (N, 3) array of positions.
    Returns: energy, gradient (N, 3)
    """
    # pos is (N, 3)
    # Compute pairwise distances
    # diff[i, j] = pos[i] - pos[j]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    
    # Avoid division by zero for self-pairs
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2**3
    r12 = r6**2
    
    # Energy: 4 * sum (r^-12 - r^-6)
    # Note: The sum is over all pairs i < j.
    # The matrix r2 contains all i,j. 
    # Sum over upper triangle.
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    r2_pairs = r2[mask]
    r6_pairs = r2_pairs**3
    r12_pairs = r6_pairs**2
    
    energy = 4.0 * np.sum(r12_pairs - r6_pairs)
    
    # Gradient
    # dE/dr = 4 * (-12*r^-13 + 6*r^-7) = 24 * (2*r^-13 - r^-7)
    # dE/dx_i = sum_j dE/dr_ij * dr_ij/dx_i
    # dr_ij/dx_i = (x_i - x_j) / r_ij
    # So dE/dx_i = sum_j 24 * (2/r_ij^13 - 1/r_ij^7) * (x_i - x_j) / r_ij
    #             = sum_j 24 * (2/r_ij^14 - 1/r_ij^8) * (x_i - x_j)
    
    # Let's compute the coefficient for each pair
    # coeff = 24 * (2/r^14 - 1/r^8)
    # We need r^8 and r^14.
    # r2_pairs is r^2.
    # r8 = r2_pairs^4
    # r14 = r2_pairs^7
    
    r8_pairs = r2_pairs**4
    r14_pairs = r2_pairs**7
    
    coeff = 24.0 * (2.0 / r14_pairs - 1.0 / r8_pairs)
    
    # coeff is a 1D array of length M (number of pairs)
    # We need to map these back to the gradient matrix
    # grad_matrix[i, j] = coeff_k * (pos[i] - pos[j]) for pair (i,j)
    
    # Initialize gradient
    grad = np.zeros_like(pos)
    
    # Get indices of the pairs
    i_idx, j_idx = np.where(mask)
    
    # Compute vector differences for the pairs
    # diff_pairs = pos[i_idx] - pos[j_idx]
    diff_pairs = pos[i_idx] - pos[j_idx]
    
    # Multiply coefficients (broadcast to 3D)
    # coeff shape (M,), diff_pairs shape (M, 3)
    force_pairs = coeff[:, np.newaxis] * diff_pairs
    
    # Accumulate into grad
    # grad[i] += force_pairs
    # grad[j] -= force_pairs
    
    np.add.at(grad, i_idx, force_pairs)
    np.add.at(grad, j_idx, -force_pairs)
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # 1. Generate FCC lattice points
    # We need enough points to select 38 closest.
    # Max coord needed is roughly 3 or 4.
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    points.append((i, j, k))
    
    # Convert to array
    points = np.array(points, dtype=float)
    
    # 2. Sort by distance from origin
    dists = np.sum(points**2, axis=1)
    sorted_indices = np.argsort(dists)
    points = points[sorted_indices]
    
    # 3. Select first 38
    init_pos = points[:N].copy()
    
    # 4. Scale to approximate LJ minimum distance
    # The minimum of LJ is at r = 2^(1/6) * sigma = 1.12246...
    # The FCC lattice constant a is such that nearest neighbors are at a/sqrt(2).
    # In our integer lattice, nearest neighbors are at distance sqrt(2).
    # So if we scale by s, distance becomes s*sqrt(2).
    # We want s*sqrt(2) = 2^(1/6) => s = 2^(1/6) / sqrt(2) = 2^(1/6 - 3/6) = 2^(-2/6) = 2^(-1/3).
    # Wait.
    # Standard FCC lattice: atoms at (0,0,0), (0.5, 0.5, 0) etc.
    # My integer lattice: (0,0,0), (1,1,0). Distance sqrt(2).
    # If I scale by 1.0, nearest neighbor distance is sqrt(2) approx 1.414.
    # LJ min is 1.122.
    # So I should scale down.
    # Scale factor = 1.122 / 1.414 = 0.7937.
    # Alternatively, just let the optimizer handle it. Starting at 1.0 is fine, it's close enough.
    # Actually, starting at the minimum distance is better.
    scale = 2.0 ** (-1.0/6.0) # This makes nearest neighbor distance 1.122... if base distance is 1.
    # But base distance is sqrt(2).
    # So scale = 1.122 / sqrt(2) = 0.7937.
    init_pos = init_pos * 0.7937
    
    best_pos = init_pos
    best_energy = lj_energy_and_grad(best_pos)[0]
    
    # 5. Optimize
    try:
        # Use L-BFGS-B
        result = minimize(
            lj_energy_and_grad,
            init_pos,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < best_energy:
            best_pos = result.x
            best_energy = result.fun
            
        # Check time
        if time.time() > deadline:
            return best_pos
            
        # Optional: Second pass with tighter tolerance or different method
        # BFGS might be faster for small N
        result2 = minimize(
            lj_energy_and_grad,
            best_pos,
            method='BFGS',
            jac=True,
            options={'maxiter': 200, 'gtol': 1e-10}
        )
        
        if result2.fun < best_energy:
            best_pos = result2.x
            best_energy = result2.fun

    except Exception:
        pass
    
    return best_pos
# EVOLVE-BLOCK-END
