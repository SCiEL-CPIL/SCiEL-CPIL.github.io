import numpy as np
import time
import sys

N = 55

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill

    # 1. Generate Mackay Icosahedron (55 atoms)
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2.0
    
    # Vertices of a regular icosahedron
    # (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    verts = []
    for sign1 in [-1, 1]:
        for sign2 in [-1, 1]:
            verts.append([0, sign1, sign2 * phi])
            verts.append([sign1, sign2 * phi, 0])
            verts.append([sign2 * phi, 0, sign1])
            
    verts = np.array(verts)
    
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    # Initial radii for the two shells
    # r1 is the first shell (12 atoms)
    # r2 is the second shell (42 atoms: 12 vertices + 30 edge midpoints)
    # Good initial guesses based on LJ minimum r=2^(1/6) ~ 1.122
    r1 = 1.12246
    r2 = 2.24492  # Approx 2 * r1
    
    # Shell 1: 12 atoms at vertices
    shell1 = verts * r1
    
    # Shell 2: 42 atoms
    # The 12 vertices of the outer icosahedron
    shell2_vertices = verts * r2
    
    # The 30 edge midpoints
    # We need to find the edges of the icosahedron.
    # An edge exists if the distance between two vertices is minimal.
    # Distance between adjacent vertices in unit icosahedron:
    # Let's compute pairwise distances to find edges.
    d = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    min_d = d.min()
    # Tolerance for floating point
    edge_mask = (d < min_d * 1.1) & (np.arange(12)[:, np.newaxis] != np.arange(12)[np.newaxis, :])
    
    # Get unique edges (i, j) where i < j
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if edge_mask[i, j]:
                edges.append((i, j))
                
    # There should be 30 edges
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        mid_unit = mid / mid_norm
        midpoints.append(mid_unit)
        
    midpoints = np.array(midpoints)
    shell2_midpoints = midpoints * r2
    
    # Combine all atoms
    center = np.array([[0.0, 0.0, 0.0]])
    positions = np.vstack([center, shell1, shell2_vertices, shell2_midpoints])
    
    # Ensure we have exactly 55 atoms
    assert len(positions) == N, f"Expected {N} atoms, got {len(positions)}"
    
    # 2. Relax the structure using L-BFGS
    # Define LJ energy and gradient
    def lj_energy_grad(pos):
        pos = pos.reshape(N, 3)
        # Calculate all pair distances
        # To save memory and time, we can do this in a vectorized way
        # diff[i, j, k] = pos[i, k] - pos[j, k]
        # But N=55 is small, so full matrix is fine.
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero (i==j)
        # We only care about i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        r_sq = r_sq[mask]
        r = np.sqrt(r_sq)
        
        # LJ potential: 4 * (r^-12 - r^-6)
        # Force is derivative. 
        # E = 4 * sum (r^-12 - r^-6)
        # dE/dr = 4 * (-12*r^-13 + 6*r^-7) = 24 * (2*r^-13 - r^-7)
        # Force on atom i from atom j: F_ij = -dE/dr * (r_vec / r)
        # But for gradient, we can use:
        # grad_i E = sum_j 4 * (2*r^-13 - r^-7) * (pos_i - pos_j) / r
        # Note: r^-13 * (pos_i - pos_j)/r = r^-14 * (pos_i - pos_j) ? No.
        # Let u = pos_i - pos_j. r = |u|.
        # d/dr (r^-12) = -12 r^-13.
        # dE/dr = 4 (-12 r^-13 + 6 r^-7).
        # Grad_i E = (dE/dr) * (u / r) = 4 (-12 r^-13 + 6 r^-7) * u * r^-1
        # = 4 (-12 r^-14 + 6 r^-8) * u
        # = 24 (2 r^-14 - r^-8) * u
        
        r_inv = 1.0 / r
        r_inv_6 = r_inv ** 6
        r_inv_12 = r_inv_6 ** 2
        r_inv_8 = r_inv_6 * r_inv_2 if False else r_inv ** 8 # just compute directly
        r_inv_14 = r_inv_12 * r_inv_2 if False else r_inv ** 14
        
        # Let's just compute the coefficient C = 24 * (2*r^-14 - r^-8)
        # u is the vector from j to i? No, diff[i,j] = pos[i] - pos[j].
        # So u = pos[i] - pos[j].
        # The force on i is in direction u.
        # Grad_i E = C * u.
        
        # We need to sum over j for each i.
        # Let's reconstruct the full matrix of u and coefficients.
        # This is easier to vectorize by creating the full (N, N, 3) array.
        
        diff_full = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r_sq_full = np.sum(diff_full**2, axis=2)
        np.fill_diagonal(r_sq_full, 1.0) # Avoid div by 0
        
        r_full = np.sqrt(r_sq_full)
        
        # Coefficient for each pair (i, j)
        # C_ij = 24 * (2 * r^-14 - r^-8)
        # Note: r^-14 = (r^-2)^7. r^-8 = (r^-2)^4.
        r_inv2 = 1.0 / r_sq_full
        r_inv4 = r_inv2 ** 2
        r_inv8 = r_inv4 ** 2
        r_inv14 = r_inv8 * r_inv4 * r_inv2
        
        coeff = 24.0 * (2.0 * r_inv14 - r_inv8)
        
        # Zero out diagonal
        np.fill_diagonal(coeff, 0.0)
        
        # Gradient for each atom i: sum_j coeff[i, j] * diff_full[i, j]
        # diff_full[i, j] is pos[i] - pos[j]
        grad = np.sum(coeff[:, :, np.newaxis] * diff_full, axis=1)
        
        # Energy
        # E = 4 * sum_{i<j} (r^-12 - r^-6)
        # Sum over all i!=j is 2 * sum_{i<j}
        # So E = 2 * 0.5 * 4 * sum_{i!=j} ... no.
        # E = 4 * sum_{i<j} (r^-12 - r^-6)
        # Let's compute sum over i!=j and divide by 2.
        r_inv6_full = r_inv2 ** 3
        r_inv12_full = r_inv6_full ** 2
        pair_energy = 4.0 * (r_inv12_full - r_inv6_full)
        np.fill_diagonal(pair_energy, 0.0)
        energy = 0.5 * np.sum(pair_energy)
        
        return energy, grad

    # Initial position
    x0 = positions
    
    # Check validity
    def is_valid(pos):
        pos = pos.reshape(N, 3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(r, np.inf)
        return r.min() > 0.5

    # Optimization
    # Use L-BFGS-B for bounded optimization if needed, but plain L-BFGS is often faster for smooth problems.
    # We'll use scipy.optimize.minimize
    from scipy.optimize import minimize
    
    best_pos = x0
    best_energy = np.inf
    
    try:
        res = minimize(lj_energy_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < best_energy:
            best_pos = res.x
            best_energy = res.fun
            
    except Exception as e:
        print(f"Optimization failed: {e}", file=sys.stderr)
    
    # Check deadline
    if time.time() > deadline:
        return best_pos.reshape(N, 3)
    
    # Final validity check
    if not is_valid(best_pos):
        # If invalid, try to push atoms apart or return seed? 
        # The Mackay icosahedron should be valid.
        # If it's not, it's a bug in generation or relaxation.
        # Let's return the initial guess if relaxation broke it.
        if is_valid(x0):
            return x0
        else:
            return x0 # Should not happen

    return best_pos.reshape(N, 3)

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Re-use the logic from the scratch design above
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill

    # 1. Generate Mackay Icosahedron (55 atoms)
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2.0
    
    # Vertices of a regular icosahedron
    # (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    verts = []
    for sign1 in [-1, 1]:
        for sign2 in [-1, 1]:
            verts.append([0, sign1, sign2 * phi])
            verts.append([sign1, sign2 * phi, 0])
            verts.append([sign2 * phi, 0, sign1])
            
    verts = np.array(verts)
    
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    # Initial radii for the two shells
    r1 = 1.12246
    r2 = 2.24492  # Approx 2 * r1
    
    # Shell 1: 12 atoms at vertices
    shell1 = verts * r1
    
    # Shell 2: 42 atoms
    # The 12 vertices of the outer icosahedron
    shell2_vertices = verts * r2
    
    # The 30 edge midpoints
    d = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    min_d = d.min()
    edge_mask = (d < min_d * 1.1) & (np.arange(12)[:, np.newaxis] != np.arange(12)[np.newaxis, :])
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if edge_mask[i, j]:
                edges.append((i, j))
                
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        mid_unit = mid / mid_norm
        midpoints.append(mid_unit)
        
    midpoints = np.array(midpoints)
    shell2_midpoints = midpoints * r2
    
    # Combine all atoms
    center = np.array([[0.0, 0.0, 0.0]])
    positions = np.vstack([center, shell1, shell2_vertices, shell2_midpoints])
    
    assert len(positions) == N, f"Expected {N} atoms, got {len(positions)}"
    
    # 2. Relax the structure using L-BFGS
    def lj_energy_grad(pos):
        pos = pos.reshape(N, 3)
        diff_full = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r_sq_full = np.sum(diff_full**2, axis=2)
        np.fill_diagonal(r_sq_full, 1.0)
        
        r_full = np.sqrt(r_sq_full)
        
        r_inv2 = 1.0 / r_sq_full
        r_inv4 = r_inv2 ** 2
        r_inv8 = r_inv4 ** 2
        r_inv14 = r_inv8 * r_inv4 * r_inv2
        
        coeff = 24.0 * (2.0 * r_inv14 - r_inv8)
        
        np.fill_diagonal(coeff, 0.0)
        
        grad = np.sum(coeff[:, :, np.newaxis] * diff_full, axis=1)
        
        r_inv6_full = r_inv2 ** 3
        r_inv12_full = r_inv6_full ** 2
        pair_energy = 4.0 * (r_inv12_full - r_inv6_full)
        np.fill_diagonal(pair_energy, 0.0)
        energy = 0.5 * np.sum(pair_energy)
        
        return energy, grad

    x0 = positions
    
    def is_valid(pos):
        pos = pos.reshape(N, 3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(r, np.inf)
        return r.min() > 0.5

    from scipy.optimize import minimize
    
    best_pos = x0
    best_energy = np.inf
    
    try:
        res = minimize(lj_energy_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < best_energy:
            best_pos = res.x
            best_energy = res.fun
            
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return best_pos.reshape(N, 3)
    
    if not is_valid(best_pos):
        if is_valid(x0):
            return x0
        else:
            return x0

    return best_pos.reshape(N, 3)
# EVOLVE-BLOCK-END
