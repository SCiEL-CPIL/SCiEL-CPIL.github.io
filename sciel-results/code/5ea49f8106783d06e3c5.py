import numpy as np
import time
import scipy.optimize

N = 55

def lj_energy_and_grad(pos):
    """
    Calculate Lennard-Jones energy and gradient for a set of positions.
    pos: (N, 3) array
    """
    N = pos.shape[0]
    # Compute all pair distances
    # diff[i, j] = pos[i] - pos[j]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    # Avoid division by zero for i=j
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2**-3
    r12 = r6**2
    
    # Energy: 4 * sum (r^-12 - r^-6) for i < j
    # The sum over all i != j counts each pair twice.
    # So energy = 2 * 4 * sum_{i<j} (r^-12 - r^-6) = 8 * sum_{i<j} (r^-12 - r^-6)
    # Alternatively, sum over all i!=j is 2 * E_pairs.
    # E = 4 * 0.5 * sum_{i!=j} (r^-12 - r^-6) = 2 * sum_{i!=j} (r^-12 - r^-6)
    
    # Mask out diagonal
    mask = ~np.eye(N, dtype=bool)
    
    e_sum = np.sum(r12[mask] - r6[mask])
    energy = 4.0 * 0.5 * e_sum # Wait, standard LJ is 4*eps*(sigma/r)^12 - 4*eps*(sigma/r)^6.
    # Sum over all pairs i<j.
    # My sum r12 - r6 is over all i!=j.
    # So sum_{i<j} (r^-12 - r^-6) = 0.5 * sum_{i!=j} (r^-12 - r^-6)
    # Energy = 4 * sum_{i<j} (r^-12 - r^-6) = 4 * 0.5 * e_sum = 2 * e_sum.
    
    # Gradient
    # d/dr (4 r^-12 - 4 r^-6) = -48 r^-13 + 24 r^-7 = 24 r^-7 (1 - 2 r^-6)
    # d/dx_i = d/dr * dr/dx_i
    # dr/dx_i = (x_i - x_j) / r
    # So force on i from j: - dE/dx_i = ...
    # Gradient of Energy w.r.t x_i:
    # grad_i = sum_{j != i} (24 r^-7 (1 - 2 r^-6) * (x_i - x_j) / r) * (-1)? 
    # Let's use the standard form:
    # F_ij = 24 * (2 r^-14 - r^-8) * (x_i - x_j)
    # Grad_i = - sum_{j != i} F_ij ? No, Grad is the derivative of E.
    # E = sum 4(r^-12 - r^-6)
    # dE/dx_i = sum_{j != i} dE/dr * dr/dx_i
    # dE/dr = 4(-12 r^-13 + 6 r^-7) = 24 r^-7 (1 - 2 r^-6)
    # dr/dx_i = (x_i - x_j)/r
    # Term = 24 r^-8 (1 - 2 r^-6) (x_i - x_j)
    
    # Vectorized gradient calculation
    # coeff[i,j] = 24 * r^-8 * (1 - 2*r^-6)
    r8 = r2**-4
    coeff = 24.0 * r8 * (1.0 - 2.0 * r6)
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    # diff is (i,j) -> pos[i]-pos[j]
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Generate Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 2. Generate Dodecahedron vertices (Face centers of Icosahedron)
    # The vertices of a regular dodecahedron can be generated from:
    # (±1, ±1, ±1) and (0, ±1/phi, ±phi), (±1/phi, ±phi, 0), (±phi, 0, ±1/phi)
    dodec_verts = []
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                dodec_verts.append([s1, s2, s3])
    inv_phi = 1.0 / phi
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            dodec_verts.append([0, s1 * inv_phi, s2 * phi])
            dodec_verts.append([s1 * inv_phi, s2 * phi, 0])
            dodec_verts.append([s1 * phi, 0, s2 * inv_phi])
    dodec_verts = np.array(dodec_verts)
    dodec_verts /= np.linalg.norm(dodec_verts, axis=1, keepdims=True)
    
    # 3. Construct Mackay icosahedron
    # Inner shell: 12 icosahedron vertices
    # Outer shell: 12 icosahedron vertices + 30 dodecahedron vertices
    # Center: 1 atom
    
    # Initial radii
    # LJ minimum is at r = 2^(1/6) ≈ 1.12246
    r1_init = 1.12246
    r2_init = r1_init * np.sqrt(2)
    
    def build_structure(r1, r2):
        inner = verts * r1
        outer_ico = verts * r2
        outer_dod = dodec_verts * r2
        # Combine
        # 1 center
        center = np.array([[0.0, 0.0, 0.0]])
        # 12 inner
        # 12 outer ico
        # 30 outer dod
        pos = np.vstack([center, inner, outer_ico, outer_dod])
        return pos
    
    def objective(params):
        r1, r2 = params
        pos = build_structure(r1, r2)
        # Check min distance constraint roughly to avoid huge energy spikes if overlapping
        # But for optimization, we just compute energy
        e, _ = lj_energy_and_grad(pos)
        return e
    
    # Optimize radii r1, r2
    # This is a 2D optimization, very fast
    initial_radii = np.array([r1_init, r2_init])
    
    # Use Nelder-Mead for robustness in 2D, or L-BFGS-B
    # Nelder-Mead is good for small dimensions without gradients
    res = scipy.optimize.minimize(objective, initial_radii, method='Nelder-Mead', 
                                   options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
    
    best_r1, best_r2 = res.x
    best_pos = build_structure(best_r1, best_r2)
    best_energy, _ = lj_energy_and_grad(best_pos)
    
    # Check time
    if time.time() > deadline:
        return best_pos
    
    # 4. Full local relaxation with L-BFGS-B
    # This will polish the structure to the exact local minimum
    # We use the analytical gradient
    res_full = scipy.optimize.minimize(lj_energy_and_grad, best_pos, method='L-BFGS-B', 
                                       jac=True, 
                                       options={'gtol': 1e-10, 'ftol': 1e-10, 'maxiter': 1000, 'maxfun': 1000})
    
    final_pos = res_full.x
    final_energy, _ = lj_energy_and_grad(final_pos)
    
    # Verify min distance constraint
    # "Atoms must be at least 0.5 apart"
    # LJ potential goes to infinity as r->0, so relaxation should naturally avoid this.
    # But let's check just in case.
    diff = final_pos[:, np.newaxis, :] - final_pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    if min_dist < 0.5:
        # This shouldn't happen with LJ, but if it does, return the previous valid one
        # Or just return best_pos which is likely valid
        pass
    else:
        best_pos = final_pos
        best_energy = final_energy

    return best_pos

# EVOLVE-BLOCK-START
# The code above is the implementation. 
# The prompt asks to keep function names. construct() is the entry point.
# The EVOLVE-BLOCK markers in the original code suggest that's where the logic goes.
# I will place the full logic inside the block or just provide the full file.
# The instruction says: "Rewrite the program from scratch... Output the COMPLETE program"
# So I will output the whole file.
# EVOLVE-BLOCK-END
