import numpy as np
import time
import itertools

N = 55

def _lennard_jones_energy(pos):
    """Compute the total Lennard-Jones energy."""
    # pos is (N, 3)
    # Calculate pairwise distances
    # Use broadcasting to get distance matrix
    # dist[i, j] = ||pos[i] - pos[j]||
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # LJ potential: 4 * (1/r^12 - 1/r^6)
    # r^-6 = 1/dist^6
    r6 = 1.0 / dist_sq**3
    r12 = r6**2
    
    energy_matrix = 4.0 * (r12 - r6)
    
    # Sum over all pairs, divide by 2 to avoid double counting
    total_energy = 0.5 * np.sum(energy_matrix)
    return total_energy

def _generate_icosahedron_vertices():
    """Generate unit vectors for icosahedron vertices."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    # Normalize
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_icosahedron_edge_midpoints(vertices):
    """Generate unit vectors for midpoints of icosahedron edges."""
    # Icosahedron edges connect vertices that are not opposite.
    # Opposite vertices have dot product -1.
    # Adjacent vertices have dot product -1/phi^2 ? No, dot product is -1/phi? 
    # Let's just check distances.
    # In a regular icosahedron, edge length is constant.
    # We can find pairs (i, j) such that ||v_i - v_j|| is minimal.
    
    dists = np.zeros((len(vertices), len(vertices)))
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            d = np.linalg.norm(vertices[i] - vertices[j])
            dists[i, j] = d
            dists[j, i] = d
    
    # Find the minimum distance (edge length)
    min_dist = np.min(dists[dists > 1e-6])
    
    midpoints = []
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            if abs(dists[i, j] - min_dist) < 1e-6:
                mid = (vertices[i] + vertices[j]) / 2.0
                norm = np.linalg.norm(mid)
                if norm > 1e-6:
                    mid = mid / norm
                    midpoints.append(mid)
    
    # There should be 30 edges
    return np.array(midpoints)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Generate base geometry
    verts = _generate_icosahedron_vertices()
    mids = _generate_icosahedron_edge_midpoints(verts)
    
    # Check counts
    assert len(verts) == 12, f"Expected 12 vertices, got {len(verts)}"
    assert len(mids) == 30, f"Expected 30 midpoints, got {len(mids)}"
    
    # Structure:
    # 1 center
    # 12 inner vertices (R1)
    # 30 edge midpoints (R2)
    # 12 outer vertices (R3) - aligned with inner vertices
    
    def build_positions(r1, r2, r3):
        center = np.zeros((1, 3))
        inner = r1 * verts
        edges = r2 * mids
        outer = r3 * verts
        return np.vstack([center, inner, edges, outer])
    
    # Initial guess for radii
    # LJ equilibrium is 2^(1/6) ~ 1.122
    # Inner shell usually slightly compressed or expanded?
    # Let's start with reasonable values
    r1_init = 1.1
    r2_init = 2.0
    r3_init = 2.8
    
    best_energy = float('inf')
    best_pos = None
    
    # 2. Optimize radii using Nelder-Mead
    def energy_from_radii(radii):
        r1, r2, r3 = radii
        pos = build_positions(r1, r2, r3)
        return _lennard_jones_energy(pos)
    
    try:
        from scipy.optimize import minimize
        
        # Optimize radii
        res_radii = minimize(energy_from_radii, [r1_init, r2_init, r3_init], method='Nelder-Mead', 
                             options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if time.time() > deadline:
            # Fallback to initial guess if optimization takes too long
            r1_opt, r2_opt, r3_opt = r1_init, r2_init, r3_init
        else:
            r1_opt, r2_opt, r3_opt = res_radii.x
        
        pos_init = build_positions(r1_opt, r2_opt, r3_opt)
        best_energy = _lennard_jones_energy(pos_init)
        best_pos = pos_init.copy()
        
        # 3. Full 3D relaxation using L-BFGS
        # We can define a gradient for L-BFGS to speed it up
        def lj_energy_and_grad(pos):
            # pos is (N, 3)
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dist_sq, 1.0)
            
            r6 = 1.0 / dist_sq**3
            r12 = r6**2
            
            energy = 0.5 * np.sum(4.0 * (r12 - r6))
            
            # Gradient calculation
            # dE/dr = 4 * (-12/r^13 + 6/r^7)
            # dE/dx_i = sum_j dE/dr_ij * d r_ij / d x_i
            # d r_ij / d x_i = (x_i - x_j) / r_ij
            # So dE/dx_i = sum_j 4 * (-12/r^13 + 6/r^7) * (x_i - x_j) / r
            #             = sum_j 4 * (-12/r^14 + 6/r^8) * (x_i - x_j)
            
            # Let's compute the term C_ij = 4 * (-12/r^14 + 6/r^8)
            # r^14 = (r^2)^7 = dist_sq^7
            # r^8 = (r^2)^4 = dist_sq^4
            
            inv_r2 = 1.0 / dist_sq
            inv_r2_4 = inv_r2**4
            inv_r2_7 = inv_r2**7
            
            # Coefficient for each pair (i,j)
            # coeff = 4 * (-12 * inv_r2^7 + 6 * inv_r2^4)
            coeff = 4.0 * (-12.0 * inv_r2_7 + 6.0 * inv_r2_4)
            
            # Zero out diagonal
            np.fill_diagonal(coeff, 0.0)
            
            # Gradient: grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
            # grad = coeff @ diff ? No.
            # diff[i,j] = pos[i] - pos[j]
            # grad[i] = sum_j coeff[i,j] * diff[i,j]
            grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            
            return energy, grad
        
        # Check time
        if time.time() < deadline - 5.0:
            res_relax = minimize(lj_energy_and_grad, pos_init, jac=True, method='L-BFGS-B',
                                 options={'maxiter': 100, 'gtol': 1e-6, 'maxls': 20})
            
            if res_relax.success and res_relax.fun < best_energy:
                best_energy = res_relax.fun
                best_pos = res_relax.x.reshape(N, 3)
                
    except Exception as e:
        # If scipy fails or errors out, return initial guess
        pass
        
    # 4. Try a few perturbations if time remains
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(42)
        for i in range(5):
            if time.time() > deadline - 2.0:
                break
            
            # Perturb the best position found so far
            perturbation = rng.normal(0, 0.1, size=(N, 3))
            pos_pert = best_pos + perturbation
            
            # Simple local relaxation
            try:
                res_pert = minimize(lj_energy_and_grad, pos_pert, jac=True, method='L-BFGS-B',
                                    options={'maxiter': 50, 'gtol': 1e-5})
                if res_pert.fun < best_energy:
                    best_energy = res_pert.fun
                    best_pos = res_pert.x.reshape(N, 3)
            except Exception:
                continue
                
    # Ensure minimum distance constraint is met (0.5)
    # The optimizer might push atoms too close.
    # We should check and potentially reject if invalid, but the task says "Atoms must be at least 0.5 apart".
    # If the energy is very low, distances are likely > 0.5 (equilibrium is 1.12).
    # However, if it collapses, it's invalid.
    
    # Final check
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 0.5:
        # If invalid, fall back to the initial icosahedron which is definitely valid
        pos_fallback = build_positions(r1_opt, r2_opt, r3_opt)
        diff_f = pos_fallback[:, np.newaxis, :] - pos_fallback[np.newaxis, :, :]
        dist_sq_f = np.sum(diff_f**2, axis=-1)
        np.fill_diagonal(dist_sq_f, 1.0)
        min_dist_f = np.sqrt(np.min(dist_sq_f))
        if min_dist_f >= 0.5:
            best_pos = pos_fallback
        else:
            # Last resort: cubic lattice
            side = int(np.ceil(N ** (1 / 3)))
            points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
            best_pos = np.array(points[:N], dtype=float) * 1.12

    return best_pos
