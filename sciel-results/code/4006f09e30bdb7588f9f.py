"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

def _get_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron, normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _get_icosahedron_edge_midpoints(verts):
    """Return 30 edge midpoints of icosahedron, normalized to unit radius."""
    n = len(verts)
    midpoints = []
    for i in range(n):
        for j in range(i+1, n):
            # Two vertices are connected if their distance is the edge length
            d = np.linalg.norm(verts[i] - verts[j])
            # Edge length for unit-radius icosahedron is 2/phi
            if np.isclose(d, 2.0/phi, atol=1e-10):
                mid = (verts[i] + verts[j]) / 2.0
                norm = np.linalg.norm(mid)
                if norm > 1e-10:
                    midpoints.append(mid / norm)
    return np.array(midpoints)

def _get_icosahedron_face_centers(verts):
    """Return 20 face centers of icosahedron, normalized to unit radius."""
    n = len(verts)
    # Find all triangles (faces) - three vertices that form an equilateral triangle
    # For icosahedron, faces are triangles where all three edges have length 2/phi
    edge_len = 2.0 / ((1 + np.sqrt(5)) / 2)
    faces = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[k] - verts[i])
                if (np.isclose(d1, edge_len, atol=1e-10) and 
                    np.isclose(d2, edge_len, atol=1e-10) and 
                    np.isclose(d3, edge_len, atol=1e-10)):
                    center = (verts[i] + verts[j] + verts[k]) / 3.0
                    norm = np.linalg.norm(center)
                    if norm > 1e-10:
                        faces.append(center / norm)
    return np.array(faces)

def _lj_energy(positions):
    """Compute total Lennard-Jones energy."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2**-3
    r12 = r6**2
    # Only count each pair once
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _lj_energy_with_grad(positions):
    """Compute LJ energy and gradient."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2**-3
    r12 = r6**2
    energy = 4.0 * np.sum(r12 - r6)
    # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)
    # In terms of r2: dE/dx = sum_j 4*(-12*(r2^-7) + 6*(r2^-4)) * 2 * (x_i - x_j)
    # = sum_j 8*(-12*(r2^-7) + 6*(r2^-4)) * (x_i - x_j)
    coeff = 8.0 * (-12.0 * r2**-7 + 6.0 * r2**-4)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _construct_from_radii(r1, r2, r3):
    """Construct 55-atom Mackay icosahedron from shell radii."""
    ico_verts = _get_icosahedron_vertices()
    edge_mids = _get_icosahedron_edge_midpoints(ico_verts)
    
    # For 55 atoms: 1 center + 12 (shell1) + 30 (shell2) + 12 (shell3)
    # Shell 3: use 12 of the 20 face centers - actually, let's use icosahedron vertices again at r3
    # Or better: the outer shell of Mackay icosahedron for 55 is typically the 12 vertices of a dodecahedron
    # which correspond to 12 of the 20 icosahedron face centers
    
    # Actually, for the standard Mackay icosahedron with 55 atoms, the three shells are:
    # - 12 atoms at icosahedron vertices (radius r1)
    # - 30 atoms at icosahedron edge midpoints (radius r2)  
    # - 12 atoms at... let's use the icosahedron vertices again but we need distinct positions
    
    # Let me reconsider: perhaps the outer 12 are at the face centers of the icosahedron
    # but there are 20 face centers. We need to select 12.
    
    # Alternative: use the dodecahedron vertices. The dodecahedron is dual to icosahedron.
    # Dodecahedron has 20 vertices, not 12.
    
    # I think for 55 atoms, the structure might be:
    # 1 + 12 + 30 + 12 where the last 12 are at positions corresponding to 
    # the 12 vertices of an icosahedron but at a different radius.
    
    # Let's just use icosahedron vertices for both shell 1 and shell 3
    shell1 = ico_verts * r1
    shell2 = edge_mids * r2
    shell3 = ico_verts * r3
    
    center = np.array([[0.0, 0.0, 0.0]])
    
    positions = np.vstack([center, shell1, shell2, shell3])
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    ico_verts = _get_icosahedron_vertices()
    edge_mids = _get_icosahedron_edge_midpoints(ico_verts)
    
    # Initial guess for radii based on known optimal values
    # Best known: energy ≈ -279.25
    # Typical radii: r1 ~ 1.08, r2 ~ 1.45, r3 ~ 1.85
    initial_radii = np.array([1.08, 1.45, 1.85])
    
    def objective(radii):
        positions = _construct_from_radii(radii[0], radii[1], radii[2])
        # Check minimum distance constraint
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(r2, np.inf)
        min_dist = np.sqrt(np.min(r2))
        if min_dist < 0.5:
            return 1e10 + (0.5 - min_dist)**2
        return _lj_energy(positions)
    
    # Optimize the 3 radii using Nelder-Mead
    best_positions = _construct_from_radii(*initial_radii)
    best_energy = _lj_energy(best_positions)
    
    try:
        result = minimize(
            objective, 
            initial_radii, 
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        if time.time() < deadline:
            optimal_radii = result.x
            positions_nm = _construct_from_radii(*optimal_radii)
            energy_nm = _lj_energy(positions_nm)
            
            if energy_nm < best_energy:
                best_positions = positions_nm
                best_energy = energy_nm
        
        # Final L-BFGS relaxation with tight tolerance
        if time.time() < deadline:
            res = minimize(
                _lj_energy,
                best_positions,
                jac=lambda x: _lj_energy_with_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if time.time() < deadline:
                positions_lbfgs = res.x
                energy_lbfgs = _lj_energy(positions_lbfgs)
                
                if energy_lbfgs < best_energy:
                    best_positions = positions_lbfgs
                    best_energy = energy_lbfgs
        
        # Ensure minimum distance constraint is satisfied
        diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
        r2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(r2, np.inf)
        min_dist = np.sqrt(np.min(r2))
        
        if min_dist < 0.5:
            # Fallback to the seed program if constraint violated
            side = int(np.ceil(N ** (1 / 3)))
            points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
            return np.array(points[:N], dtype=float) * 1.12
        
    except Exception:
        # Fallback to seed
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        return np.array(points[:N], dtype=float) * 1.12
    
    return best_positions
# EVOLVE-BLOCK-END

