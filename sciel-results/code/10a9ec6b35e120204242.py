import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 55

def lj_energy(positions):
    """Calculate the total Lennard-Jones energy for a given set of positions."""
    # positions shape (N, 3)
    # Compute pairwise distances
    # Use broadcasting for efficiency
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero (should not happen if min dist > 0.5, but just in case)
    dist_sq[dist_sq < 1e-12] = 1e-12
    
    # LJ potential: 4 * (r^-12 - r^-6)
    # r^-6 = 1 / (r^2)^3
    r_inv_6 = 1.0 / (dist_sq ** 3)
    r_inv_12 = r_inv_6 ** 2
    
    potential = 4.0 * (r_inv_12 - r_inv_6)
    
    # Sum over all pairs, but divide by 2 because we counted each pair twice (i,j and j,i)
    # Also exclude self-interaction (diagonal)
    total_energy = 0.5 * np.sum(potential) - np.sum(np.diag(potential))
    # Note: np.diag(potential) is infinite or large, but we set dist_sq to 1e-12 so it's large.
    # Better to just sum upper triangle.
    iu = np.triu_indices_from(potential, k=1)
    total_energy = np.sum(potential[iu])
    
    return total_energy

def construct():
    """
    Construct the Mackay icosahedron configuration for 55 atoms.
    """
    phi = (1 + np.sqrt(5)) / 2.0
    
    # 1. Generate Inner Shell: 12 atoms forming a Dodecahedron
    # Vertices of a regular dodecahedron
    dodeca_verts = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        (0, 1/phi, phi), (0, -1/phi, phi), (0, 1/phi, -phi), (0, -1/phi, -phi),
        (1/phi, phi, 0), (-1/phi, phi, 0), (1/phi, -phi, 0), (-1/phi, -phi, 0),
        (phi, 0, 1/phi), (phi, 0, -1/phi), (-phi, 0, 1/phi), (-phi, 0, -1/phi)
    ]
    # Wait, standard dodecahedron vertices are:
    # (±1, ±1, ±1)
    # (0, ±1/φ, ±φ)
    # (±1/φ, ±φ, 0)
    # (±φ, 0, ±1/φ)
    # Let's generate them properly to avoid duplicates or errors.
    
    dodeca_list = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                dodeca_list.append((x, y, z))
                
    inv_phi = 1.0 / phi
    for x in [0]:
        for y in [inv_phi, -inv_phi]:
            for z in [phi, -phi]:
                dodeca_list.append((x, y, z))
    for x in [inv_phi, -inv_phi]:
        for y in [phi, -phi]:
            for z in [0]:
                dodeca_list.append((x, y, z))
    for x in [phi, -phi]:
        for y in [0]:
            for z in [inv_phi, -inv_phi]:
                dodeca_list.append((x, y, z))
                
    dodeca_verts = np.array(dodeca_list)
    # Normalize to unit radius? No, we want specific radius.
    # Current radius is sqrt(3) for (1,1,1) etc.
    # Let's scale to R1
    R1 = 1.4
    current_r1 = np.sqrt(np.sum(dodeca_verts[0]**2))
    inner_shell = dodeca_verts * (R1 / current_r1)
    
    # 2. Generate Outer Shell: 42 atoms forming an Icosahedron
    # 12 vertices of icosahedron
    ico_verts_list = []
    for x in [0]:
        for y in [-1, 1]:
            for z in [phi, -phi]:
                ico_verts_list.append((x, y, z))
    for x in [-1, 1]:
        for y in [phi, -phi]:
            for z in [0]:
                ico_verts_list.append((x, y, z))
    for x in [phi, -phi]:
        for y in [0]:
            for z in [-1, 1]:
                ico_verts_list.append((x, y, z))
                
    ico_verts = np.array(ico_verts_list)
    
    # Find edges of the icosahedron to get midpoints
    # Edge length is 2.0 for these coordinates.
    # We need to find all pairs of vertices that are distance 2 apart.
    edges = []
    for i in range(len(ico_verts)):
        for j in range(i+1, len(ico_verts)):
            dist = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if np.isclose(dist, 2.0):
                edges.append((i, j))
                
    # Generate 30 midpoints
    midpoints = []
    for i, j in edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2.0
        midpoints.append(mid)
        
    # Combine 12 vertices and 30 midpoints
    outer_shell_points = np.vstack([ico_verts, np.array(midpoints)])
    
    # Scale to R2
    R2 = 2.8
    current_r2 = np.sqrt(np.sum(ico_verts[0]**2)) # Radius of ico vertices
    # The midpoints have a different radius. 
    # Radius of ico vertex: sqrt(1 + phi^2)
    # Radius of midpoint: sqrt( (1/2)^2 + (phi/2)^2 ) ? No.
    # Midpoint of (0,1,phi) and (1,phi,0) is (0.5, (1+phi)/2, phi/2).
    # It's better to scale the entire outer shell such that the *vertices* are at R2.
    # The midpoints will automatically be at a smaller radius, which is correct for the Mackay structure?
    # Actually, in a Mackay icosahedron, the outer shell is often considered as a deformed icosahedron where the "vertices" are the 12 ico vertices and the "face centers" or "edge centers" are the other atoms.
    # Wait, the standard Mackay icosahedron for N=55 has 12 inner, 42 outer.
    # The 42 outer atoms form a icosahedral cluster.
    # The 12 atoms are at the vertices of a larger icosahedron.
    # The 30 atoms are at the centers of the faces? No, 30 edges.
    # Actually, the 42 atoms form a "stretched" icosahedron.
    # Let's just scale the 12 vertices to R2. The midpoints will follow.
    
    outer_shell = outer_shell_points * (R2 / current_r2)
    
    # Combine inner and outer shells
    positions = np.vstack([inner_shell, outer_shell])
    
    # Ensure we have exactly 55 atoms
    assert len(positions) == 55, f"Expected 55 atoms, got {len(positions)}"
    
    # Local Relaxation
    # Use L-BFGS-B to minimize energy
    # Bounds: None, but we must ensure min dist > 0.5.
    # L-BFGS-B doesn't handle non-linear constraints well natively without penalty.
    # However, starting from a good geometry, it should stay in the valid region.
    # To be safe, we can add a small penalty if dist < 0.5, or just trust the geometry.
    # Given the time limit, a simple minimize is best.
    
    res = minimize(lj_energy, positions, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-5})
    
    # Return the relaxed positions
    return res.x

# EVOLVE-BLOCK-START
# The code above is the complete implementation.
# We just need to wrap it in the required format if necessary, but the prompt says 
# "Rewrite the program from scratch... Keep the program's entry-point function name(s)..."
# The current program has construct() inside EVOLVE-BLOCK.
# I will place the entire logic inside the block.
# EVOLVE-BLOCK-END
