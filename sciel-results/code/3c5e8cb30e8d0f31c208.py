"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for a set of positions."""
    dists = pdist(pos, 'euclidean')
    if np.any(dists < 1e-12):
        return 1e10
    inv_r2 = 1.0 / (dists * dists)
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    return energy

def _lj_gradient(pos):
    """Compute gradient of Lennard-Jones energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_3d = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists_3d, 1.0) # Avoid division by zero
    
    inv_r2 = 1.0 / (dists_3d ** 2)
    inv_r6 = inv_r2 ** 3
    inv_r8 = inv_r2 ** 4
    inv_r14 = inv_r2 ** 7
    
    # dE/dx_i = sum_j (x_i - x_j) * 4 * (-12 * r^-14 + 6 * r^-8)
    #         = sum_j (x_i - x_j) * (-48 * r^-14 + 24 * r^-8)
    coeff = (-48.0 * inv_r14 + 24.0 * inv_r8)
    np.fill_diagonal(coeff, 0.0)
    
    grad_3d = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad_3d.flatten()

def _relax(pos):
    """Relax a structure using L-BFGS-B."""
    pos = pos.copy()
    
    def energy_and_grad(x):
        pos_reshaped = x.reshape(-1, 3)
        if np.any(pdist(pos_reshaped) < 1e-12):
            return 1e10, np.zeros_like(x)
        energy = _lj_energy(pos_reshaped)
        grad = _lj_gradient(pos_reshaped)
        return energy, grad
    
    try:
        result = minimize(
            energy_and_grad,
            pos.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        relaxed = result.x.reshape(-1, 3)
        return relaxed
    except Exception:
        return pos

def _valid(pos):
    """Check if positions are valid (min distance >= 0.5)."""
    dists = pdist(pos, 'euclidean')
    return np.min(dists) >= 0.5

def _make_icosahedral_seed():
    """Create a Mackay icosahedron seed for 55 atoms."""
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    # Scale first shell
    R1 = 1.12
    shell1 = verts * R1
    
    # Second shell: 42 atoms
    # 30 edge midpoints
    # Find edges of icosahedron (distance between verts is min)
    dists = pdist(verts)
    min_dist = np.min(dists)
    edge_mask = dists < (min_dist * 1.01)
    
    # Reconstruct edge pairs
    edges = []
    idx = 0
    for i in range(12):
        for j in range(i+1, 12):
            if edge_mask[idx]:
                edges.append((i, j))
            idx += 1
            
    edge_midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)
    edge_midpoints = edge_midpoints / np.linalg.norm(edge_midpoints, axis=1)[:, np.newaxis]
    
    # 12 atoms at vertices of a rotated icosahedron
    # Rotation by 36.87 degrees (angle between vertex and face center)
    # Face normal of face (0,1,phi), (0,-1,phi), (1,phi,0)
    # Let's just use a known rotation matrix or generate dual
    # Simpler: The 12 atoms are at the vertices of a smaller icosahedron 
    # that is rotated such that its vertices point to the face centers of the original.
    # Face centers of icosahedron form a dodecahedron.
    # Vertices of dodecahedron are at (0, 1, +/-1/phi), etc.
    # Let's use dodecahedron vertices scaled.
    dodeca_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, -1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [-1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ], dtype=float)
    # Wait, dodecahedron has 20 vertices. I need 12.
    # The 12 atoms in the second shell of Mackay icosahedron are actually
    # the vertices of a smaller icosahedron rotated by 36.87 deg.
    # Let's rotate the original icosahedron vertices.
    theta = np.arccos(1/np.sqrt(5)) # Angle between vertex and face center? No.
    # Angle between vertex (0,1,phi) and face center of adjacent face.
    # Let's just use the 30 edge midpoints and 12 points from a second icosahedron
    # rotated by 36.87 degrees around an axis.
    
    # Rotation matrix for 36.87 degrees around z-axis? No, symmetry axis.
    # Let's just use the 30 edge midpoints and 12 points from the dual dodecahedron?
    # No, the 42 atoms are 30 edge midpoints and 12 vertices of a smaller icosahedron.
    # The smaller icosahedron is rotated by 36.87 degrees.
    
    # Let's approximate the 12 atoms by taking the 12 vertices of the icosahedron
    # and rotating them by 36.87 degrees around the z-axis? No, that breaks symmetry.
    # Correct rotation: Rotate around an axis passing through a vertex and center.
    # Let's use the fact that the 12 atoms are at distance R2.
    # We can just take the 12 vertices of the icosahedron, rotate by 36.87 deg 
    # around the x-axis, and scale.
    
    angle = np.arctan(1/phi) # 31.7 deg? No.
    # Angle between vertex and face center is 36.87 deg.
    rot_angle = np.deg2rad(36.87)
    
    # Rotation around x-axis
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    R_x = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    
    rotated_verts = verts @ R_x.T
    rotated_verts = rotated_verts / np.linalg.norm(rotated_verts, axis=1)[:, np.newaxis]
    
    # Combine
    R2 = 2.24
    shell2 = np.vstack([edge_midpoints * R2, rotated_verts * R2])
    
    center = np.array([[0.0, 0.0, 0.0]])
    all_points = np.vstack([center, shell1, shell2])
    
    return all_points

def construct():
    """Genetic algorithm to find the global minimum of 55-atom LJ cluster."""
    np.random.seed(42)
    start_time = time.time()
    timeout = 110 # Leave 10s buffer
    
    pop_size = 12
    population = []
    
    # 1. Icosahedral Seed
    ico_seed = _make_icosahedral_seed()
    if _valid(ico_seed):
        population.append(ico_seed.copy())
    else:
        population.append(np.random.randn(N, 3) * 2.0)
        
    # 2. Random Seeds
    while len(population) < pop_size:
        pts = np.random.randn(N, 3) * 2.5
        if _valid(pts):
            population.append(pts)
        else:
            # Force valid by expanding
            for _ in range(10):
                pts *= 1.1
                if _valid(pts):
                    break
            if _valid(pts):
                population.append(pts)
            else:
                population.append(np.random.randn(N, 3) * 3.0)
                
    # Relax initial population
    relaxed_population = []
    for pos in population:
        relaxed = _relax(pos)
        if _valid(relaxed):
            relaxed_population.append(relaxed)
        else:
            relaxed_population.append(pos)
            
    # Sort by energy
    relaxed_population.sort(key=lambda x: _lj_energy(x))
    best_pos = relaxed_population[0].copy()
    best_energy = _lj_energy(best_pos)
    
    # GA Loop
    max_generations = 30
    for gen in range(max_generations):
        if time.time() - start_time > timeout:
            break
            
        new_population = []
        
        # Elitism: Keep best 2
        new_population.append(relaxed_population[0].copy())
        if len(relaxed_population) > 1:
            new_population.append(relaxed_population[1].copy())
            
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection for parents
            idx1 = np.random.choice(len(relaxed_population), 3, replace=False)
            idx2 = np.random.choice(len(relaxed_population), 3, replace=False)
            parent1 = relaxed_population[idx1[np.argmin([_lj_energy(relaxed_population[i]) for i in idx1])]]
            parent2 = relaxed_population[idx2[np.argmin([_lj_energy(relaxed_population[i]) for i in idx2])]]
            
            child = parent1.copy()
            
            # Crossover: Swap 10-20 atoms
            n_swap = np.random.randint(10, 21)
            swap_idx = np.random.choice(N, n_swap, replace=False)
            child[swap_idx] = parent2[swap_idx]
            
            # Mutation
            child += np.random.randn(N, 3) * 0.5
            
            # Relax
            relaxed_child = _relax(child)
            if _valid(relaxed_child):
                new_population.append(relaxed_child)
            else:
                new_population.append(child)
                
        # Sort new population
        new_population.sort(key=lambda x: _lj_energy(x))
        
        # Update best
        if _lj_energy(new_population[0]) < best_energy:
            best_energy = _lj_energy(new_population[0])
            best_pos = new_population[0].copy()
            
        relaxed_population = new_population
        
    # Final relaxation
    final_relaxed = _relax(best_pos)
    if _valid(final_relaxed):
        best_pos = final_relaxed
        
    return best_pos
# EVOLVE-BLOCK-END
