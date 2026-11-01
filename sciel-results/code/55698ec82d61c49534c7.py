"""Genetic algorithm for Lennard-Jones cluster global minimum, 55 atoms."""
import numpy as np
from scipy.optimize import minimize
import time

N = 55


def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4*sum(r^-12 - r^-6) over all pairs."""
    # positions: (N, 3)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    r6 = r ** 6
    r12 = r6 * r6
    # Only sum over i < j to avoid double counting
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    energy = 4.0 * np.sum((r12[mask] - r6[mask]))
    return energy


def relax(positions, maxiter=200):
    """L-BFGS local relaxation."""
    def objective(x):
        pos = x.reshape(N, 3)
        return lj_energy(pos)
    
    result = minimize(objective, positions.flatten(), method='L-BFGS-B', 
                      options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    return result.x.reshape(N, 3)


def generate_icosahedron(n_atoms):
    """Generate a Mackay icosahedral structure for n_atoms up to 147."""
    # Mackay icosahedron shells: 1, 12, 42, 92 -> cumulative 1, 13, 55, 147
    # For 55 atoms: core (1) + first shell (12) + second shell (42)
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12)
    # Standard icosahedron vertices
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    # Core atom at origin
    core = np.array([[0.0, 0.0, 0.0]])
    
    # First shell: 12 atoms at icosahedron vertices, scaled to distance ~1.2
    shell1_radius = 1.2
    shell1 = ico_vertices * shell1_radius
    
    # Second shell: 42 atoms
    # These are at positions that form the next icosahedral shell
    # The 42 positions can be generated from the icosahedral symmetry
    # They correspond to the vertices of a truncated icosahedron-like arrangement
    # or more precisely, they are the 42 positions in the second Mackay shell
    
    # The 42 second-shell positions in a Mackay icosahedron are located at:
    # - 12 positions above/below the 12 vertices (but these are part of the 42)
    # - 30 positions on the faces
    
    # A better approach: generate all 55 positions using known coordinates
    # or use the fact that the 55-atom cluster is well-known
    
    # Let's use a different approach: generate the 55 positions using
    # the icosahedral group symmetry operations applied to a few seed positions
    
    # Actually, let's use a simpler heuristic: place atoms on concentric icosahedral shells
    # Shell 0: 1 atom at center
    # Shell 1: 12 atoms at icosahedron vertices
    # Shell 2: 42 atoms - we need to find these positions
    
    # The 42 positions in the second shell of a Mackay icosahedron can be generated
    # by taking the icosahedron vertices and applying certain transformations
    # They are located at the midpoints of edges and face centers, scaled appropriately
    
    # Edge midpoints: 30 edges, each midpoint gives one position -> 30 positions
    # Face centers: 20 faces, each face center gives one position -> 20 positions
    # Total: 50, but we need 42
    
    # Actually, the Mackay icosahedron second shell has:
    # - 12 atoms at the "vertices" of a larger icosahedron (but not the same as shell 1)
    # - 30 atoms on the "edges"
    # Total = 42
    
    # Let me use a different strategy: start from a reasonable icosahedral-like
    # arrangement and let the GA/relaxation refine it
    
    # For now, let's create a rough icosahedral structure:
    # Place 55 atoms on a deformed icosahedral lattice
    
    # Alternative: use the known fact that the 55-atom LJ minimum is the icosahedron
    # and generate it approximately, then relax
    
    # Let's create positions using icosahedral symmetry
    # The 55 positions are: 1 + 12 + 42
    
    # For the 42, we can use:
    # 12 positions at radius r2 along the 12 icosahedron directions
    # 30 positions at radius r2' along the 30 edge-midpoint directions
    
    # Icosahedron edge midpoints (30 directions)
    # An icosahedron has 30 edges. The midpoints of these edges give 30 directions.
    
    # Let's compute edge midpoints from the icosahedron vertices
    # First, find which vertices are connected (edges)
    # In a regular icosahedron, each vertex connects to 5 others
    
    # For simplicity, let's just use a radial distribution approach:
    # Place 42 atoms on a sphere of radius ~2.2 with icosahedral-like symmetry
    
    # Actually, the simplest robust approach: generate a dense icosahedral-like
    # point set and take the best 55
    
    # Let's try: generate points on a Fibonacci sphere and then select 55
    # that are well-distributed, then relax
    
    # Better yet: use the analytic Mackay icosahedron coordinates
    # The 55-atom Mackay icosahedron has known structure
    
    # I'll use a pragmatic approach: create an approximate icosahedron
    # by placing atoms at icosahedral symmetry positions
    
    # 12-fold axis directions (icosahedron vertices)
    dir12 = ico_vertices  # 12 directions
    
    # 5-fold axis directions are the same as 12-fold (icosahedron has 6 5-fold axes)
    # Wait, icosahedron has 6 5-fold axes (through opposite vertices), 10 3-fold axes (through opposite faces), 15 2-fold axes (through opposite edges)
    
    # The 12 vertices are along the 6 5-fold axes (2 per axis)
    # The 20 face centers are along the 10 3-fold axes (2 per axis)  
    # The 30 edge midpoints are along the 15 2-fold axes (2 per axis)
    
    # For the second shell (42 atoms):
    # - 12 atoms: along the 6 5-fold axes at a different radius? No...
    # 
    # Let me look at this differently. The Mackay icosahedron shells are:
    # Shell 1 (13 atoms): 1 center + 12 at icosahedron vertices
    # Shell 2 (42 atoms): These form a larger icosahedral shell
    #   - 12 atoms at the "vertices" of a larger icosahedron (same directions as shell 1 vertices but further out)
    #   - 30 atoms at the "edge midpoint" directions
    # Total = 42
    
    # So:
    # 12 atoms along the 12 icosahedron vertex directions at radius R2
    # 30 atoms along the 30 edge-midpoint directions at radius R2
    
    # Let's compute the 30 edge-midpoint directions
    # In a regular icosahedron, the edges connect vertices that are at distance 2 (if edge length = 2)
    # Let's find edges by checking distances between all pairs
    
    dists = np.linalg.norm(ico_vertices[:, np.newaxis, :] - ico_vertices[np.newaxis, :, :], axis=2)
    # Find the minimum distance (edge length)
    min_dist = np.min(dists[dists > 1e-10])
    # Get edge pairs
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    # Compute edge midpoints (30 of them)
    edge_midpoints = []
    for i, j in edges:
        mid = (ico_vertices[i] + ico_vertices[j]) / 2
        mid = mid / np.linalg.norm(mid)
        edge_midpoints.append(mid)
    
    edge_midpoints = np.array(edge_midpoints)  # 30 directions
    
    # Now create the 42 second-shell positions
    # 12 along icosahedron vertex directions, 30 along edge-midpoint directions
    # Choose radii to match the LJ minimum spacing
    
    # The first shell is at radius 1.2
    # The second shell should be at a radius such that the nearest-neighbor
    # distances are close to 1.12 (LJ minimum)
    
    # For a Mackay icosahedron, the ratio of shell radii is approximately
    # r2/r1 = phi ≈ 1.618 (golden ratio) for the ideal icosahedron
    # But for LJ, it's optimized
    
    r1 = 1.2
    r2 = r1 * 1.6  # Approximate second shell radius
    
    shell2_12 = dir12 * r2  # 12 atoms
    shell2_30 = edge_midpoints * r2  # 30 atoms
    
    # Combine all 55 atoms
    all_positions = np.vstack([core, shell1, shell2_12, shell2_30])
    
    # Center the structure
    all_positions -= np.mean(all_positions, axis=0)
    
    return all_positions


def genetic_algorithm(pop_size=20, generations=15, mutation_scale=0.3, time_budget=110.0):
    """Genetic algorithm for LJ cluster optimization."""
    start_time = time.time()
    
    # Initialize population
    population = []
    
    # First individual: icosahedral structure
    ico = generate_icosahedron(N)
    ico_relaxed = relax(ico, maxiter=100)
    population.append(ico_relaxed)
    
    # Add diverse random structures
    for i in range(pop_size - 1):
        # Random positions in a sphere
        r = 2.5 * np.cbrt(np.random.random(3))
        theta = np.random.random() * 2 * np.pi
        phi_angle = np.arccos(2 * np.random.random() - 1)
        
        # Actually, let's use random positions in a box
        pos = np.random.uniform(-2, 2, (N, 3))
        # Relax
        try:
            pos_relaxed = relax(pos, maxiter=50)
            population.append(pos_relaxed)
        except:
            population.append(pos.copy())
    
    # Track best
    best_energy = float('inf')
    best_position = None
    
    for pos in population:
        e = lj_energy(pos)
        if e < best_energy:
            best_energy = e
            best_position = pos.copy()
    
    # Main GA loop
    for gen in range(generations):
        if time.time() - start_time > time_budget:
            break
        
        # Evaluate fitness
        energies = []
        for pos in population:
            e = lj_energy(pos)
            energies.append(e)
        energies = np.array(energies)
        
        # Update best
        min_idx = np.argmin(energies)
        if energies[min_idx] < best_energy:
            best_energy = energies[min_idx]
            best_position = population[min_idx].copy()
        
        # Create next generation
        new_population = []
        
        # Elitism: keep the best individual
        new_population.append(best_position.copy())
        
        # Crossover and mutation
        for i in range(pop_size - 1):
            if time.time() - start_time > time_budget:
                break
            
            # Select parents using tournament selection
            parent1_idx = np.random.choice(pop_size, size=3, replace=False)
            parent1_idx = parent1_idx[np.argmin(energies[parent1_idx])]
            
            parent2_idx = np.random.choice(pop_size, size=3, replace=False)
            parent2_idx = parent2_idx[np.argmin(energies[parent2_idx])]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover: exchange subsets of atoms
            # Randomly assign each atom to come from parent1 or parent2
            mask = np.random.random(N) < 0.5
            child = np.where(mask[:, np.newaxis], parent1, parent2)
            
            # Mutation: small Gaussian displacements
            if np.random.random() < 0.8:
                child = child + np.random.normal(0, mutation_scale, child.shape)
            
            # Relax
            try:
                child_relaxed = relax(child, maxiter=80)
                new_population.append(child_relaxed)
            except:
                new_population.append(child.copy())
        
        population = new_population[:pop_size]
        
        # Update mutation scale
        mutation_scale *= 0.95
    
    return best_position, best_energy


def construct():
    """Construct 55-atom LJ cluster configuration."""
    # Run genetic algorithm
    best_pos, best_energy = genetic_algorithm(
        pop_size=12,
        generations=10,
        mutation_scale=0.25,
        time_budget=110.0
    )
    
    # Final relaxation
    final_pos = relax(best_pos, maxiter=200)
    
    return final_pos
