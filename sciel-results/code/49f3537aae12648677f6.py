"""Genetic algorithm approach for Lennard-Jones 55-atom cluster global minimum."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations


def lj_energy(positions):
    """Compute total Lennard-Jones energy for a set of positions."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r2 = np.dot(rij, rij)
            if r2 < 1e-10:
                return 1e10  # penalty for overlapping atoms
            r = np.sqrt(r2)
            if r < 0.5:
                return 1e10  # penalty for too-close atoms
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 ** 3
            inv_r12 = inv_r6 ** 2
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy


def lj_energy_vec(positions):
    """Vectorized Lennard-Jones energy computation."""
    n = len(positions)
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    # Mask out diagonal
    np.fill_diagonal(dist2, np.inf)
    # Only use upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    d2 = dist2[iu]
    d2 = np.maximum(d2, 1e-10)
    inv_d2 = 1.0 / d2
    inv_d6 = inv_d2 ** 3
    inv_d12 = inv_d6 ** 2
    energy = 4.0 * np.sum(inv_d12 - inv_d6)
    return energy


def relax(positions):
    """Relax a structure using L-BFGS."""
    try:
        result = minimize(
            lj_energy_vec,
            positions,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-4}
        )
        if result.success:
            return result.x, result.fun
        else:
            # If optimization fails, return the best point found
            return result.x, lj_energy_vec(result.x)
    except Exception:
        return positions, lj_energy_vec(positions)


def generate_icosahedral_seed():
    """Generate a Mackay icosahedron-like structure for 55 atoms."""
    # Mackay icosahedron: 13 atoms in inner shell (icosahedron), 42 in outer shell
    # Inner shell: 12 vertices of icosahedron + 1 center
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # Icosahedron vertices (12)
    inner_shell = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit length
    norms = np.linalg.norm(inner_shell, axis=1, keepdims=True)
    inner_shell = inner_shell / norms
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Inner shell radius ~1.0 (in LJ units, minimum at r=2^(1/6)≈1.122)
    inner_radius = 1.122
    inner_positions = inner_shell * inner_radius
    inner_positions = np.vstack([center, inner_positions])  # 13 atoms
    
    # Outer shell: 42 atoms - place on faces and edges of icosahedron
    # For simplicity, generate points on a larger icosahedron-like shell
    outer_radius = 2.244  # approximately 2 * inner_radius
    
    # Generate outer shell points: use combinations of icosahedron vertices
    # The 42 outer atoms in Mackay icosahedron are:
    # - 30 on the faces (2 per face * 20 faces... actually it's more complex)
    # - 12 on the edges
    # Let's use a simpler approach: generate points on a sphere with icosahedral symmetry
    
    # Use the icosahedron vertices and their midpoints and face centers
    outer_points = []
    
    # 12 edge midpoints (scaled)
    # Icosahedron has 30 edges
    # Get all pairs of vertices that are connected (distance ~2)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(inner_shell[i] - inner_shell[j])
            if 1.8 < d < 2.2:  # connected vertices
                edges.append((i, j))
    
    # Take 12 of these edges for outer shell points
    for idx, (i, j) in enumerate(edges[:12]):
        mid = (inner_shell[i] + inner_shell[j]) / 2
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 0:
            mid_dir = mid / mid_norm
            outer_points.append(mid_dir)
    
    # Add face centers (20 faces)
    # For each set of 3 connected vertices, add face center
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                di = np.linalg.norm(inner_shell[i] - inner_shell[j])
                dj = np.linalg.norm(inner_shell[j] - inner_shell[k])
                dk = np.linalg.norm(inner_shell[i] - inner_shell[k])
                if (1.8 < di < 2.2) and (1.8 < dj < 2.2) and (1.8 < dk < 2.2):
                    face_center = (inner_shell[i] + inner_shell[j] + inner_shell[k]) / 3
                    fc_norm = np.linalg.norm(face_center)
                    if fc_norm > 0:
                        face_dir = face_center / fc_norm
                        faces.append(face_dir)
    
    # Take 30 outer points from edges and faces
    outer_points = np.array(outer_points) if outer_points else np.array([])
    
    # If we don't have enough, add more points on a sphere
    while len(outer_points) < 42:
        # Add random points on sphere
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-1, 1)
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        outer_points = np.vstack([outer_points, [[x, y, z]]]) if len(outer_points) > 0 else np.array([[x, y, z]])
    
    outer_points = outer_points[:42]
    outer_positions = outer_points * outer_radius
    
    all_positions = np.vstack([inner_positions, outer_positions])
    
    # Center the structure
    all_positions -= np.mean(all_positions, axis=0)
    
    return all_positions


def generate_random_structure():
    """Generate a random structure with minimum distance constraint."""
    positions = np.random.randn(N, 3) * 2.0
    # Ensure minimum distance
    for _ in range(100):
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    min_dist = d
        if min_dist >= 0.5:
            break
        # Push apart too-close atoms
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 0.5:
                    direction = (positions[i] - positions[j]) / max(d, 1e-6)
                    push = (0.5 - d) / 2 + 0.1
                    positions[i] += direction * push
                    positions[j] -= direction * push
    return positions


def crossover(parent1, parent2):
    """Exchange subsets of atoms between parents."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Select a random subset of atoms to exchange
    n_exchange = np.random.randint(10, 26)  # exchange 10-25 atoms
    indices = np.random.choice(N, size=n_exchange, replace=False)
    
    child1[indices] = parent2[indices]
    child2[indices] = parent1[indices]
    
    return child1, child2


def mutation(positions, sigma=0.5):
    """Apply small Gaussian displacements."""
    mutated = positions + np.random.randn(N, 3) * sigma
    return mutated


def genetic_algorithm():
    """Run genetic algorithm to find low-energy LJ cluster configuration."""
    np.random.seed(42)
    
    population_size = 20
    generations = 15
    
    # Initialize population
    population = []
    energies = []
    
    # Add icosahedral seed
    ico_seed = generate_icosahedral_seed()
    relaxed_ico, energy_ico = relax(ico_seed)
    population.append(relaxed_ico)
    energies.append(energy_ico)
    
    # Add random structures
    for i in range(population_size - 1):
        pos = generate_random_structure()
        relaxed_pos, energy = relax(pos)
        population.append(relaxed_pos)
        energies.append(energy)
    
    best_energy = min(energies)
    best_position = population[energies.index(best_energy)]
    
    for gen in range(generations):
        # Sort by energy
        sorted_indices = np.argsort(energies)
        population = [population[i] for i in sorted_indices]
        energies = [energies[i] for i in sorted_indices]
        
        # Elitism: keep best 5
        elite_size = 5
        new_population = population[:elite_size]
        new_energies = energies[:elite_size]
        
        # Generate offspring
        while len(new_population) < population_size:
            # Select parents via tournament selection
            idx1 = np.random.choice(min(elite_size, len(population)))
            idx2 = np.random.choice(min(elite_size, len(population)))
            
            parent1 = population[idx1]
            parent2 = population[idx2]
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutation(child1, sigma=0.3)
            child2 = mutation(child2, sigma=0.3)
            
            # Relax and evaluate
            relaxed1, energy1 = relax(child1)
            relaxed2, energy2 = relax(child2)
            
            new_population.append(relaxed1)
            new_energies.append(energy1)
            
            if len(new_population) < population_size:
                new_population.append(relaxed2)
                new_energies.append(energy2)
        
        # Update best
        gen_best_idx = np.argmin(new_energies)
        if new_energies[gen_best_idx] < best_energy:
            best_energy = new_energies[gen_best_idx]
            best_position = new_population[gen_best_idx]
        
        population = new_population
        energies = new_energies
    
    return best_position, best_energy


def construct():
    """Construct the 55-atom LJ cluster configuration."""
    # Try genetic algorithm
    try:
        best_pos, best_energy = genetic_algorithm()
        # Final relaxation
        final_pos, final_energy = relax(best_pos)
        return final_pos
    except Exception:
        # Fallback to icosahedral structure
        pos = generate_icosahedral_seed()
        relaxed_pos, _ = relax(pos)
        return relaxed_pos
# EVOLVE-BLOCK-END
