"""Program for broadband antireflection coating design using discrete GA search."""
import numpy as np
import itertools
import time

# EVOLVE-BLOCK-START

def _char_matrix_reflectance_vectorized(wavelengths_nm, ambient_idx, substrate_idx,
                              material_indices, thicknesses_nm):
    """
    Compute normal-incidence reflectance (fraction, not percent) for each wavelength
    using the characteristic-matrix method, vectorized over wavelengths.
    """
    n_layers = len(material_indices)
    n_wav = len(wavelengths_nm)
    
    # M is the total transfer matrix, initialized to Identity for all wavelengths
    # Shape: (n_wav, 2, 2)
    M = np.zeros((n_wav, 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for j in range(n_layers):
        n_j = material_indices[j]
        d_j = thicknesses_nm[j]
        
        # delta = 2*pi*n*d/lambda
        # wavelengths_nm shape: (n_wav,)
        # delta shape: (n_wav,)
        delta = 2.0 * np.pi * n_j * d_j / wavelengths_nm
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Layer matrix L for each wavelength
        # L[:, 0, 0] = cos_d
        # L[:, 0, 1] = 1j * sin_d / n_j
        # L[:, 1, 0] = 1j * n_j * sin_d
        # L[:, 1, 1] = cos_d
        
        # M_new = M @ L
        # M is (N, 2, 2), L is effectively (N, 2, 2)
        
        M_00 = M[:, 0, 0]
        M_01 = M[:, 0, 1]
        M_10 = M[:, 1, 0]
        M_11 = M[:, 1, 1]
        
        L_00 = cos_d
        L_01 = 1j * sin_d / n_j
        L_10 = 1j * n_j * sin_d
        L_11 = cos_d
        
        # Matrix multiplication
        new_M_00 = M_00 * L_00 + M_01 * L_10
        new_M_01 = M_00 * L_01 + M_01 * L_11
        new_M_10 = M_10 * L_00 + M_11 * L_10
        new_M_11 = M_10 * L_01 + M_11 * L_11
        
        M[:, 0, 0] = new_M_00
        M[:, 0, 1] = new_M_01
        M[:, 1, 0] = new_M_10
        M[:, 1, 1] = new_M_11
        
    n_sub = substrate_idx
    M00 = M[:, 0, 0]
    M01 = M[:, 0, 1]
    M10 = M[:, 1, 0]
    M11 = M[:, 1, 1]
    
    denom = M00 + M01 * n_sub
    # Avoid division by zero
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    
    Y = (M10 + M11 * n_sub) / denom
    r = (ambient_idx - Y) / (ambient_idx + Y)
    reflectances = np.abs(r) ** 2
    
    return reflectances


def _evaluate_candidate(materials, thicknesses, problem):
    """
    Evaluate a candidate design. Returns mean reflectance in percent, or infinity if invalid.
    """
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    band_nm = problem.get("band_nm", [400, 800])
    materials_dict = problem.get("materials", {"L": 1.38, "H": 2.35})
    ambient_index = problem.get("ambient_index", 1.0)
    substrate_index = problem.get("substrate_index", 1.52)
    
    # Check constraints
    if len(materials) > max_layers:
        return float('inf')
    if len(materials) != len(thicknesses):
        return float('inf')
    
    total_thickness = sum(thicknesses)
    if total_thickness > max_total_nm:
        return float('inf')
    
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return float('inf')
    
    if len(materials) == 0:
        # Bare glass reflectance
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return 100.0 * r ** 2
    
    material_indices = []
    for m in materials:
        if m not in materials_dict:
            return float('inf')
        material_indices.append(materials_dict[m])
    
    # Compute reflectance over 401 equally spaced wavelengths
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    reflectances = _char_matrix_reflectance_vectorized(wavelengths, ambient_index, substrate_index,
                                             material_indices, thicknesses)
    mean_R = np.mean(reflectances)
    return 100.0 * mean_R


def design(problem):
    """
    Design a broadband antireflection coating using a genetic algorithm over
    discrete layer sequences and thickness indices.
    """
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    band_nm = problem.get("band_nm", [400, 800])
    materials_dict = problem.get("materials", {"L": 1.38, "H": 2.35})
    ambient_index = problem.get("ambient_index", 1.0)
    substrate_index = problem.get("substrate_index", 1.52)
    
    # Define discrete thickness options based on Optical Thickness (OT)
    # We explore OT from 80 to 220 nm in steps of 5 nm for finer granularity
    # This gives 29 options.
    # Physical thickness d = OT / n
    
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    
    ot_values = np.arange(80, 221, 5)  # 80, 85, ..., 220
    
    # Convert to physical thicknesses
    L_thicknesses = ot_values / n_L
    H_thicknesses = ot_values / n_H
    
    # Filter out thicknesses that exceed max_layer_nm
    L_thicknesses = L_thicknesses[L_thicknesses <= max_layer_nm]
    H_thicknesses = H_thicknesses[H_thicknesses <= max_layer_nm]
    
    # If filtering removes all, fallback to a single valid value
    if len(L_thicknesses) == 0:
        L_thicknesses = np.array([max_layer_nm / 2.0])
    if len(H_thicknesses) == 0:
        H_thicknesses = np.array([max_layer_nm / 2.0])
    
    n_L_opts = len(L_thicknesses)
    n_H_opts = len(H_thicknesses)
    
    # GA setup
    pop_size = 400
    max_generations = 200
    crossover_rate = 0.7
    mutation_rate = 0.15
    elite_size = 30
    
    # Chromosome representation:
    # - num_layers: 1 to max_layers
    # - For each layer: material (0=L, 1=H) and thickness index
    
    max_ga_layers = min(max_layers, 16)  # Limit GA layers to keep search space manageable
    
    # Initialize population with diverse, physically-motivated designs
    population = []
    
    # Quarter wave indices (closest to 150 nm OT)
    qw_idx_L = np.argmin(np.abs(L_thicknesses * n_L - 150))
    qw_idx_H = np.argmin(np.abs(H_thicknesses * n_H - 150))
    
    # Add structured individuals with various patterns
    # Single layer
    for i in range(5):
        mat = 'L' if i % 2 == 0 else 'H'
        t_idx = np.random.randint(0, n_L_opts if mat == 'L' else n_H_opts)
        t = L_thicknesses[t_idx] if mat == 'L' else H_thicknesses[t_idx]
        population.append(([mat], [float(t)]))
    
    # Two layers
    for i in range(10):
        mat1 = 'L' if i % 2 == 0 else 'H'
        mat2 = 'H' if mat1 == 'L' else 'L'
        t1 = L_thicknesses[np.random.randint(0, n_L_opts)] if mat1 == 'L' else H_thicknesses[np.random.randint(0, n_H_opts)]
        t2 = L_thicknesses[np.random.randint(0, n_L_opts)] if mat2 == 'L' else H_thicknesses[np.random.randint(0, n_H_opts)]
        population.append(([mat1, mat2], [float(t1), float(t2)]))
    
    # Three layers - alternating patterns
    for i in range(15):
        if i % 2 == 0:
            mats = ['H', 'L', 'H']
        else:
            mats = ['L', 'H', 'L']
        thicknesses = []
        for m in mats:
            if m == 'L':
                t = L_thicknesses[np.random.randint(0, n_L_opts)]
            else:
                t = H_thicknesses[np.random.randint(0, n_H_opts)]
            thicknesses.append(float(t))
        population.append((mats, thicknesses))
    
    # Four layers
    for i in range(15):
        if i % 2 == 0:
            mats = ['H', 'L', 'H', 'L']
        else:
            mats = ['L', 'H', 'L', 'H']
        thicknesses = []
        for m in mats:
            if m == 'L':
                t = L_thicknesses[np.random.randint(0, n_L_opts)]
            else:
                t = H_thicknesses[np.random.randint(0, n_H_opts)]
            thicknesses.append(float(t))
        population.append((mats, thicknesses))
    
    # Five layers
    for i in range(10):
        if i % 2 == 0:
            mats = ['H', 'L', 'H', 'L', 'H']
        else:
            mats = ['L', 'H', 'L', 'H', 'L']
        thicknesses = []
        for m in mats:
            if m == 'L':
                t = L_thicknesses[np.random.randint(0, n_L_opts)]
            else:
                t = H_thicknesses[np.random.randint(0, n_H_opts)]
            thicknesses.append(float(t))
        population.append((mats, thicknesses))
    
    # Six layers
    for i in range(10):
        if i % 2 == 0:
            mats = ['H', 'L', 'H', 'L', 'H', 'L']
        else:
            mats = ['L', 'H', 'L', 'H', 'L', 'H']
        thicknesses = []
        for m in mats:
            if m == 'L':
                t = L_thicknesses[np.random.randint(0, n_L_opts)]
            else:
                t = H_thicknesses[np.random.randint(0, n_H_opts)]
            thicknesses.append(float(t))
        population.append((mats, thicknesses))
    
    # Eight layers (for broader coverage)
    for i in range(5):
        if i % 2 == 0:
            mats = ['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L']
        else:
            mats = ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H']
        thicknesses = []
        for m in mats:
            if m == 'L':
                t = L_thicknesses[np.random.randint(0, n_L_opts)]
            else:
                t = H_thicknesses[np.random.randint(0, n_H_opts)]
            thicknesses.append(float(t))
        population.append((mats, thicknesses))
    
    # Fill rest with random
    while len(population) < pop_size:
        n_layers = np.random.randint(1, max_ga_layers + 1)
        materials = []
        thicknesses = []
        for _ in range(n_layers):
            mat = 'L' if np.random.random() < 0.5 else 'H'
            if mat == 'L':
                t_idx = np.random.randint(0, n_L_opts)
                t = L_thicknesses[t_idx]
            else:
                t_idx = np.random.randint(0, n_H_opts)
                t = H_thicknesses[t_idx]
            materials.append(mat)
            thicknesses.append(float(t))
        population.append((materials, thicknesses))
    
    # Evaluate initial population
    def evaluate_individual(ind):
        materials, thicknesses = ind
        return _evaluate_candidate(materials, thicknesses, problem)
    
    fitnesses = [evaluate_individual(ind) for ind in population]
    
    # Track best
    best_fitness = min(fitnesses)
    best_idx = fitnesses.index(best_fitness)
    best_individual = population[best_idx]
    
    # GA loop with adaptive time management
    start_time = time.time()
    time_limit = 55.0  # Leave buffer for overhead
    
    for gen in range(max_generations):
        if time.time() - start_time > time_limit:
            break
            
        # Sort by fitness (lower is better)
        indexed = list(enumerate(fitnesses))
        indexed.sort(key=lambda x: x[1])
        sorted_pop = [population[i] for i, _ in indexed]
        sorted_fit = [f for _, f in indexed]
        
        # Update best
        if sorted_fit[0] < best_fitness:
            best_fitness = sorted_fit[0]
            best_individual = sorted_pop[0]
        
        # Elite selection
        new_population = [sorted_pop[i] for i in range(min(elite_size, len(sorted_pop)))]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            k = 3
            indices = np.random.choice(len(sorted_pop), size=min(k, len(sorted_pop)), replace=False)
            best_i1 = indices[np.argmin([sorted_fit[i] for i in indices])]
            parent1 = sorted_pop[best_i1]
            
            indices = np.random.choice(len(sorted_pop), size=min(k, len(sorted_pop)), replace=False)
            best_i2 = indices[np.argmin([sorted_fit[i] for i in indices])]
            parent2 = sorted_pop[best_i2]
            
            # Crossover
            if np.random.random() < crossover_rate:
                child1, child2 = _crossover(parent1, parent2, max_ga_layers,
                                              L_thicknesses, H_thicknesses,
                                              n_L_opts, n_H_opts)
            else:
                child1 = (list(parent1[0]), list(parent1[1]))
                child2 = (list(parent2[0]), list(parent2[1]))
            
            # Mutation
            child1 = _mutate(child1, mutation_rate, max_ga_layers,
                             L_thicknesses, H_thicknesses, n_L_opts, n_H_opts)
            child2 = _mutate(child2, mutation_rate, max_ga_layers,
                             L_thicknesses, H_thicknesses, n_L_opts, n_H_opts)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        population = new_population
        fitnesses = [evaluate_individual(ind) for ind in population]
        
        # Check if we've converged
        if gen > 30 and best_fitness < 0.25:
            break
            # If we're doing well, do a local refinement
            if best_fitness < 0.3:
                # Perform local search around the best individual
                for _ in range(50):
                    if time.time() - start_time > time_limit:
                        break
                    mat, th = best_individual
                    mat_copy = list(mat)
                    th_copy = list(th)
                    # Mutate one layer
                    if len(mat_copy) > 0:
                        pos = np.random.randint(0, len(mat_copy))
                        if np.random.random() < 0.5:
                            # Change material
                            mat_copy[pos] = 'L' if mat_copy[pos] != 'L' else 'H'
                            if mat_copy[pos] == 'L':
                                th_copy[pos] = float(L_thicknesses[np.random.randint(0, n_L_opts)])
                            else:
                                th_copy[pos] = float(H_thicknesses[np.random.randint(0, n_H_opts)])
                        else:
                            # Change thickness
                            if mat_copy[pos] == 'L':
                                th_copy[pos] = float(L_thicknesses[np.random.randint(0, n_L_opts)])
                            else:
                                th_copy[pos] = float(H_thicknesses[np.random.randint(0, n_H_opts)])
                    fit = _evaluate_candidate(mat_copy, th_copy, problem)
                    if fit < best_fitness:
                        best_fitness = fit
                        best_individual = (mat_copy, th_copy)
            break
    
    # Return the best individual
    materials, thicknesses = best_individual
    return materials, thicknesses


def _crossover(parent1, parent2, max_layers, L_thicknesses, H_thicknesses,
               n_L_opts, n_H_opts):
    """
    Single-point crossover for variable-length chromosomes.
    """
    mat1, t1 = parent1
    mat2, t2 = parent2
    
    len1 = len(mat1)
    len2 = len(mat2)
    
    # Choose a crossover point
    max_len = max(len1, len2)
    if max_len <= 1:
        return (list(mat1), list(t1)), (list(mat2), list(t2))
    
    cp = np.random.randint(1, max_len)
    
    # Child 1: first part from parent1, second part from parent2
    child1_mat = list(mat1[:cp])
    child1_t = list(t1[:cp])
    
    # Pad or truncate parent2's second part
    if len2 > cp:
        child1_mat.extend(mat2[cp:])
        child1_t.extend(t2[cp:])
    
    # Child 2: first part from parent2, second part from parent1
    child2_mat = list(mat2[:cp])
    child2_t = list(t2[:cp])
    
    if len1 > cp:
        child2_mat.extend(mat1[cp:])
        child2_t.extend(t1[cp:])
    
    # Truncate to max_layers
    child1_mat = child1_mat[:max_layers]
    child1_t = child1_t[:max_layers]
    child2_mat = child2_mat[:max_layers]
    child2_t = child2_t[:max_layers]
    
    # Ensure at least 1 layer
    if len(child1_mat) == 0:
        child1_mat = ['L']
        child1_t = [float(L_thicknesses[0])]
    if len(child2_mat) == 0:
        child2_mat = ['H']
        child2_t = [float(H_thicknesses[0])]
    
    return (child1_mat, child1_t), (child2_mat, child2_t)


def _mutate(individual, mutation_rate, max_layers, L_thicknesses, H_thicknesses,
            n_L_opts, n_H_opts):
    """
    Mutate an individual: change material, change thickness, add/remove layer.
    """
    materials = list(individual[0])
    thicknesses = list(individual[1])
    
    n_layers = len(materials)
    
    # Mutate each layer
    for i in range(n_layers):
        if np.random.random() < mutation_rate:
            # Change material
            if np.random.random() < 0.5:
                materials[i] = 'L' if materials[i] != 'L' else 'H'
                # Reset thickness to a random valid value for the new material
                if materials[i] == 'L':
                    thicknesses[i] = float(L_thicknesses[np.random.randint(0, n_L_opts)])
                else:
                    thicknesses[i] = float(H_thicknesses[np.random.randint(0, n_H_opts)])
            else:
                # Change thickness
                if materials[i] == 'L':
                    thicknesses[i] = float(L_thicknesses[np.random.randint(0, n_L_opts)])
                else:
                    thicknesses[i] = float(H_thicknesses[np.random.randint(0, n_H_opts)])
    
    # Add a layer
    if np.random.random() < mutation_rate * 0.5 and n_layers < max_layers:
        mat = 'L' if np.random.random() < 0.5 else 'H'
        if mat == 'L':
            t = float(L_thicknesses[np.random.randint(0, n_L_opts)])
        else:
            t = float(H_thicknesses[np.random.randint(0, n_H_opts)])
        # Insert at a random position
        pos = np.random.randint(0, n_layers + 1)
        materials.insert(pos, mat)
        thicknesses.insert(pos, t)
    
    # Remove a layer
    elif np.random.random() < mutation_rate * 0.5 and n_layers > 1:
        pos = np.random.randint(0, n_layers)
        materials.pop(pos)
        thicknesses.pop(pos)
    
    return (materials, thicknesses)

# EVOLVE-BLOCK-END
