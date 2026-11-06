"""Program for broadband antireflection coating design using discrete GA search."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _char_matrix_reflectance(wavelengths_nm, ambient_idx, substrate_idx,
                              material_indices, thicknesses_nm):
    """
    Compute normal-incidence reflectance (fraction, not percent) for each wavelength
    using the characteristic-matrix method.
    """
    n_layers = len(material_indices)
    reflectances = np.zeros(len(wavelengths_nm))
    
    for i, lam in enumerate(wavelengths_nm):
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = material_indices[j]
            d_j = thicknesses_nm[j]
            delta = 2.0 * np.pi * n_j * d_j / lam
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            L = np.array([[cos_d, 1j * sin_d / n_j],
                          [1j * n_j * sin_d, cos_d]], dtype=complex)
            M = M @ L
        
        n_sub = substrate_idx
        M00, M01 = M[0,0], M[0,1]
        M10, M11 = M[1,0], M[1,1]
        
        denom = M00 + M01 * n_sub
        if abs(denom) < 1e-15:
            reflectances[i] = 1.0
            continue
        Y = (M10 + M11 * n_sub) / denom
        r = (ambient_idx - Y) / (ambient_idx + Y)
        reflectances[i] = np.abs(r) ** 2
    
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
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return 100.0 * r ** 2
    
    material_indices = []
    for m in materials:
        if m not in materials_dict:
            return float('inf')
        material_indices.append(materials_dict[m])
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    reflectances = _char_matrix_reflectance(wavelengths, ambient_index, substrate_index,
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
    
    # Define discrete thickness options centered on quarter-wave optical thicknesses
    # For broadband AR, quarter-wave at center wavelength (600nm) is a good starting point
    # L (n=1.38): QW at 600nm => d = 600/(4*1.38) ≈ 108.7 nm
    # H (n=2.35): QW at 600nm => d = 600/(4*2.35) ≈ 63.8 nm
    # We'll use a finer grid around these values
    
    # L thicknesses: range from 50 to 200 nm, with emphasis on QW region
    L_thicknesses = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 175, 190, 200], dtype=float)
    # H thicknesses: range from 30 to 130 nm, with emphasis on QW region  
    H_thicknesses = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110, 120, 130], dtype=float)
    
    # Ensure all are within max_layer_nm
    L_thicknesses = L_thicknesses[L_thicknesses <= max_layer_nm]
    H_thicknesses = H_thicknesses[H_thicknesses <= max_layer_nm]
    
    if len(L_thicknesses) == 0:
        L_thicknesses = np.array([max_layer_nm / 2.0])
    if len(H_thicknesses) == 0:
        H_thicknesses = np.array([max_layer_nm / 2.0])
    
    n_L_opts = len(L_thicknesses)
    n_H_opts = len(H_thicknesses)
    
    # GA setup - increased for better exploration
    pop_size = 200
    max_generations = 120
    crossover_rate = 0.85
    mutation_rate = 0.12
    elite_size = 15
    
    max_ga_layers = min(max_layers, 15)
    
    # Time budget: 60s CPU, leave margin for safety
    start_time = time.time()
    time_budget = 55.0  # seconds
    
    # Initialize population with a mix of random and informed starts
    population = []
    
    # Add some known good starting points
    # Single QW layer on glass
    population.append((['L'], [108.7]))
    population.append((['H'], [63.8]))
    # Two-layer QW stack (L/H or H/L)
    population.append((['L', 'H'], [108.7, 63.8]))
    population.append((['H', 'L'], [63.8, 108.7]))
    # Three-layer stack
    population.append((['L', 'H', 'L'], [108.7, 63.8, 108.7]))
    population.append((['H', 'L', 'H'], [63.8, 108.7, 63.8]))
    
    # Fill rest with random individuals
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
    
    def evaluate_individual(ind):
        materials, thicknesses = ind
        return _evaluate_candidate(materials, thicknesses, problem)
    
    fitnesses = [evaluate_individual(ind) for ind in population]
    
    best_fitness = min(fitnesses)
    best_idx = fitnesses.index(best_fitness)
    best_individual = population[best_idx]
    
    # GA loop
    for gen in range(max_generations):
        # Check time
        if time.time() - start_time > time_budget:
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
            indices = np.random.choice(len(sorted_pop), size=k, replace=False)
            parent1 = sorted_pop[indices[np.argmin([sorted_fit[i] for i in indices])]]
            indices = np.random.choice(len(sorted_pop), size=k, replace=False)
            parent2 = sorted_pop[indices[np.argmin([sorted_fit[i] for i in indices])]]
            
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
        
        # Check if we've converged well
        if gen > 30 and best_fitness < 0.3:
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
    
    max_len = max(len1, len2)
    if max_len <= 1:
        return (list(mat1), list(t1)), (list(mat2), list(t2))
    
    cp = np.random.randint(1, max_len)
    
    child1_mat = list(mat1[:cp])
    child1_t = list(t1[:cp])
    
    if len2 > cp:
        child1_mat.extend(mat2[cp:])
        child1_t.extend(t2[cp:])
    
    child2_mat = list(mat2[:cp])
    child2_t = list(t2[:cp])
    
    if len1 > cp:
        child2_mat.extend(mat1[cp:])
        child2_t.extend(t1[cp:])
    
    child1_mat = child1_mat[:max_layers]
    child1_t = child1_t[:max_layers]
    child2_mat = child2_mat[:max_layers]
    child2_t = child2_t[:max_layers]
    
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
    
    for i in range(n_layers):
        if np.random.random() < mutation_rate:
            if np.random.random() < 0.5:
                materials[i] = 'L' if materials[i] != 'L' else 'H'
                if materials[i] == 'L':
                    thicknesses[i] = float(L_thicknesses[np.random.randint(0, n_L_opts)])
                else:
                    thicknesses[i] = float(H_thicknesses[np.random.randint(0, n_H_opts)])
            else:
                if materials[i] == 'L':
                    thicknesses[i] = float(L_thicknesses[np.random.randint(0, n_L_opts)])
                else:
                    thicknesses[i] = float(H_thicknesses[np.random.randint(0, n_H_opts)])
    
    if np.random.random() < mutation_rate * 0.4 and n_layers < max_layers:
        mat = 'L' if np.random.random() < 0.5 else 'H'
        if mat == 'L':
            t = float(L_thicknesses[np.random.randint(0, n_L_opts)])
        else:
            t = float(H_thicknesses[np.random.randint(0, n_H_opts)])
        pos = np.random.randint(0, n_layers + 1)
        materials.insert(pos, mat)
        thicknesses.insert(pos, t)
    
    elif np.random.random() < mutation_rate * 0.4 and n_layers > 1:
        pos = np.random.randint(0, n_layers)
        materials.pop(pos)
        thicknesses.pop(pos)
    
    return (materials, thicknesses)

# EVOLVE-BLOCK-END
