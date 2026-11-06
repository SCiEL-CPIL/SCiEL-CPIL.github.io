"""Program for broadband antireflection coating design using discrete GA search."""
import numpy as np
import itertools

# EVOLVE-BLOCK-START

def _char_matrix_reflectance(wavelengths_nm, ambient_idx, substrate_idx,
                              material_indices, thicknesses_nm):
    """
    Compute normal-incidence reflectance (fraction, not percent) for each wavelength
    using the characteristic-matrix method.
    
    Parameters:
    wavelengths_nm: array of wavelengths in nm
    ambient_idx: refractive index of ambient (air)
    substrate_idx: refractive index of substrate (glass)
    material_indices: list of refractive indices for each layer (from air to substrate)
    thicknesses_nm: list of physical thicknesses in nm for each layer
    
    Returns:
    array of reflectance values (0 to 1) for each wavelength
    """
    n_layers = len(material_indices)
    # Convert wavelengths to angular frequency or just use k = 2*pi*n/lambda
    # For each wavelength, build the transfer matrix
    # M = product of layer matrices from top to bottom
    # Each layer matrix: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # where delta = 2*pi*n*d/lambda
    
    reflectances = np.zeros(len(wavelengths_nm))
    
    for i, lam in enumerate(wavelengths_nm):
        # Build total matrix
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = material_indices[j]
            d_j = thicknesses_nm[j]
            delta = 2.0 * np.pi * n_j * d_j / lam
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            # Layer characteristic matrix
            L = np.array([[cos_d, 1j * sin_d / n_j],
                          [1j * n_j * sin_d, cos_d]], dtype=complex)
            M = M @ L
        
        # Boundary conditions
        # At the interface between ambient and first layer (or substrate if no layers)
        # Y = M[0,0] * n_sub + M[0,1]
        # X = M[1,0] * n_sub + M[1,1]
        # Actually, the standard formula:
        # Let's use the admittance approach.
        # The total transfer matrix relates fields at top to fields at bottom.
        # E_top = M[0,0]*E_bottom + M[0,1]*H_bottom
        # H_top = M[1,0]*E_bottom + M[1,1]*H_bottom
        # At the substrate, H_bottom = E_bottom / n_sub (for forward wave only, assuming no back reflection from beyond substrate)
        # Wait, that's not quite right. The characteristic matrix method:
        # For a stack on a substrate, the effective admittance Y is:
        # Y = (M[1,0] + M[1,1] * n_sub) / (M[0,0] + M[0,1] * n_sub)
        # Then reflectance r = (n_amb - Y) / (n_amb + Y)
        # R = |r|^2
        
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
        lam = 0.5 * (band_nm[0] + band_nm[1])
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return 100.0 * r ** 2
    
    material_indices = []
    for m in materials:
        if m not in materials_dict:
            return float('inf')
        material_indices.append(materials_dict[m])
    
    # Compute reflectance over 401 equally spaced wavelengths
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
    
    # Define discrete thickness options: quarter-wave optical thicknesses
    # Optical thickness = n * d, so d = OT / n
    # We'll use a set of discrete optical thickness values
    # Range: from ~50 nm to ~200 nm optical thickness, in steps of ~10-20 nm
    # But we need to be careful: total thickness constraint is on physical thickness
    
    # Let's define discrete physical thickness options for each material type
    # For L (n=1.38): quarter-wave at 600nm center => d = 600/(4*1.38) ≈ 108.7 nm
    # For H (n=2.35): quarter-wave at 600nm center => d = 600/(4*2.35) ≈ 63.8 nm
    
    # We'll use a grid of thickness values. Let's pick several options per material.
    # For L: thicknesses from ~50 to ~200 nm in steps of ~10 nm
    # For H: thicknesses from ~30 to ~120 nm in steps of ~5 nm
    
    # But to keep the search space manageable, let's use fewer options.
    # Let's define a set of "thickness indices" that map to actual thicknesses.
    
    # Strategy: Use a small set of discrete thickness values for each material.
    # L thicknesses: [50, 75, 100, 125, 150, 175, 200] nm (7 options)
    # H thicknesses: [30, 50, 70, 90, 110, 130, 150] nm (7 options)
    
    L_thicknesses = np.array([50, 75, 100, 125, 150, 175, 200], dtype=float)
    H_thicknesses = np.array([30, 50, 70, 90, 110, 130, 150], dtype=float)
    
    # Ensure all are within max_layer_nm
    L_thicknesses = L_thicknesses[L_thicknesses <= max_layer_nm]
    H_thicknesses = H_thicknesses[H_thicknesses <= max_layer_nm]
    
    if len(L_thicknesses) == 0:
        L_thicknesses = np.array([max_layer_nm / 2.0])
    if len(H_thicknesses) == 0:
        H_thicknesses = np.array([max_layer_nm / 2.0])
    
    # Number of thickness options per material
    n_L_opts = len(L_thicknesses)
    n_H_opts = len(H_thicknesses)
    
    # GA setup
    pop_size = 150
    max_generations = 80
    crossover_rate = 0.8
    mutation_rate = 0.15
    elite_size = 10
    
    # Chromosome representation:
    # - num_layers: 1 to max_layers (but we'll fix a maximum to simplify)
    # - For each layer: material (0=L, 1=H) and thickness index
    # We'll use a variable-length chromosome, capped at max_layers
    
    # To simplify, let's use a fixed maximum number of layers in the GA, say 12
    # and allow some layers to be "empty" (thickness 0) to effectively reduce the count.
    # But thickness 0 might cause numerical issues. Instead, let's just use up to 12 layers
    # and evaluate with only the non-zero layers.
    
    max_ga_layers = min(max_layers, 12)
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        # Random number of layers: 1 to max_ga_layers
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
    
    # GA loop
    for gen in range(max_generations):
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
            def tournament_select(pop, fit, k=3):
                indices = np.random.choice(len(pop), size=min(k, len(pop)), replace=False)
                best_i = indices[np.argmin([fit[i] for i in indices])]
                return pop[best_i]
            
            parent1 = tournament_select(sorted_pop, sorted_fit)
            parent2 = tournament_select(sorted_pop, sorted_fit)
            
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
        if gen > 20 and best_fitness < 0.5:
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
