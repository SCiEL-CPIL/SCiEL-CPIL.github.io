"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _compute_reflectance(thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """
    Compute normal-incidence reflectance using the characteristic matrix method.
    thicknesses: list of physical thicknesses in nm
    materials: dict with 'L' and 'H' indices
    wavelengths: array of wavelengths in nm
    Returns: array of reflectances (fraction, not percent)
    """
    n_L = materials['L']
    n_H = materials['H']
    n_sub = substrate_index
    n_amb = ambient_index
    
    # Convert wavelengths to radians (phase)
    # Phase = 2 * pi * n * d / lambda
    # We'll compute the characteristic matrix for each layer
    
    num_layers = len(thicknesses)
    if num_layers == 0:
        # Bare substrate
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.abs(r)**2
    
    # Build the total characteristic matrix M = M1 * M2 * ... * MN
    # For a single layer with index n and optical thickness nd:
    # M = [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # where delta = 2*pi*n*d/lambda
    
    # We'll compute M for each wavelength
    reflectances = np.zeros(len(wavelengths))
    
    for i, lam in enumerate(wavelengths):
        # Initialize total matrix as identity
        M_total = np.array([[1.0 + 0.0j, 0.0 + 0.0j],
                           [0.0 + 0.0j, 1.0 + 0.0j]])
        
        for j in range(num_layers):
            d = thicknesses[j]
            # Determine index based on layer type - we need to track which is L/H
            # Actually, we need to pass the layer types too. Let me restructure.
            pass
        
        # This approach won't work without layer types. Let me rewrite.
        pass
    
    # Let me restructure: the function should take layer types and thicknesses
    return reflectances

def _compute_reflectance_full(layer_types, thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """
    Compute normal-incidence reflectance using the characteristic matrix method.
    layer_types: list of 'L' or 'H'
    thicknesses: list of physical thicknesses in nm
    materials: dict with 'L' and 'H' indices
    wavelengths: array of wavelengths in nm
    Returns: array of reflectances (fraction, not percent)
    """
    n_sub = substrate_index
    n_amb = ambient_index
    
    num_layers = len(thicknesses)
    if num_layers == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.abs(r)**2
    
    reflectances = np.zeros(len(wavelengths))
    
    for i, lam in enumerate(wavelengths):
        # Initialize total matrix as identity
        M_total = np.array([[1.0 + 0.0j, 0.0 + 0.0j],
                           [0.0 + 0.0j, 1.0 + 0.0j]])
        
        for j in range(num_layers):
            d = thicknesses[j]
            n = materials[layer_types[j]]
            delta = 2.0 * np.pi * n * d / lam
            
            c = np.cos(delta)
            s = np.sin(delta)
            
            # Characteristic matrix for this layer
            M_layer = np.array([[c, 1j * s / n],
                               [1j * n * s, c]])
            
            M_total = M_total @ M_layer
        
        # Admittance
        Y = M_total[1, 0] / M_total[0, 0]
        
        # Reflectance
        r = (n_amb - Y) / (n_amb + Y)
        reflectances[i] = np.abs(r)**2
    
    return reflectances

def design(problem):
    """
    Design an antireflection coating using a genetic algorithm.
    """
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Genetic algorithm parameters
    pop_size = 50
    num_generations = 30
    mutation_rate = 0.3
    crossover_rate = 0.7
    elite_size = 5
    
    n_L = materials['L']
    n_H = materials['H']
    
    def random_individual():
        """Generate a random valid individual."""
        num_layers = np.random.randint(1, max_layers + 1)
        # Alternating L and H, starting with either
        start_type = np.random.choice(['L', 'H'])
        layer_types = []
        for i in range(num_layers):
            if i % 2 == 0:
                layer_types.append(start_type)
            else:
                layer_types.append('H' if start_type == 'L' else 'L')
        
        # Random thicknesses, ensuring sum <= max_total_nm
        # Use Dirichlet-like distribution
        raw = np.random.rand(num_layers)
        raw = raw / raw.sum()
        thicknesses = raw * max_total_nm
        # Clip to max_layer_nm
        thicknesses = np.clip(thicknesses, 0, max_layer_nm)
        
        # Ensure total <= max_total_nm
        total = thicknesses.sum()
        if total > max_total_nm:
            thicknesses = thicknesses * (max_total_nm / total)
        
        return layer_types, thicknesses.tolist()
    
    def evaluate(individual):
        """Evaluate an individual: return mean reflectance in percent."""
        layer_types, thicknesses = individual
        refl = _compute_reflectance_full(layer_types, thicknesses, materials, ambient_index, substrate_index, wavelengths)
        return np.mean(refl) * 100.0
    
    def is_valid(individual):
        """Check if individual is valid."""
        layer_types, thicknesses = individual
        if len(layer_types) != len(thicknesses):
            return False
        if len(layer_types) > max_layers:
            return False
        if len(layer_types) == 0:
            return False
        for t in thicknesses:
            if t < -1e-9 or t > max_layer_nm + 1e-9:
                return False
        if sum(thicknesses) > max_total_nm + 1e-9:
            return False
        # Check alternating pattern
        for i in range(1, len(layer_types)):
            if layer_types[i] == layer_types[i-1]:
                return False
        return True
    
    def crossover(ind1, ind2):
        """Uniform crossover with repair."""
        lt1, t1 = ind1
        lt2, t2 = ind2
        
        max_len = max(len(lt1), len(lt2))
        child_lt = []
        child_t = []
        
        for i in range(max_len):
            if np.random.rand() < 0.5:
                if i < len(lt1):
                    child_lt.append(lt1[i])
                    child_t.append(t1[i])
            else:
                if i < len(lt2):
                    child_lt.append(lt2[i])
                    child_t.append(t2[i])
        
        # Ensure alternating pattern
        fixed_lt = []
        fixed_t = []
        if len(child_lt) > 0:
            fixed_lt.append(child_lt[0])
            fixed_t.append(child_t[0])
            for i in range(1, len(child_lt)):
                if child_lt[i] != fixed_lt[-1]:
                    fixed_lt.append(child_lt[i])
                    fixed_t.append(child_t[i])
        
        # Truncate to max_layers
        if len(fixed_lt) > max_layers:
            fixed_lt = fixed_lt[:max_layers]
            fixed_t = fixed_t[:max_layers]
        
        # Scale thicknesses to fit within max_total_nm
        if len(fixed_t) > 0:
            total = sum(fixed_t)
            if total > max_total_nm:
                fixed_t = [t * (max_total_nm / total) for t in fixed_t]
            # Clip
            fixed_t = [min(max(t, 0.0), max_layer_nm) for t in fixed_t]
        
        return fixed_lt, fixed_t
    
    def mutate(individual):
        """Mutate an individual."""
        lt, t = individual
        lt = list(lt)
        t = list(t)
        
        # Possibly change number of layers
        if np.random.rand() < 0.1:
            # Add or remove a layer
            if len(lt) < max_layers and np.random.rand() < 0.5:
                # Add layer at random position
                pos = np.random.randint(0, len(lt) + 1)
                new_type = 'H' if lt[pos-1] == 'L' else 'L' if pos > 0 else ('H' if lt[0] == 'L' else 'L')
                if pos == 0:
                    new_type = 'H' if lt[0] == 'L' else 'L'
                else:
                    new_type = 'H' if lt[pos-1] == 'L' else 'L'
                lt.insert(pos, new_type)
                t.insert(pos, np.random.uniform(0, max_layer_nm))
            elif len(lt) > 1:
                # Remove a layer
                pos = np.random.randint(0, len(lt))
                lt.pop(pos)
                t.pop(pos)
        
        # Mutate thicknesses
        for i in range(len(t)):
            if np.random.rand() < mutation_rate:
                t[i] = np.random.uniform(0, max_layer_nm)
        
        # Scale to fit
        total = sum(t)
        if total > max_total_nm and total > 0:
            t = [x * (max_total_nm / total) for x in t]
        
        return lt, t
    
    # Initialize population
    population = [random_individual() for _ in range(pop_size)]
    # Ensure all are valid
    valid_pop = []
    for ind in population:
        if is_valid(ind):
            valid_pop.append(ind)
        else:
            valid_pop.append(random_individual())
    
    # Evaluate initial population
    scores = [evaluate(ind) for ind in valid_pop]
    
    best_ind = valid_pop[np.argmin(scores)]
    best_score = min(scores)
    
    for gen in range(num_generations):
        # Sort by score
        sorted_indices = np.argsort(scores)
        sorted_pop = [valid_pop[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        best_ind = sorted_pop[0]
        best_score = sorted_scores[0]
        
        # Create new population
        new_pop = []
        # Elitism
        for i in range(min(elite_size, len(sorted_pop))):
            new_pop.append(sorted_pop[i])
        
        while len(new_pop) < pop_size:
            # Selection (tournament)
            def tournament_select(pop, sc, k=3):
                indices = np.random.choice(len(pop), k, replace=False)
                best_idx = indices[np.argmin([sc[i] for i in indices])]
                return pop[best_idx]
            
            parent1 = tournament_select(sorted_pop, sorted_scores)
            parent2 = tournament_select(sorted_pop, sorted_scores)
            
            if np.random.rand() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = (list(parent1[0]), list(parent1[1]))
            
            child = mutate(child)
            
            # Ensure validity
            if not is_valid(child):
                child = random_individual()
            
            new_pop.append(child)
        
        # Evaluate new population
        valid_pop = new_pop
        scores = [evaluate(ind) for ind in valid_pop]
    
    # Return best individual
    layer_types, thicknesses = best_ind
    return layer_types, thicknesses
# EVOLVE-BLOCK-END
