# EVOLVE-BLOCK-START
import numpy as np

def _compute_reflectance_vec(layer_types, thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """
    Compute normal-incidence reflectance using the characteristic matrix method.
    """
    n_sub = substrate_index
    n_amb = ambient_index
    
    # Filter out layers with zero thickness
    active_indices = [i for i, t in enumerate(thicknesses) if t > 1e-9]
    if not active_indices:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.full(len(wavelengths), np.abs(r)**2)
        
    active_types = [layer_types[i] for i in active_indices]
    active_thicknesses = [thicknesses[i] for i in active_indices]
    
    reflectances = np.zeros(len(wavelengths))
    
    # Pre-compute indices for active layers
    n_active = [materials[t] for t in active_types]
    
    for i, lam in enumerate(wavelengths):
        # Initialize total matrix as identity
        M_total = np.array([[1.0 + 0.0j, 0.0 + 0.0j],
                           [0.0 + 0.0j, 1.0 + 0.0j]])
        
        for j in range(len(active_types)):
            d = active_thicknesses[j]
            n = n_active[j]
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
    # Use coarse grid for GA, fine grid for final check
    wavelengths_coarse = np.linspace(band_nm[0], band_nm[1], 50)
    wavelengths_fine = np.linspace(band_nm[0], band_nm[1], 401)
    
    n_L = materials['L']
    n_H = materials['H']
    
    def get_layer_types(start_type):
        types = []
        for i in range(max_layers):
            if i % 2 == 0:
                types.append(start_type)
            else:
                types.append('H' if start_type == 'L' else 'L')
        return types
    
    def evaluate_individual(start_type, thicknesses, wavelengths):
        """
        Evaluate an individual: return mean reflectance in percent.
        """
        # Check validity
        if sum(thicknesses) > max_total_nm + 1e-9:
            return float('inf')
        for t in thicknesses:
            if t < -1e-9 or t > max_layer_nm + 1e-9:
                return float('inf')
        
        layer_types = get_layer_types(start_type)
        refl = _compute_reflectance_vec(layer_types, thicknesses, materials, ambient_index, substrate_index, wavelengths)
        return np.mean(refl) * 100.0

    def random_individual():
        """Generate a random valid individual."""
        start_type = np.random.choice(['L', 'H'])
        thicknesses = [0.0] * max_layers
        num_active = np.random.randint(1, 8)
        active_indices = np.random.choice(max_layers, num_active, replace=False)
        for idx in active_indices:
            thicknesses[idx] = np.random.uniform(10, max_layer_nm)
        
        total = sum(thicknesses)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicknesses = [t * scale for t in thicknesses]
            
        return start_type, thicknesses

    def crossover(parent1, parent2):
        st1, t1 = parent1
        st2, t2 = parent2
        
        start_type = st1 if np.random.rand() < 0.5 else st2
        
        child_t = []
        for i in range(max_layers):
            if np.random.rand() < 0.5:
                child_t.append(t1[i])
            else:
                child_t.append(t2[i])
                
        total = sum(child_t)
        if total > max_total_nm:
            scale = max_total_nm / total
            child_t = [t * scale for t in child_t]
            
        child_t = [min(max(t, 0.0), max_layer_nm) for t in child_t]
        
        return start_type, child_t

    def mutate(individual):
        start_type, t = individual
        t = list(t)
        
        if np.random.rand() < 0.1:
            start_type = 'H' if start_type == 'L' else 'L'
            
        for i in range(max_layers):
            r = np.random.rand()
            if r < 0.05:
                t[i] = 0.0
            elif r < 0.15:
                t[i] = np.random.uniform(10, max_layer_nm)
            elif r < 0.4:
                if t[i] > 0:
                    delta = np.random.uniform(-50, 50)
                    t[i] = max(0.0, min(max_layer_nm, t[i] + delta))
                else:
                    t[i] = np.random.uniform(0, max_layer_nm)
            
        total = sum(t)
        if total > max_total_nm:
            scale = max_total_nm / total
            t = [x * scale for x in t]
            
        return start_type, t

    pop_size = 50
    population = [random_individual() for _ in range(pop_size)]
    
    # Heuristic starts
    d_qw_L = 600.0 / (4.0 * n_L)
    d_qw_H = 600.0 / (4.0 * n_H)
    population[0] = ('L', [d_qw_L] + [0.0]*(max_layers-1))
    population[1] = ('H', [d_qw_H] + [0.0]*(max_layers-1))
    population[2] = ('L', [d_qw_L, d_qw_H] + [0.0]*(max_layers-2))
    population[3] = ('H', [d_qw_H, d_qw_L] + [0.0]*(max_layers-2))
    
    scores = [evaluate_individual(st, t, wavelengths_coarse) for st, t in population]
    
    best_ind = population[np.argmin(scores)]
    best_score = min(scores)
    
    num_generations = 50
    
    for gen in range(num_generations):
        sorted_indices = np.argsort(scores)
        sorted_pop = [population[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        best_ind = sorted_pop[0]
        best_score = sorted_scores[0]
        
        new_pop = []
        elite_size = 5
        for i in range(min(elite_size, len(sorted_pop))):
            new_pop.append(sorted_pop[i])
        
        while len(new_pop) < pop_size:
            def tournament_select(pop, sc, k=3):
                indices = np.random.choice(len(pop), k, replace=False)
                best_idx = indices[np.argmin([sc[i] for i in indices])]
                return pop[best_idx]
            
            parent1 = tournament_select(sorted_pop, sorted_scores)
            parent2 = tournament_select(sorted_pop, sorted_scores)
            
            if np.random.rand() < 0.7:
                child = crossover(parent1, parent2)
            else:
                child = (parent1[0], list(parent1[1]))
            
            child = mutate(child)
            new_pop.append(child)
        
        population = new_pop
        scores = [evaluate_individual(st, t, wavelengths_coarse) for st, t in population]
        
        if best_score < 0.1:
            break
            
    # Final evaluation with fine grid to ensure accuracy
    final_score = evaluate_individual(best_ind[0], best_ind[1], wavelengths_fine)
    
    start_type, thicknesses = best_ind
    layer_types = get_layer_types(start_type)
    
    final_types = []
    final_thicknesses = []
    for i in range(max_layers):
        if thicknesses[i] > 1e-9:
            final_types.append(layer_types[i])
            final_thicknesses.append(thicknesses[i])
            
    return final_types, final_thicknesses
# EVOLVE-BLOCK-END
