"""Broadband antireflection coating design via Hybrid GA-SA with variable structure."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _vectorized_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance fraction for a single design across all wavelengths.
    Uses vectorized operations over the wavelength axis.
    
    thicknesses: array of physical thicknesses in nm
    materials: array of 'L' or 'H' strings
    wls: array of wavelengths in nm
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return float(r * r)
    
    # Pre-compute refractive indices for each layer
    n_layers_arr = np.array([n_H if m == 'H' else n_L for m in materials])
    
    # M = Identity
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for i in range(n_layers):
        n = n_layers_arr[i]
        d = thicknesses[i]
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(b)
        
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        # M = M @ L
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Admittance: Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.real(r * r.conjugate())
    
    return float(np.mean(R))

def _evaluate_design(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm):
    """
    Evaluate a design. Returns reflectance percentage if valid, inf otherwise.
    """
    if len(thicknesses) == 0:
        return float('inf')
    
    # Check constraints
    if any(t < 0 or t > max_layer_nm for t in thicknesses):
        return float('inf')
    if sum(thicknesses) > max_total_nm:
        return float('inf')
        
    R_frac = _vectorized_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls)
    return R_frac * 100.0

def _polish_with_sa(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm, 
                    start_temp=10.0, cool_rate=0.95, max_iter=50, time_budget=1.0):
    """
    Polish a design using simulated annealing on thicknesses only.
    """
    current_thicknesses = np.array(thicknesses, dtype=float)
    current_score = _evaluate_design(current_thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
    if not np.isfinite(current_score):
        return thicknesses, materials, current_score
    
    best_thicknesses = current_thicknesses.copy()
    best_score = current_score
    temp = start_temp
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Perturb one random layer thickness
        idx = np.random.randint(0, len(current_thicknesses))
        new_thicknesses = current_thicknesses.copy()
        new_thicknesses[idx] += np.random.normal(0, 10.0)
        new_thicknesses[idx] = np.clip(new_thicknesses[idx], 0, max_layer_nm)
        
        new_score = _evaluate_design(new_thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
        
        if not np.isfinite(new_score):
            temp *= cool_rate
            continue
        
        delta = new_score - current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current_thicknesses = new_thicknesses
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_thicknesses = current_thicknesses.copy()
        
        temp *= cool_rate
    
    return best_thicknesses, materials, best_score

def design(problem):
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 401 equally spaced wavelengths
    wls = np.linspace(wl_min, wl_max, 401)
    wl_center = 0.5 * (wl_min + wl_max)
    
    # Initial best: Single quarter-wave L layer
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    # Fallback solution
    best_score = float('inf')
    best_thicknesses = [qw_L]
    best_materials = ['L']
    
    # Evaluate fallback
    score = _evaluate_design(best_thicknesses, best_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
    if score < best_score:
        best_score = score
    
    # GA Parameters
    pop_size = 30
    max_generations = 100
    elite_size = 5
    mutation_rate = 0.3
    
    # Initialize population
    # Each individual: (materials_list, thicknesses_array)
    population = []
    for i in range(pop_size):
        # Random number of layers between 3 and 10
        n_layers = np.random.randint(3, 11)
        # Random material sequence
        materials = [np.random.choice(['L', 'H']) for _ in range(n_layers)]
        # Random thicknesses around quarter-wave
        thicknesses = np.zeros(n_layers)
        for j in range(n_layers):
            if materials[j] == 'H':
                thicknesses[j] = qw_H + np.random.uniform(-20, 20)
            else:
                thicknesses[j] = qw_L + np.random.uniform(-20, 20)
        thicknesses = np.clip(thicknesses, 0, max_layer_nm)
        
        # Ensure total thickness constraint
        if np.sum(thicknesses) > max_total_nm:
            thicknesses *= max_total_nm / np.sum(thicknesses)
        
        population.append((materials, thicknesses))
    
    # Evaluate initial population
    pop_scores = []
    for materials, thicknesses in population:
        s = _evaluate_design(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
        pop_scores.append(s)
    
    # Sort population
    order = np.argsort(pop_scores)
    population = [population[i] for i in order]
    pop_scores = [pop_scores[i] for i in order]
    
    # Update best
    if pop_scores[0] < best_score:
        best_score = pop_scores[0]
        best_materials = population[0][0]
        best_thicknesses = population[0][1].tolist()
    
    # GA Loop
    for gen in range(max_generations):
        if time.time() > deadline - 10.0:
            break
        
        # Elitism: keep top elite_size
        new_population = [population[i] for i in range(min(elite_size, len(population)))]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Select parents via tournament
            def tournament_select(pop, scores, k=3):
                indices = np.random.randint(0, len(pop), k)
                best_idx = indices[np.argmin([scores[i] for i in indices])]
                return pop[best_idx]
            
            parent1 = tournament_select(population, pop_scores)
            parent2 = tournament_select(population, pop_scores)
            
            # Crossover: combine materials and thicknesses
            # Random crossover point
            cross_point = np.random.randint(1, max(len(parent1[0]), len(parent2[0])))
            
            # New materials
            new_materials = parent1[0][:cross_point] + parent2[0][cross_point:]
            
            # New thicknesses: interpolate
            len1 = len(parent1[1])
            len2 = len(parent2[1])
            new_len = len(new_materials)
            new_thicknesses = np.zeros(new_len)
            for j in range(new_len):
                if j < len1:
                    new_thicknesses[j] = parent1[1][j]
                else:
                    new_thicknesses[j] = parent2[1][j - len1]
            
            # Mutation: randomly change materials and thicknesses
            for j in range(len(new_materials)):
                if np.random.rand() < mutation_rate:
                    new_materials[j] = 'L' if new_materials[j] == 'H' else 'H'
                if np.random.rand() < mutation_rate:
                    new_thicknesses[j] += np.random.normal(0, 15.0)
                    new_thicknesses[j] = np.clip(new_thicknesses[j], 0, max_layer_nm)
            
            # Ensure total thickness constraint
            if np.sum(new_thicknesses) > max_total_nm:
                new_thicknesses *= max_total_nm / np.sum(new_thicknesses)
            
            new_population.append((new_materials, new_thicknesses))
        
        # Evaluate new population
        new_pop_scores = []
        for materials, thicknesses in new_population:
            s = _evaluate_design(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
            new_pop_scores.append(s)
        
        # Sort new population
        order = np.argsort(new_pop_scores)
        new_population = [new_population[i] for i in order]
        new_pop_scores = [new_pop_scores[i] for i in order]
        
        # Update best
        if new_pop_scores[0] < best_score:
            best_score = new_pop_scores[0]
            best_materials = new_population[0][0]
            best_thicknesses = new_population[0][1].tolist()
        
        # Polish top individuals with SA
        if time.time() < deadline - 20.0:
            for i in range(min(3, len(new_population))):
                materials, thicknesses = new_population[i]
                polished_t, polished_m, polished_score = _polish_with_sa(
                    thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls, 
                    max_total_nm, max_layer_nm, time_budget=2.0
                )
                if polished_score < new_pop_scores[i]:
                    new_population[i] = (polished_m, polished_t)
                    new_pop_scores[i] = polished_score
                    if polished_score < best_score:
                        best_score = polished_score
                        best_materials = polished_m
                        best_thicknesses = polished_t.tolist()
        
        # Next generation
        population = new_population
        pop_scores = new_pop_scores
        
        # Early stopping if converged
        if pop_scores[0] < 0.01:
            break
    
    # Final validity check and cleanup
    # Ensure we return a valid design
    if not best_thicknesses or any(t < 0 or t > max_layer_nm for t in best_thicknesses) or sum(best_thicknesses) > max_total_nm:
        # Fallback to single layer
        return ["L"], [qw_L]
    
    # Remove zero-thickness layers if any (though they should be clamped to >=0, tiny values are okay)
    # The problem allows 0 thickness? "every thickness must lie in [0, max_layer_nm]"
    # So 0 is allowed.
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
