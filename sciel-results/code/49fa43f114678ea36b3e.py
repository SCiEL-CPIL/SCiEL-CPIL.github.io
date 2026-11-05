import numpy as np
import time

def _compute_mean_reflectance_batch(thicknesses, labels, problem):
    """
    Compute the mean reflectance in percent for a batch of stacks.
    thicknesses: array of shape (P, N)
    labels: list of 'L' or 'H' of length N
    problem: dict with material indices and band
    Returns: array of shape (P,) with mean reflectance in percent.
    """
    n_amb = problem['ambient_index']
    n_sub = problem['substrate_index']
    n_L = problem['materials']['L']
    n_H = problem['materials']['H']
    band = problem['band_nm']
    
    P, N = thicknesses.shape
    
    # 401 equally spaced wavelengths
    wavelengths = np.linspace(band[0], band[1], 401)
    W = len(wavelengths)
    
    # Precompute n for each layer
    n_layers = np.array([n_L if l == 'L' else n_H for l in labels])
    
    # Initialize M_total for all individuals and wavelengths
    # Shape: (P, W, 2, 2)
    M_total = np.zeros((P, W, 2, 2), dtype=complex)
    M_total[:, :, 0, 0] = 1.0
    M_total[:, :, 0, 1] = 0.0
    M_total[:, :, 1, 0] = 0.0
    M_total[:, :, 1, 1] = 1.0
    
    for j in range(N):
        n_j = n_layers[j]
        # t_j shape: (P,)
        t_j = thicknesses[:, j]
        
        # delta shape: (P, W)
        delta = 2.0 * np.pi * n_j * t_j[:, np.newaxis] / wavelengths[np.newaxis, :]
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # M_j shape: (P, W, 2, 2)
        M_j = np.zeros((P, W, 2, 2), dtype=complex)
        M_j[:, :, 0, 0] = cos_d
        M_j[:, :, 0, 1] = 1j * sin_d / n_j
        M_j[:, :, 1, 0] = 1j * n_j * sin_d
        M_j[:, :, 1, 1] = cos_d
        
        # M_total = M_total @ M_j
        # Using einsum: 'pij,pjk->pik'
        M_total = np.einsum('pij,pjk->pik', M_total, M_j)
        
    # Calculate Admittance Y = C / D
    C = M_total[:, :, 1, 0]
    D = M_total[:, :, 1, 1]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Y = C / D
        
    # Reflectance R = |(Y - n_amb) / (Y + n_amb)|^2
    numerator = Y - n_amb
    denominator = Y + n_amb
    R = np.abs(numerator / denominator)**2
    
    # Mean reflectance in percent for each individual
    return np.mean(R, axis=1) * 100.0

def design(problem):
    """
    Design an antireflection coating using a Genetic Algorithm.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    n_amb = problem['ambient_index']
    n_sub = problem['substrate_index']
    n_L = problem['materials']['L']
    n_H = problem['materials']['H']
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    # Fixed structure: 20 layers, alternating L, H, L, H...
    N = max_layers
    labels = ['L' if i % 2 == 0 else 'H' for i in range(N)]
    
    # Initialize population
    pop_size = 50
    thicknesses = np.random.uniform(0, max_layer_nm, (pop_size, N))
    
    # Ensure total thickness constraint by scaling
    totals = np.sum(thicknesses, axis=1)
    scale_factors = np.where(totals > max_total_nm, max_total_nm / totals, 1.0)
    thicknesses *= scale_factors[:, np.newaxis]
    
    # Evaluate initial population
    scores = _compute_mean_reflectance_batch(thicknesses, labels, problem)
    best_idx = np.argmin(scores)
    best_thicknesses = thicknesses[best_idx].copy()
    best_score = scores[best_idx]
    
    # GA Parameters
    generations = 200
    mutation_rate = 0.3
    crossover_rate = 0.8
    tournament_size = 5
    mutation_strength = 15.0
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
            
        # Selection: Tournament
        new_pop = np.zeros_like(thicknesses)
        
        # Vectorized selection logic is complex, loop is fine for P=50
        for i in range(pop_size):
            parents = np.random.choice(pop_size, tournament_size, replace=False)
            parent_scores = scores[parents]
            parent1_idx = parents[np.argmin(parent_scores)]
            
            if np.random.rand() < crossover_rate:
                parents2 = np.random.choice(pop_size, tournament_size, replace=False)
                parent_scores2 = scores[parents2]
                parent2_idx = parents2[np.argmin(parent_scores2)]
                
                # Crossover: Uniform
                mask = np.random.rand(N) < 0.5
                new_pop[i][mask] = thicknesses[parent1_idx][mask]
                new_pop[i][~mask] = thicknesses[parent2_idx][~mask]
            else:
                new_pop[i] = thicknesses[parent1_idx]
                
        # Mutation
        # Apply mutation to a subset
        mut_mask = np.random.rand(pop_size) < mutation_rate
        if np.any(mut_mask):
            new_pop[mut_mask] += np.random.normal(0, mutation_strength, (np.sum(mut_mask), N))
            new_pop[mut_mask] = np.clip(new_pop[mut_mask], 0, max_layer_nm)
            
            # Check total thickness constraint for mutated individuals
            totals_mut = np.sum(new_pop[mut_mask], axis=1)
            scale_factors_mut = np.where(totals_mut > max_total_nm, max_total_nm / totals_mut, 1.0)
            new_pop[mut_mask] *= scale_factors_mut[:, np.newaxis]
            
        thicknesses = new_pop
        
        # Evaluate
        scores = _compute_mean_reflectance_batch(thicknesses, labels, problem)
        
        # Update best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_thicknesses = thicknesses[current_best_idx].copy()
            
        # Elitism
        best_in_pop_idx = np.argmin(scores)
        thicknesses[0] = thicknesses[best_in_pop_idx]
        scores[0] = scores[best_in_pop_idx]
        
    return labels, list(best_thicknesses)
