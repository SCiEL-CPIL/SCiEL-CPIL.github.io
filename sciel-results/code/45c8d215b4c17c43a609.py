"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _compute_reflectance(materials, thicknesses, problem, wavelengths):
    """Compute reflectance spectrum using characteristic matrix method (vectorized)."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_layers = len(materials)
    
    if n_layers == 0:
        # Bare substrate
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.full(len(wavelengths), np.abs(r)**2 * 100.0)
    
    n_vals = np.array([problem["materials"][m] for m in materials])
    d_vals = np.array(thicknesses)
    
    # Vectorize over wavelengths
    # For each layer j and wavelength i, compute the layer matrix
    # M_j = [[cos(delta), i*sin(delta)/n_j], [i*n_j*sin(delta), cos(delta)]]
    # delta_j(i) = 2*pi*n_j*d_j/lambda_i
    
    # Build total matrix M = M_1 @ M_2 @ ... @ M_N for each wavelength
    # M is 2x2 complex, so we can stack all wavelengths: shape (n_waves, 2, 2)
    
    n_waves = len(wavelengths)
    # Initialize M as identity for all wavelengths
    M = np.zeros((n_waves, 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for j in range(n_layers):
        n_j = n_vals[j]
        d_j = d_vals[j]
        # delta for all wavelengths
        delta = 2.0 * np.pi * n_j * d_j / wavelengths  # shape (n_waves,)
        cd = np.cos(delta)
        sd = np.sin(delta)
        
        # Layer matrix elements (shape n_waves,)
        m11 = cd
        m12 = 1j * sd / n_j
        m21 = 1j * n_j * sd
        m22 = cd
        
        # M_new = M @ M_j for each wavelength
        # M is (n_waves, 2, 2), M_j is same for all wavelengths but varies with wavelength
        # M_new[i, :, :] = M[i, :, :] @ M_j[i, :, :]
        
        # M @ M_j:
        # new_m11 = m11*M11 + m12*M21
        # new_m12 = m11*M12 + m12*M22
        # new_m21 = m21*M11 + m22*M21
        # new_m22 = m21*M12 + m22*M22
        
        new_m11 = m11 * M[:, 0, 0] + m12 * M[:, 1, 0]
        new_m12 = m11 * M[:, 0, 1] + m12 * M[:, 1, 1]
        new_m21 = m21 * M[:, 0, 0] + m22 * M[:, 1, 0]
        new_m22 = m21 * M[:, 0, 1] + m22 * M[:, 1, 1]
        
        M[:, 0, 0] = new_m11
        M[:, 0, 1] = new_m12
        M[:, 1, 0] = new_m21
        M[:, 1, 1] = new_m22
    
    M11 = M[:, 0, 0]
    M12 = M[:, 0, 1]
    M21 = M[:, 1, 0]
    M22 = M[:, 1, 1]
    
    # Input admittance
    Y_in = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
    
    # Reflection coefficient
    r = (n_amb - Y_in) / (n_amb + Y_in)
    
    R = np.abs(r)**2 * 100.0
    return R

def _evaluate(materials, thicknesses, problem, wavelengths):
    """Compute mean reflectance in percent."""
    R = _compute_reflectance(materials, thicknesses, problem, wavelengths)
    return np.mean(R)

def _is_valid(materials, thicknesses, problem):
    """Check constraints."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return True
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def _crossover(parent1, parent2):
    """Crossover: pick a random split point for the material sequence."""
    len1 = len(parent1[0])
    len2 = len(parent2[0])
    
    if random.random() < 0.5:
        # Use parent1's length
        target_len = len1
        # Take first part from parent1, rest from parent2 (padded/truncated)
        split = random.randint(0, target_len)
        mat1 = parent1[0][:split]
        mat2 = parent2[0][split:split + (target_len - split)]
        if len(mat2) < target_len - split:
            # Pad with random layers
            for _ in range(target_len - split - len(mat2)):
                mat2.append(random.choice(['L', 'H']))
        mat2 = mat2[:target_len - split]
        materials = mat1 + mat2
        
        # Thicknesses: similar approach
        th1 = parent1[1][:split]
        th2 = parent2[1][split:split + (target_len - split)]
        if len(th2) < target_len - split:
            for _ in range(target_len - split - len(th2)):
                th2.append(random.uniform(50, 300))
        th2 = th2[:target_len - split]
        thicknesses = th1 + th2
    else:
        target_len = len2
        split = random.randint(0, target_len)
        mat1 = parent1[0][:split]
        mat2 = parent2[0][split:split + (target_len - split)]
        if len(mat2) < target_len - split:
            for _ in range(target_len - split - len(mat2)):
                mat2.append(random.choice(['L', 'H']))
        mat2 = mat2[:target_len - split]
        materials = mat1 + mat2
        
        th1 = parent1[1][:split]
        th2 = parent2[1][split:split + (target_len - split)]
        if len(th2) < target_len - split:
            for _ in range(target_len - split - len(th2)):
                th2.append(random.uniform(50, 300))
        th2 = th2[:target_len - split]
        thicknesses = th1 + th2
    
    return materials, thicknesses

def _mutation(materials, thicknesses, problem):
    """Mutation: random changes to material pattern and/or thicknesses."""
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    materials = list(materials)
    thicknesses = list(thicknesses)
    
    # Mutate material pattern
    if materials:
        for i in range(len(materials)):
            if random.random() < 0.1:
                materials[i] = 'L' if materials[i] == 'H' else 'H'
    
    # Mutate thicknesses
    for i in range(len(thicknesses)):
        if random.random() < 0.3:
            if random.random() < 0.5:
                # Gaussian perturbation
                thicknesses[i] += random.gauss(0, 30)
            else:
                # Random reset
                thicknesses[i] = random.uniform(20, 350)
            thicknesses[i] = max(0, min(max_layer_nm, thicknesses[i]))
    
    # Occasionally add or remove a layer
    if random.random() < 0.1:
        if materials and len(materials) < problem["max_layers"]:
            # Add a layer
            pos = random.randint(0, len(materials))
            materials.insert(pos, random.choice(['L', 'H']))
            thicknesses.insert(pos, random.uniform(50, 250))
        elif len(materials) > 1:
            # Remove a layer
            pos = random.randint(0, len(materials) - 1)
            materials.pop(pos)
            thicknesses.pop(pos)
    
    # Enforce total thickness constraint
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    return materials, thicknesses

def design(problem):
    """Design an antireflection coating using genetic algorithm."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Initial population
    pop_size = 30
    population = []
    scores = []
    
    # Add some good initial guesses
    # Quarter-wave stack patterns
    center_wl = 0.5 * (band[0] + band[1])
    
    # Single layer
    population.append((["L"], [center_wl / (4.0 * n_L)]))
    population.append((["H"], [center_wl / (4.0 * n_H)]))
    
    # Two-layer: L-H or H-L
    population.append((["L", "H"], [center_wl/(4*n_L), center_wl/(4*n_H)]))
    population.append((["H", "L"], [center_wl/(4*n_H), center_wl/(4*n_L)]))
    
    # Three-layer: L-H-L or H-L-H
    population.append((["L", "H", "L"], [center_wl/(4*n_L), center_wl/(4*n_H), center_wl/(4*n_L)]))
    population.append((["H", "L", "H"], [center_wl/(4*n_H), center_wl/(4*n_L), center_wl/(4*n_H)]))
    
    # Four-layer: L-H-L-H or H-L-H-L
    population.append((["L", "H", "L", "H"], [center_wl/(4*n_L), center_wl/(4*n_H), center_wl/(4*n_L), center_wl/(4*n_H)]))
    population.append((["H", "L", "H", "L"], [center_wl/(4*n_H), center_wl/(4*n_L), center_wl/(4*n_H), center_wl/(4*n_L)]))
    
    # Five-layer
    population.append((["L", "H", "L", "H", "L"], [center_wl/(4*n_L), center_wl/(4*n_H), center_wl/(4*n_L), center_wl/(4*n_H), center_wl/(4*n_L)]))
    population.append((["H", "L", "H", "L", "H"], [center_wl/(4*n_H), center_wl/(4*n_L), center_wl/(4*n_H), center_wl/(4*n_L), center_wl/(4*n_H)]))
    
    # Random individuals
    while len(population) < pop_size:
        n_layers = random.randint(1, min(8, max_layers))
        materials = [random.choice(['L', 'H']) for _ in range(n_layers)]
        thicknesses = [random.uniform(50, 300) for _ in range(n_layers)]
        # Enforce constraints
        total = sum(thicknesses)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicknesses = [t * scale for t in thicknesses]
        population.append((materials, thicknesses))
    
    # Evaluate initial population
    for mat, th in population:
        if _is_valid(mat, th, problem):
            score = _evaluate(mat, th, problem, wavelengths)
        else:
            score = 100.0  # Penalty
        scores.append(score)
    
    best_idx = np.argmin(scores)
    best_materials = population[best_idx][0]
    best_thicknesses = population[best_idx][1]
    best_score = scores[best_idx]
    
    # Genetic algorithm loop
    gen = 0
    while time.time() < deadline:
        gen += 1
        
        # Sort by score
        sorted_indices = np.argsort(scores)
        population = [population[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Update best
        if scores[0] < best_score:
            best_score = scores[0]
            best_materials = population[0][0]
            best_thicknesses = population[0][1]
        
        # Create new population
        new_population = []
        new_scores = []
        
        # Elitism: keep top 5
        for i in range(min(5, len(population))):
            new_population.append(population[i])
            new_scores.append(scores[i])
        
        # Fill rest with offspring
        while len(new_population) < pop_size and time.time() < deadline:
            # Tournament selection
            def tournament_select(pop, scores, k=3):
                indices = random.sample(range(len(pop)), min(k, len(pop)))
                best_i = min(indices, key=lambda i: scores[i])
                return pop[best_i]
            
            p1 = tournament_select(population, scores)
            p2 = tournament_select(population, scores)
            
            # Crossover
            child_mat, child_th = _crossover(p1, p2)
            
            # Mutation
            child_mat, child_th = _mutation(child_mat, child_th, problem)
            
            # Check validity
            if _is_valid(child_mat, child_th, problem):
                score = _evaluate(child_mat, child_th, problem, wavelengths)
            else:
                score = 100.0
            
            new_population.append((child_mat, child_th))
            new_scores.append(score)
        
        population = new_population
        scores = new_scores
        
        # Occasionally add a random individual for diversity
        if len(population) < pop_size and random.random() < 0.3:
            n_layers = random.randint(1, min(10, max_layers))
            materials = [random.choice(['L', 'H']) for _ in range(n_layers)]
            thicknesses = [random.uniform(30, 350) for _ in range(n_layers)]
            total = sum(thicknesses)
            if total > max_total_nm:
                scale = max_total_nm / total
                thicknesses = [t * scale for t in thicknesses]
            if _is_valid(materials, thicknesses, problem):
                score = _evaluate(materials, thicknesses, problem, wavelengths)
            else:
                score = 100.0
            population.append((materials, thicknesses))
            scores.append(score)
    
    # Final check
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Fallback to a known valid solution
        center_wl = 0.5 * (band[0] + band[1])
        best_materials = ["L"]
        best_thicknesses = [center_wl / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
