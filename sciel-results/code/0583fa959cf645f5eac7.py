"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START

def _characteristic_matrix(n, d, lam):
    """Return the 2x2 characteristic matrix for a single layer."""
    beta = 2.0 * np.pi * n * d / lam
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    # At normal incidence, B = n (admittance), Y = 1/n
    return np.array([[cos_b, 1.0j * sin_b / n],
                     [1.0j * n * sin_b, cos_b]])


def _reflectance_stack(materials, thicknesses, substrate_index, wavelengths):
    """
    Compute reflectance (as fraction, 0-1) at each wavelength using characteristic matrix method.
    materials: list of 'L' or 'H'
    thicknesses: list of physical thicknesses in nm
    """
    n_L = 1.38
    n_H = 2.35
    n_amb = 1.0
    
    R = np.zeros(len(wavelengths))
    for i, lam in enumerate(wavelengths):
        # Start with substrate admittance
        M = np.array([[1.0, 0.0], [0.0, 1.0]])
        # Multiply from substrate side to air side
        for j in range(len(materials) - 1, -1, -1):
            n = n_L if materials[j] == 'L' else n_H
            d = thicknesses[j]
            Mj = _characteristic_matrix(n, d, lam)
            M = M @ Mj
        
        # Final matrix includes substrate
        n_sub = substrate_index
        beta_sub = 0.0  # substrate is semi-infinite, but we treat it as a layer with beta=0 effectively
        # Actually, for semi-infinite substrate, the characteristic matrix is:
        # [[1, 0], [i/n_sub, 1]] ... wait, let me use the standard formula.
        
        # Standard approach: 
        # The characteristic matrix M_total = M_1 @ M_2 @ ... @ M_N
        # where M_j is for layer j.
        # Then Y_in = (M[0,0] + M[0,1]*n_sub) / (M[1,0] + M[1,1]*n_sub)
        # R = |(Y_in - n_amb) / (Y_in + n_amb)|^2
        
        Y_in = (M[0, 0] + M[0, 1] * n_sub) / (M[1, 0] + M[1, 1] * n_sub)
        R[i] = np.abs((Y_in - n_amb) / (Y_in + n_amb)) ** 2
    return R


def _evaluate(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    n_waves = 401
    wavelengths = np.linspace(band[0], band[1], n_waves)
    substrate_index = problem["substrate_index"]
    R = _reflectance_stack(materials, thicknesses, substrate_index, wavelengths)
    return np.mean(R) * 100.0


def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return False
    total = sum(thicknesses)
    if total > max_total_nm + 1e-9:
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    for m in materials:
        if m not in ('L', 'H'):
            return False
    return True


def design(problem):
    """
    Design a broadband antireflection coating using a genetic algorithm
    with discrete quarter-wave thicknesses.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    substrate_index = problem["substrate_index"]
    
    # Discrete thickness values: quarter-wave optical thicknesses at various wavelengths
    # OT = lambda/4, physical thickness = OT / n
    # Let's use a set of discrete physical thicknesses
    # For L (n=1.38): quarter-wave at 400nm -> 72.46nm, at 800nm -> 144.93nm
    # For H (n=2.35): quarter-wave at 400nm -> 42.55nm, at 800nm -> 85.11nm
    
    # Let's create a discrete set of thickness values
    # Use steps of ~10nm for finer control
    L_thicknesses = np.arange(10, max_layer_nm + 1, 10.0)
    H_thicknesses = np.arange(10, max_layer_nm + 1, 10.0)
    
    # Also include some quarter-wave specific values
    center_wl = 0.5 * (band[0] + band[1])
    qw_L = center_wl / (4.0 * 1.38)
    qw_H = center_wl / (4.0 * 2.35)
    
    # Add quarter-wave values to the discrete sets
    L_thicknesses = np.unique(np.concatenate([L_thicknesses, [qw_L]]))
    H_thicknesses = np.unique(np.concatenate([H_thicknesses, [qw_H]]))
    
    # Filter to valid range
    L_thicknesses = L_thicknesses[(L_thicknesses >= 0) & (L_thicknesses <= max_layer_nm)]
    H_thicknesses = H_thicknesses[(H_thicknesses >= 0) & (H_thicknesses <= max_layer_nm)]
    
    # Genetic algorithm setup
    pop_size = 50
    n_layers_range = (3, max_layers)
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        n_layers = np.random.randint(n_layers_range[0], n_layers_range[1] + 1)
        materials = ['L' if np.random.rand() < 0.5 else 'H' for _ in range(n_layers)]
        thicknesses = []
        for m in materials:
            if m == 'L':
                thicknesses.append(float(np.random.choice(L_thicknesses)))
            else:
                thicknesses.append(float(np.random.choice(H_thicknesses)))
        # Check total thickness
        while sum(thicknesses) > max_total_nm and thicknesses:
            # Remove a random layer
            idx = np.random.randint(len(thicknesses))
            materials.pop(idx)
            thicknesses.pop(idx)
        if len(materials) > 0:
            population.append((materials, thicknesses))
    
    # Evaluate initial population
    scores = []
    for materials, thicknesses in population:
        if _is_valid(materials, thicknesses, problem):
            score = _evaluate(materials, thicknesses, problem)
        else:
            score = 100.0
        scores.append(score)
    
    best_idx = np.argmin(scores)
    best_materials = population[best_idx][0][:]
    best_thicknesses = population[best_idx][1][:]
    best_score = scores[best_idx]
    
    # Genetic algorithm loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Selection: tournament selection
        def select():
            i = np.random.randint(pop_size)
            j = np.random.randint(pop_size)
            if scores[i] <= scores[j]:
                return population[i]
            else:
                return population[j]
        
        new_population = []
        # Elitism: keep best individuals
        sorted_indices = np.argsort(scores)
        n_elite = max(2, pop_size // 10)
        for i in range(n_elite):
            new_population.append(population[sorted_indices[i]])
        
        # Generate new individuals
        while len(new_population) < pop_size:
            parent1 = select()
            parent2 = select()
            
            # Crossover
            if np.random.rand() < 0.7:
                # Single-point crossover
                len1 = len(parent1[0])
                len2 = len(parent2[0])
                if len1 == len2:
                    point = np.random.randint(1, len1) if len1 > 1 else 0
                    child_materials = parent1[0][:point] + parent2[0][point:]
                    child_thicknesses = parent1[1][:point] + parent2[1][point:]
                else:
                    # Take from the longer one
                    if len1 > len2:
                        child_materials = parent1[0][:]
                        child_thicknesses = parent1[1][:]
                    else:
                        child_materials = parent2[0][:]
                        child_thicknesses = parent2[1][:]
            else:
                child_materials = parent1[0][:]
                child_thicknesses = parent1[1][:]
            
            # Mutation
            # Mutate material
            for i in range(len(child_materials)):
                if np.random.rand() < 0.1:
                    child_materials[i] = 'L' if child_materials[i] == 'H' else 'H'
            
            # Mutate thickness
            for i in range(len(child_thicknesses)):
                if np.random.rand() < 0.1:
                    if child_materials[i] == 'L':
                        child_thicknesses[i] = float(np.random.choice(L_thicknesses))
                    else:
                        child_thicknesses[i] = float(np.random.choice(H_thicknesses))
            
            # Mutate number of layers
            if np.random.rand() < 0.05:
                # Add a layer
                if len(child_materials) < max_layers:
                    m = 'L' if np.random.rand() < 0.5 else 'H'
                    if m == 'L':
                        t = float(np.random.choice(L_thicknesses))
                    else:
                        t = float(np.random.choice(H_thicknesses))
                    if sum(child_thicknesses) + t <= max_total_nm:
                        child_materials.append(m)
                        child_thicknesses.append(t)
            elif np.random.rand() < 0.05:
                # Remove a layer
                if len(child_materials) > 1:
                    idx = np.random.randint(len(child_materials))
                    child_materials.pop(idx)
                    child_thicknesses.pop(idx)
            
            # Ensure validity
            if _is_valid(child_materials, child_thicknesses, problem):
                new_population.append((child_materials, child_thicknesses))
            else:
                # Keep parent if child is invalid
                new_population.append(parent1)
        
        # Evaluate new population
        population = new_population
        for i, (materials, thicknesses) in enumerate(population):
            if _is_valid(materials, thicknesses, problem):
                scores[i] = _evaluate(materials, thicknesses, problem)
            else:
                scores[i] = 100.0
        
        # Update best
        best_idx = np.argmin(scores)
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_materials = population[best_idx][0][:]
            best_thicknesses = population[best_idx][1][:]
    
    # Final polish: try small perturbations around the best solution
    if time.time() < deadline - 5.0:
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            mat = best_materials[:]
            thk = best_thicknesses[:]
            
            # Perturb one thickness
            if len(thk) > 0:
                idx = np.random.randint(len(thk))
                if mat[idx] == 'L':
                    thk[idx] = float(np.random.choice(L_thicknesses))
                else:
                    thk[idx] = float(np.random.choice(H_thicknesses))
            
            if _is_valid(mat, thk, problem):
                score = _evaluate(mat, thk, problem)
                if score < best_score:
                    best_score = score
                    best_materials = mat[:]
                    best_thicknesses = thk[:]
    
    # Ensure final result is valid
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Fallback to a simple valid design
        center = 0.5 * (band[0] + band[1])
        return ["L"], [center / (4.0 * 1.38)]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
