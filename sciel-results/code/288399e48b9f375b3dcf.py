"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _reflectance_vectorized(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """
    Compute reflectance at all wavelengths using the characteristic matrix method (vectorized).
    """
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    # Initialize total matrix as identity for all wavelengths
    # Shape: (N_wls, 2, 2)
    total = np.eye(2, dtype=complex)
    total = np.broadcast_to(total, (len(wls), 2, 2)).copy()
    
    for mat, thick in zip(materials, thicknesses):
        if thick < 1e-9:
            continue
        n = materials_dict[mat]
        d = thick
        # beta = 2*pi*n*d / wls
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Matrix M for this layer at each wavelength
        # M = [[cos, i*sin/n], [i*n*sin, cos]]
        
        t00 = total[:, 0, 0]
        t01 = total[:, 0, 1]
        t10 = total[:, 1, 0]
        t11 = total[:, 1, 1]
        
        new_t00 = t00 * cos_b + t01 * (1.0j * sin_b / n)
        new_t01 = t00 * (1.0j * sin_b / n) + t01 * cos_b
        new_t10 = t10 * cos_b + t11 * (1.0j * n * sin_b)
        new_t11 = t10 * (1.0j * n * sin_b) + t11 * cos_b
        
        total[:, 0, 0] = new_t00
        total[:, 0, 1] = new_t01
        total[:, 1, 0] = new_t10
        total[:, 1, 1] = new_t11

    # Calculate admittance Y
    # Y = (T[1,0] + T[1,1]*Y_s) / (T[0,0] + T[0,1]*Y_s)
    Y_s = substrate_index
    Y_0 = ambient_index
    
    B = total[:, 0, 0]
    C = total[:, 0, 1]
    D = total[:, 1, 0]
    E = total[:, 1, 1]
    
    Y = (D + E * Y_s) / (B + C * Y_s)
    
    r = (Y_0 - Y) / (Y_0 + Y)
    return np.abs(r) ** 2

def _mean_reflectance(materials, thicknesses, band_nm=(400, 800), n_points=401,
                       ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """Compute mean reflectance in percent over the band."""
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    wls = np.linspace(band_nm[0], band_nm[1], n_points)
    refls = _reflectance_vectorized(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def _ga(problem, deadline):
    """
    Run a Genetic Algorithm to optimize both layer sequence and thicknesses.
    """
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    pop_size = 60
    mutation_rate = 0.15
    crossover_rate = 0.8
    elite_count = 5
    
    # Initial population
    population = []
    for _ in range(pop_size):
        n_layers = random.randint(1, max_layers)
        # Bias towards alternating L/H for better performance
        mats = []
        for i in range(n_layers):
            if random.random() < 0.8:
                mats.append('L' if i % 2 == 0 else 'H')
            else:
                mats.append(random.choice(['L', 'H']))
        
        # Initial thicknesses: random quarter-wave-ish
        thicks = []
        for m in mats:
            n = nL if m == 'L' else nH
            # Random thickness between 0.25 and 1.0 quarter wave at center
            wl_center = 0.5 * (band_nm[0] + band_nm[1])
            qw = wl_center / (4.0 * n)
            t = random.uniform(0.25 * qw, 1.0 * qw)
            thicks.append(min(t, max_layer_nm))
        
        # Ensure total thickness constraint
        total_t = sum(thicks)
        if total_t > max_total_nm:
            scale = max_total_nm / total_t * 0.9
            thicks = [t * scale for t in thicks]
        
        population.append((mats, thicks))
    
    # Evaluate initial population
    scores = []
    for mats, thicks in population:
        if _is_valid(mats, thicks, problem):
            try:
                score = _mean_reflectance(mats, thicks, band_nm, 401, n_amb, n_sub, materials_dict)
            except:
                score = 100.0
        else:
            score = 100.0
        scores.append(score)
    
    best_score = min(scores)
    best_idx = scores.index(best_score)
    best_mats = population[best_idx][0]
    best_thicks = population[best_idx][1]
    
    # GA Loop
    gen = 0
    while time.time() < deadline - 5.0:
        gen += 1
        
        # Sort population by score
        sorted_pop = sorted(zip(scores, population), key=lambda x: x[0])
        
        # Elitism
        new_population = []
        for i in range(elite_count):
            new_population.append(sorted_pop[i][1])
        
        # Selection and Crossover
        while len(new_population) < pop_size:
            if time.time() > deadline - 5.0:
                break
                
            # Tournament selection
            def select_tournament():
                idxs = random.sample(range(len(sorted_pop)), 3)
                best_idx_t = min(idxs, key=lambda i: sorted_pop[i][0])
                return sorted_pop[best_idx_t][1]
            
            parent1 = select_tournament()
            parent2 = select_tournament()
            
            child_mats = parent1[0]
            child_thicks = parent1[1]
            
            if random.random() < crossover_rate and len(parent2[0]) > 0:
                # Crossover: exchange a random segment
                n1 = len(child_mats)
                n2 = len(parent2[0])
                
                # Handle different lengths by taking min or resizing
                min_n = min(n1, n2)
                if min_n > 1:
                    cut = random.randint(1, min_n - 1)
                    # Swap segments
                    new_mats = child_mats[:cut] + parent2[0][cut:cut + (n1-cut)]
                    new_thicks = child_thicks[:cut] + parent2[1][cut:cut + (n1-cut)]
                    
                    # If lengths don't match up, we might have issues, so keep it simple:
                    # Only crossover if lengths are same, otherwise just mutate
                    if len(new_mats) == n1 and len(new_thicks) == n1:
                        child_mats = new_mats
                        child_thicks = new_thicks
            
            # Mutation
            if random.random() < mutation_rate:
                # Mutate one thickness
                if len(child_thicks) > 0:
                    idx = random.randint(0, len(child_thicks) - 1)
                    # Random new thickness
                    child_thicks[idx] = random.uniform(10, max_layer_nm)
                
                # Mutate one material
                if len(child_mats) > 0:
                    idx = random.randint(0, len(child_mats) - 1)
                    child_mats[idx] = 'L' if child_mats[idx] == 'H' else 'H'
                
                # Occasionally add or remove a layer
                if random.random() < 0.1:
                    if len(child_mats) < max_layers:
                        pos = random.randint(0, len(child_mats))
                        child_mats.insert(pos, random.choice(['L', 'H']))
                        child_thicks.insert(pos, random.uniform(10, max_layer_nm))
                    elif len(child_mats) > 1:
                        pos = random.randint(0, len(child_mats) - 1)
                        child_mats.pop(pos)
                        child_thicks.pop(pos)
            
            # Enforce constraints
            if _is_valid(child_mats, child_thicks, problem):
                new_population.append((child_mats, child_thicks))
            else:
                # Repair: clip thicknesses
                fixed_thicks = []
                for t in child_thicks:
                    fixed_thicks.append(min(max(t, 0), max_layer_nm))
                total_t = sum(fixed_thicks)
                if total_t > max_total_nm:
                    scale = max_total_nm / total_t * 0.9
                    fixed_thicks = [t * scale for t in fixed_thicks]
                new_population.append((child_mats, fixed_thicks))
        
        # Evaluate new population
        new_scores = []
        for mats, thicks in new_population:
            if _is_valid(mats, thicks, problem):
                try:
                    score = _mean_reflectance(mats, thicks, band_nm, 401, n_amb, n_sub, materials_dict)
                except:
                    score = 100.0
            else:
                score = 100.0
            new_scores.append(score)
        
        # Update best
        min_idx = new_scores.index(min(new_scores))
        if new_scores[min_idx] < best_score:
            best_score = new_scores[min_idx]
            best_mats = new_population[min_idx][0]
            best_thicks = new_population[min_idx][1]
            
        population = new_population
        scores = new_scores

    return best_mats, best_thicks, best_score

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layers = problem["max_layers"]
    
    best_score = np.inf
    best_materials = ["L"]
    best_thicknesses = [wl_center / (4.0 * nL)]
    
    # 1. Try a few simple known good designs as seeds
    simple_designs = [
        (["L"], [wl_center / (4.0 * nL)]),
        (["H"], [wl_center / (4.0 * nH)]),
        (["L", "H"], [wl_center / (4.0 * nL), wl_center / (4.0 * nH)]),
        (["H", "L"], [wl_center / (4.0 * nH), wl_center / (4.0 * nL)]),
    ]
    
    for mats, thicks in simple_designs:
        if _is_valid(mats, thicks, problem):
            score = _mean_reflectance(mats, thicks, band_nm, 401, n_amb, n_sub, materials_dict)
            if score < best_score:
                best_score = score
                best_materials = mats
                best_thicknesses = thicks
    
    # 2. Run Genetic Algorithm
    if time.time() < deadline - 10:
        ga_mats, ga_thicks, ga_score = _ga(problem, deadline)
        if ga_score < best_score:
            best_score = ga_score
            best_materials = ga_mats
            best_thicknesses = ga_thicks
            
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
