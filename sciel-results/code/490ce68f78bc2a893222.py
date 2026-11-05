import time
import math
import numpy as np
import random

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)

    ns = np.array([n_mat[m] for m in materials])
    ds = np.array(thicknesses)
    beta = 2.0 * math.pi * np.outer(ns * ds, 1.0 / wls)

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i in range(n_layers):
        n = ns[i]
        c = cosb[i]
        s = sinb[i]

        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D

        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent with constraint penalties.
    """
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6

    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]

    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6

    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6 + (total_th - max_total) * 10.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _decode_individual(ind, max_layer):
    """
    Decode a GA individual into materials and thicknesses.
    ind is a flat array: [mat0, th0, mat1, th1, ..., matN-1, thN-1]
    mat is 0.0 or 1.0 (thresholded to 'L' or 'H')
    th is the thickness
    """
    n_layers = len(ind) // 2
    materials = []
    thicknesses = []
    for i in range(n_layers):
        mat_val = ind[2*i]
        th_val = ind[2*i + 1]
        
        # Threshold material
        if mat_val < 0.5:
            materials.append('L')
        else:
            materials.append('H')
        
        # Clip thickness
        thicknesses.append(np.clip(th_val, 0.0, max_layer))
        
    return materials, thicknesses

def _ga_optimize(objective, initial_individual, max_layer, deadline, pop_size=40, max_generations=200, mutation_rate=0.1, crossover_rate=0.8):
    """
    Genetic Algorithm to optimize the individual.
    objective takes a flat numpy array and returns a score (lower is better).
    """
    n_genes = len(initial_individual)
    
    # Initialize population
    population = []
    for i in range(pop_size):
        if i == 0:
            population.append(initial_individual.copy())
        else:
            # Random initialization around the best known or random
            # Mix of random and perturbed best
            if random.random() < 0.5:
                ind = np.random.uniform(0, 1, n_genes) # Random material
                ind[1::2] = np.random.uniform(0, max_layer, n_genes//2) # Random thickness
            else:
                ind = initial_individual + np.random.normal(0, 0.1, n_genes)
                ind[0::2] = np.clip(ind[0::2], 0, 1)
                ind[1::2] = np.clip(ind[1::2], 0, max_layer)
            population.append(ind)
            
    fitnesses = [objective(ind) for ind in population]
    best_ind = population[0]
    best_fit = fitnesses[0]
    
    for i in range(1, pop_size):
        if fitnesses[i] < best_fit:
            best_fit = fitnesses[i]
            best_ind = population[i]
            
    for gen in range(max_generations):
        if time.time() > deadline:
            break
            
        # Sort by fitness
        sorted_idx = np.argsort(fitnesses)
        population = [population[i] for i in sorted_idx]
        fitnesses = [fitnesses[i] for i in sorted_idx]
        
        # Elitism: keep top 2
        new_population = [population[0].copy(), population[1].copy()]
        
        # Generate rest of population
        while len(new_population) < pop_size:
            # Selection: Tournament
            def tournament():
                idxs = random.sample(range(len(population)), 3)
                best_idx = min(idxs, key=lambda i: fitnesses[i])
                return population[best_idx]
                
            if random.random() < crossover_rate:
                p1 = tournament()
                p2 = tournament()
                # Uniform crossover
                child = np.random.choice([p1, p2], size=n_genes)
            else:
                p1 = tournament()
                child = p1.copy()
                
            # Mutation
            for j in range(n_genes):
                if random.random() < mutation_rate:
                    if j % 2 == 0: # Material gene
                        child[j] = 1.0 - child[j] # Flip
                    else: # Thickness gene
                        # Gaussian mutation
                        child[j] += np.random.normal(0, 0.1 * max_layer)
                        child[j] = np.clip(child[j], 0, max_layer)
                        
            new_population.append(child)
            
        population = new_population
        fitnesses = [objective(ind) for ind in population]
        
        # Update best
        min_idx = np.argmin(fitnesses)
        if fitnesses[min_idx] < best_fit:
            best_fit = fitnesses[min_idx]
            best_ind = population[min_idx]
            
    return best_ind, best_fit

def _nelder_mead_fast(objective, x0, lo, hi, deadline, max_iter=100, tol=1e-8):
    """
    Fast Nelder-Mead with bound handling and deadline check.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    
    if n == 0:
        return x0, objective(x0)

    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.05 * (hi[i] - lo[i]) if hi[i] - lo[i] > 1e-6 else 1.0
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))

    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        if abs(f_vals[0] - f_vals[-1]) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)

        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue

        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue

        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue

        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])

    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]

    best_val = float('inf')
    best_mats = None
    best_ths = None

    def update_best(mats, ths):
        nonlocal best_val, best_mats, best_ths
        ths_clipped = [min(max(t, 0.0), max_layer) for t in ths]
        s = sum(ths_clipped)
        if s > max_total:
            if s > 1e-9:
                f = max_total / s
                ths_clipped = [t * f for t in ths_clipped]
        
        # Remove zero layers to reduce layer count
        final_mats = []
        final_ths = []
        for m, t in zip(mats, ths_clipped):
            if t > 1e-6:
                final_mats.append(m)
                final_ths.append(t)
        
        if len(final_mats) > max_layers:
            return
        
        val = _mean_reflectance_percent(final_mats, final_ths, problem, wls, n_mat)
        if val < best_val:
            best_val = val
            best_mats = list(final_mats)
            best_ths = list(final_ths)
        return val

    # Create objective for GA that takes a flat array
    def ga_objective(ind):
        mats, ths = _decode_individual(ind, max_layer)
        if len(mats) == 0:
            return 1e6
        # Check total thickness constraint with penalty
        total_th = sum(ths)
        if total_th > max_total + 1e-6:
            penalty = (total_th - max_total) * 10.0
            return 1e6 + penalty
        
        return _mean_reflectance_percent(mats, ths, problem, wls, n_mat)

    # Generate initial candidates for GA
    # We will run GA for a few different layer counts
    best_ind = None
    best_fit = float('inf')
    
    # Target wavelengths for quarter-wave initialization
    wl_targets = [500.0, 450.0, 550.0, 400.0, 600.0]
    
    for n_layers in [4, 6, 8, 10, 12]:
        if n_layers > max_layers:
            continue
            
        # Create a flat individual: [mat, th, mat, th, ...]
        # Start with alternating L/H and quarter-wave thicknesses
        ind = np.zeros(2 * n_layers)
        for i in range(n_layers):
            # Alternate materials
            if i % 2 == 0:
                ind[2*i] = 0.0 # L
            else:
                ind[2*i] = 1.0 # H
                
            # Set thickness to quarter wave for 500nm
            mat = 'L' if i % 2 == 0 else 'H'
            n = n_mat[mat]
            ind[2*i + 1] = 500.0 / (4.0 * n)
            
        # Run GA
        if time.time() > deadline - 20:
            break
            
        ind_result, fit_result = _ga_optimize(
            ga_objective, 
            ind, 
            max_layer, 
            deadline - 5, 
            pop_size=30, 
            max_generations=150
        )
        
        if fit_result < best_fit:
            best_fit = fit_result
            best_ind = ind_result
            
        # Decode and update best
        mats, ths = _decode_individual(best_ind, max_layer)
        update_best(mats, ths)

    # If we found a good individual, refine it with Nelder-Mead on the thicknesses only
    # (fixing the material pattern found by GA)
    if best_ind is not None and time.time() < deadline - 10:
        mats, ths = _decode_individual(best_ind, max_layer)
        
        # Remove zero layers for refinement
        ref_mats = []
        ref_ths = []
        for m, t in zip(mats, ths):
            if t > 1e-6:
                ref_mats.append(m)
                ref_ths.append(t)
                
        if len(ref_mats) > 0:
            # Define objective for Nelder-Mead (thicknesses only)
            def nm_objective(th_arr):
                ths_list = [min(max(t, 0.0), max_layer) for t in th_arr]
                # Check total thickness
                if sum(ths_list) > max_total + 1e-6:
                    return 1e6 + (sum(ths_list) - max_total) * 10.0
                return _mean_reflectance_percent(ref_mats, ths_list, problem, wls, n_mat)
                
            x0 = np.array(ref_ths)
            lo = np.zeros(len(x0))
            hi = np.full(len(x0), max_layer)
            
            x_opt, f_opt = _nelder_mead_fast(nm_objective, x0, lo, hi, deadline - 2, max_iter=100)
            update_best(ref_mats, x_opt)

    # Fallback
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        update_best(best_mats, best_ths)
    
    # Final constraint check
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    s = sum(best_ths)
    if s > max_total + 1e-6:
        f = max_total / s
        best_ths = [t * f for t in best_ths]
        best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    
    return list(best_mats), list(best_ths)
