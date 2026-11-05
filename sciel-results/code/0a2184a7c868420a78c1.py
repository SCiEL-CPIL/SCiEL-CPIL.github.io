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

    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    # Vectorized calculation
    ns = np.array([n_mat[m] for m in materials])
    ds = np.array(thicknesses)
    # beta[i, j] = 2*pi*ns[i]*ds[i] / wls[j]
    beta = 2.0 * math.pi * np.outer(ns * ds, 1.0 / wls)

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
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

def _optimize_nelder_mead(objective, x0, lo, hi, deadline, max_iter=300, tol=1e-7):
    """
    Nelder-Mead with bound handling and deadline check.
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

def _ga_optimize(materials, problem, wls, n_mat, deadline, pop_size=50, generations=50):
    """
    Genetic Algorithm to optimize thicknesses for a fixed material pattern.
    """
    n_layers = len(materials)
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    # Objective function for GA: penalizes constraint violations
    def objective(x):
        # x is an array of thicknesses
        # Clip to valid range
        x_clipped = np.clip(x, 0, max_layer)
        # Check total thickness
        total = np.sum(x_clipped)
        if total > max_total:
            # Scale down
            x_clipped = x_clipped * (max_total / total)
        
        return _mean_reflectance_percent(materials, list(x_clipped), problem, wls, n_mat)

    # Initialize population
    population = []
    for _ in range(pop_size):
        # Random thicknesses
        x = np.random.uniform(0, max_layer, n_layers)
        # Scale to fit total if necessary
        total = np.sum(x)
        if total > max_total:
            x = x * (max_total / total)
        population.append(x)
    
    # Evaluate initial population
    fitness = [objective(ind) for ind in population]
    
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx]
    best_fitness = fitness[best_idx]
    
    for gen in range(generations):
        if time.time() > deadline:
            break
            
        # Sort population by fitness
        indices = np.argsort(fitness)
        population = [population[i] for i in indices]
        fitness = [fitness[i] for i in indices]
        
        new_population = []
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        new_population.extend(population[:elite_count])
        
        # Generate new individuals
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament():
                contenders = random.sample(range(pop_size), 3)
                best_c = min(contenders, key=lambda i: fitness[i])
                return population[best_c].copy()
            
            p1 = tournament()
            p2 = tournament()
            
            # Crossover (Uniform)
            child = np.random.choice([p1, p2], size=n_layers)
            
            # Mutation
            # Gaussian mutation
            mutation_rate = 0.1
            for i in range(n_layers):
                if random.random() < mutation_rate:
                    child[i] += random.gauss(0, max_layer * 0.1)
                    child[i] = np.clip(child[i], 0, max_layer)
            
            # Constraint handling: scale if total > max_total
            total = np.sum(child)
            if total > max_total:
                child = child * (max_total / total)
            
            new_population.append(child)
        
        population = new_population
        fitness = [objective(ind) for ind in population]
        
        # Update best
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_individual = population[gen_best_idx].copy()
            
    return best_individual, best_fitness

def design(problem):
    """
    Design a broadband antireflection coating using a Genetic Algorithm.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

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

    # Try different layer counts and starting materials
    # GA optimizes thicknesses for a fixed material pattern
    # We will try patterns of length 4, 6, 8, 10, 12
    # Starting with L or H
    
    patterns = []
    for n_layers in [4, 6, 8, 10, 12]:
        if n_layers > max_layers:
            continue
        for start_char in ['L', 'H']:
            mats = []
            for i in range(n_layers):
                if (i % 2 == 0) == (start_char == 'L'):
                    mats.append('L')
                else:
                    mats.append('H')
            patterns.append(mats)
    
    # Shuffle patterns
    random.shuffle(patterns)
    
    # Run GA for each pattern
    for mats in patterns:
        if time.time() > deadline - 10:
            break
            
        # Run GA
        x_ga, f_ga = _ga_optimize(mats, problem, wls, n_mat, deadline, pop_size=40, generations=30)
        
        # Post-process: remove zero layers
        active_mats = []
        active_ths = []
        for m, t in zip(mats, x_ga):
            if t > 1e-6:
                active_mats.append(m)
                active_ths.append(t)
        
        if len(active_mats) == 0:
            continue
            
        # Refine with Nelder-Mead on active layers
        if time.time() < deadline - 5:
            lo = np.zeros(len(active_ths))
            hi = np.full(len(active_ths), max_layer)
            
            def objective(th):
                # Ensure total thickness constraint
                th_clipped = np.clip(th, 0, max_layer)
                total = np.sum(th_clipped)
                if total > max_total:
                    th_clipped = th_clipped * (max_total / total)
                return _mean_reflectance_percent(active_mats, list(th_clipped), problem, wls, n_mat)
            
            x0 = np.array(active_ths)
            x_opt, f_opt = _optimize_nelder_mead(objective, x0, lo, hi, deadline, max_iter=100)
            
            # Update best with refined result
            update_best(active_mats, x_opt)
            
            # Also update with GA result in case NM failed
            update_best(active_mats, x_ga)
            
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
