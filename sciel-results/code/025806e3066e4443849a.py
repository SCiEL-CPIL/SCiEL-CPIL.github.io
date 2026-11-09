"""Broadband antireflection coating design: Continuous index profile with differential evolution."""
import time
import random
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)
    
    # delta_j = 2*pi*n_j*d_j/lambda  -> (N, K)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)
    
    # Initialize characteristic matrix product as identity for all wavelengths
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)
    
    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new
    
    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def design(problem):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    
    # Precompute quarter-wave thicknesses at center wavelength
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)
    
    N = 20  # Number of layers
    
    def profile_to_layers(node_values):
        """Convert continuous node values to discrete L/H pattern and thicknesses.
        
        node_values: array of shape (N,) with values in [n_L, n_H]
        Each node represents the effective index of that layer.
        We map each node to either L or H based on proximity, and set thickness
        to maintain a target optical path length per layer.
        """
        # Target optical path length: total OD / N
        # We want total physical thickness <= max_total
        # Let's aim for total OD = max_total * n_avg where n_avg is some average
        # Actually, let's just set each layer's OD to be max_total/N, then check constraints
        
        target_od = max_total / N  # Optical path length per layer
        
        materials = []
        thicknesses = []
        
        for i in range(N):
            n_eff = node_values[i]
            # Map to closest material
            if abs(n_eff - n_L) < abs(n_eff - n_H):
                mat = 'L'
                n_mat = n_L
            else:
                mat = 'H'
                n_mat = n_H
            materials.append(mat)
            # Thickness to achieve target OD with this material
            d = target_od / n_mat
            # Clamp to max_layer
            d = min(d, max_layer)
            thicknesses.append(d)
        
        # Check total thickness constraint
        total = sum(thicknesses)
        if total > max_total:
            scale = max_total / total
            thicknesses = [t * scale for t in thicknesses]
        
        # Ensure each thickness is within [0, max_layer]
        thicknesses = [max(0.0, min(max_layer, t)) for t in thicknesses]
        
        # Re-check total after clamping
        total2 = sum(thicknesses)
        if total2 > max_total:
            scale = max_total / total2
            thicknesses = [t * scale for t in thicknesses]
        
        return materials, thicknesses
    
    def evaluate_nodes(node_values):
        """Evaluate a set of node values by converting to layers and computing reflectance."""
        materials, thicknesses = profile_to_layers(node_values)
        return _mean_reflectance(materials, thicknesses, problem)
    
    # Differential Evolution parameters
    pop_size = 30
    F = 0.8  # Mutation scale
    CR = 0.9  # Crossover rate
    
    # Initialize population
    pop = []
    for _ in range(pop_size):
        # Start with a graded profile from air to substrate
        # Linear interpolation in log space for better physical intuition
        # Actually, let's use a simple linear ramp in index space
        node_vals = np.linspace(n_L, n_H, N)
        # Add some randomness
        node_vals += np.random.uniform(-0.3, 0.3, N)
        # Clamp to valid range
        node_vals = np.clip(node_vals, n_L, n_H)
        pop.append(node_vals)
    
    # Also add some structured initial guesses
    # Quarter-wave alternating pattern
    qw_pattern_nodes = np.array([n_L if i % 2 == 0 else n_H for i in range(N)])
    pop.append(qw_pattern_nodes)
    
    # Graded from L to H
    graded = np.linspace(n_L, n_H, N)
    pop.append(graded)
    
    # Graded from H to L
    graded_rev = np.linspace(n_H, n_L, N)
    pop.append(graded_rev)
    
    # Evaluate initial population
    scores = [evaluate_nodes(ind) for ind in pop]
    
    best_score = min(scores)
    best_index = scores.index(best_score)
    best_nodes = pop[best_index].copy()
    
    # Track best result
    best_materials, best_thicknesses = profile_to_layers(best_nodes)
    
    # Differential Evolution loop
    gen = 0
    max_gens = 500
    stagnation_count = 0
    stagnation_threshold = 20
    best_score_history = []
    
    while gen < max_gens and time.time() < deadline - 5:
        gen += 1
        
        for i in range(len(pop)):
            if time.time() > deadline - 3:
                break
            
            # Select three distinct random individuals
            a, b, c = random.sample(range(len(pop)), 3)
            
            # Mutation
            mutant = pop[a] + F * (pop[b] - pop[c])
            # Clamp to valid range
            mutant = np.clip(mutant, n_L, n_H)
            
            # Crossover
            trial = pop[i].copy()
            for j in range(N):
                if random.random() < CR or j == random.randint(0, N - 1):
                    trial[j] = mutant[j]
            
            # Evaluate trial
            trial_score = evaluate_nodes(trial)
            
            # Selection
            if trial_score < scores[i]:
                pop[i] = trial
                scores[i] = trial_score
        
        # Update best
        current_best = min(scores)
        current_best_idx = scores.index(current_best)
        
        if current_best < best_score - 1e-6:
            best_score = current_best
            best_index = current_best_idx
            best_nodes = pop[best_index].copy()
            best_materials, best_thicknesses = profile_to_layers(best_nodes)
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        best_score_history.append(best_score)
        
        # Adaptive mutation factor
        if stagnation_count > stagnation_threshold:
            F = max(0.3, F * 0.9)
            CR = min(0.99, CR + 0.01)
            stagnation_count = 0
        
        # Check for convergence
        if len(best_score_history) > 10:
            if best_score_history[-1] - best_score_history[-10] < 1e-5:
                break
    
    # Final local optimization using Nelder-Mead on the best solution
    def nelder_mead_local(init_nodes, max_iters=200):
        """Local optimization in node space."""
        nodes = list(init_nodes)
        best_score_lm = evaluate_nodes(nodes)
        best_nodes_lm = list(nodes)
        
        n = len(nodes)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial step size
        step = 0.1
        simplex = [list(nodes)]
        for i in range(n):
            pt = list(nodes)
            pt[i] = max(n_L, min(n_H, pt[i] + step))
            simplex.append(pt)
        
        lm_scores = [evaluate_nodes(simplex[i]) for i in range(n + 1)]
        
        it = 0
        while it < max_iters and time.time() < deadline - 2:
            it += 1
            order = np.argsort(lm_scores)
            simplex = [simplex[i] for i in order]
            lm_scores = [lm_scores[i] for i in order]
            
            if lm_scores[0] < best_score_lm:
                best_score_lm = lm_scores[0]
                best_nodes_lm = list(simplex[0])
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, n_L, n_H)
            fr = evaluate_nodes(xr)
            
            if fr < lm_scores[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, n_L, n_H)
                fe = evaluate_nodes(xe)
                if fe < fr:
                    simplex[-1] = list(xe)
                    lm_scores[-1] = fe
                else:
                    simplex[-1] = list(xr)
                    lm_scores[-1] = fr
            elif fr < lm_scores[-2]:
                simplex[-1] = list(xr)
                lm_scores[-1] = fr
            else:
                if fr < lm_scores[-1]:
                    xc = centroid + rho * (np.array(simplex[-1]) - centroid)
                    xc = np.clip(xc, n_L, n_H)
                    fc = evaluate_nodes(xc)
                    if fc <= fr:
                        simplex[-1] = list(xc)
                        lm_scores[-1] = fc
                    else:
                        break
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = np.clip(simplex[i], n_L, n_H)
                        lm_scores[i] = evaluate_nodes(simplex[i])
                    if max(lm_scores) - min(lm_scores) < 1e-7:
                        break
                        
        return best_nodes_lm, best_score_lm
    
    # Polish the best solution
    if time.time() < deadline - 5:
        polished_nodes, polished_score = nelder_mead_local(best_nodes, max_iters=200)
        if polished_score < best_score:
            best_score = polished_score
            best_nodes = polished_nodes
            best_materials, best_thicknesses = profile_to_layers(best_nodes)
    
    # Final validation and return
    final_materials, final_thicknesses = profile_to_layers(best_nodes)
    final_score = _mean_reflectance(final_materials, final_thicknesses, problem)
    
    # Ensure constraints are met
    if len(final_materials) > max_layers:
        final_materials = final_materials[:max_layers]
        final_thicknesses = final_thicknesses[:max_layers]
    
    total = sum(final_thicknesses)
    if total > max_total:
        scale = max_total / total
        final_thicknesses = [t * scale for t in final_thicknesses]
    
    for i in range(len(final_thicknesses)):
        final_thicknesses[i] = max(0.0, min(max_layer, final_thicknesses[i]))
    
    # Re-check total after clamping
    total2 = sum(final_thicknesses)
    if total2 > max_total:
        scale = max_total / total2
        final_thicknesses = [t * scale for t in final_thicknesses]
    
    return final_materials, final_thicknesses
