"""Broadband antireflection coating design: Global Differential Evolution over joint pattern and thickness space."""
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
    
    if len(ns) == 0:
        return 0.0

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
    
    # Precompute quarter-wave thicknesses at center wavelength
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)
        
    # We will optimize over a fixed maximum number of layers (max_layers).
    # Layers with thickness 0 are effectively removed, but the pattern index remains.
    # This simplifies the representation to a vector of length max_layers.
    # x[0:max_layers] are thicknesses.
    # x[max_layers:2*max_layers] are binary indicators for material (0 for L, 1 for H).
    
    N = max_layers
    dim = 2 * N
    
    # Helper to decode a candidate vector
    def decode(x):
        ths = x[:N]
        mats = x[N:]
        
        # Clamp thicknesses
        ths = np.clip(ths, 0.0, max_layer)
        
        # Round materials to 0 or 1
        mats = (mats > 0.5).astype(int)
        
        # Build list of active layers
        active_ths = []
        active_mats = []
        for i in range(N):
            if ths[i] > 1e-6: # Ignore negligible layers
                active_ths.append(ths[i])
                active_mats.append('H' if mats[i] == 1 else 'L')
                
        # Check total thickness constraint
        total = sum(active_ths)
        if total > max_total:
            scale = max_total / total
            active_ths = [t * scale for t in active_ths]
            
        return active_mats, active_ths

    def evaluate(x):
        mats, ths = decode(x)
        if not mats:
            return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                            (problem["substrate_index"] + problem["ambient_index"])) ** 2
        return _mean_reflectance(mats, ths, problem)

    # Differential Evolution Setup
    pop_size = 60
    F = 0.8  # Mutation factor
    CR = 0.9 # Crossover probability
    
    # Initialize population
    population = []
    
    # Start 1: Quarter-wave stack of L H L H...
    qw_stack_ths = []
    qw_stack_mats = []
    for i in range(N):
        if i % 2 == 0:
            qw_stack_mats.append('L')
            qw_stack_ths.append(qw['L'])
        else:
            qw_stack_mats.append('H')
            qw_stack_ths.append(qw['H'])
            
    # Encode to vector
    x_qw = np.zeros(dim)
    for i in range(N):
        x_qw[i] = qw_stack_ths[i]
        x_qw[N+i] = 1.0 if qw_stack_mats[i] == 'H' else 0.0
    population.append(x_qw)
    
    # Start 2: Random valid stack
    def random_individual():
        x = np.zeros(dim)
        # Random thicknesses
        x[:N] = np.random.uniform(0, max_layer, N)
        # Random materials
        x[N:] = np.random.choice([0.0, 1.0], N)
        return x

    for _ in range(pop_size - 1):
        population.append(random_individual())
        
    population = np.array(population)
    scores = np.array([evaluate(ind) for ind in population])
    
    best_idx = np.argmin(scores)
    best_x = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # DE Loop
    gen = 0
    max_gens = 2000
    stall_count = 0
    stall_limit = 50
    
    while gen < max_gens:
        if time.time() > deadline:
            break
            
        gen += 1
        current_best = np.min(scores)
        
        if current_best < best_score:
            best_score = current_best
            best_idx = np.argmin(scores)
            best_x = population[best_idx].copy()
            stall_count = 0
        else:
            stall_count += 1
            if stall_count > stall_limit:
                # Inject diversity
                for _ in range(5):
                    idx = np.random.randint(0, pop_size)
                    population[idx] = random_individual()
                    scores[idx] = evaluate(population[idx])
                stall_count = 0

        # Generate offspring
        for i in range(pop_size):
            # Select 3 distinct random individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            v = population[r1] + F * (population[r2] - population[r3])
            
            # Crossover
            j_rand = np.random.randint(0, dim)
            trial = np.zeros(dim)
            for j in range(dim):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = population[i][j]
                    
            # Clamp to bounds
            trial[:N] = np.clip(trial[:N], 0.0, max_layer)
            trial[N:] = np.clip(trial[N:], 0.0, 1.0)
            
            # Evaluate
            trial_score = evaluate(trial)
            
            # Selection
            if trial_score < scores[i]:
                population[i] = trial
                scores[i] = trial_score

    # Final polish with Nelder-Mead on the best found solution
    # We need to handle the discrete nature of materials. 
    # For the final polish, we fix the material pattern of the best DE solution 
    # and optimize only the thicknesses using Nelder-Mead, as continuous optimization 
    # on binary variables is ill-defined without relaxation.
    
    best_mats, best_ths = decode(best_x)
    
    if best_mats:
        def nm_constrain(ths):
            ths = list(ths)
            total = sum(ths)
            if total > max_total:
                scale = max_total / total
                ths = [t * scale for t in ths]
            for i in range(len(ths)):
                ths[i] = max(0.0, min(max_layer, ths[i]))
            total2 = sum(ths)
            if total2 > max_total:
                scale = max_total / total2
                ths = [t * scale for t in ths]
            return ths

        def nm_eval(ths):
            return _mean_reflectance(best_mats, ths, problem)

        # Nelder-Mead on thicknesses only
        n = len(best_ths)
        if n > 0:
            init_ths = nm_constrain(best_ths)
            simplex = [init_ths]
            step = 5.0
            for i in range(n):
                pt = list(init_ths)
                pt[i] += step
                pt = nm_constrain(pt)
                simplex.append(pt)
            
            scores_nm = [nm_eval(s) for s in simplex]
            
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            max_iters = 100
            
            it = 0
            while it < max_iters and time.time() < deadline:
                it += 1
                order = np.argsort(scores_nm)
                simplex = [simplex[i] for i in order]
                scores_nm = [scores_nm[i] for i in order]
                
                if scores_nm[0] < best_score:
                    best_score = scores_nm[0]
                    best_ths = list(simplex[0])
                    
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = nm_constrain(list(xr))
                fr = nm_eval(xr)
                
                if fr < scores_nm[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = nm_constrain(list(xe))
                    fe = nm_eval(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        scores_nm[-1] = fe
                    else:
                        simplex[-1] = xr
                        scores_nm[-1] = fr
                elif fr < scores_nm[-2]:
                    simplex[-1] = xr
                    scores_nm[-1] = fr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = nm_constrain(list(xc))
                    fc = nm_eval(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        scores_nm[-1] = fc
                    else:
                        break

    # If DE found an empty stack, we should probably return a valid single layer
    if not best_mats:
         best_mats = ['L']
         best_ths = [qw['L']]
    
    # Final validation and return
    # Ensure we don't exceed max layers in the return (though decode handles it)
    # The problem asks for a list of labels and thicknesses.
    
    return list(best_mats), list(best_ths)
