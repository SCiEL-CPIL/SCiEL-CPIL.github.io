"""
Broadband antireflection coating design: Genetic Algorithm co-optimizing sequence and thicknesses.
"""
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

    # delta_j = 2*pi*n_j*d_j/lambda -> (N, K)
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
    deadline = start_time + 145  # Safety margin for 150s limit
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    
    # GA Parameters
    pop_size = 40
    mutation_rate_thick = 0.1
    mutation_rate_seq = 0.05
    crossover_rate = 0.8
    
    # Helper to evaluate a genome: (materials_list, thicknesses_list)
    def evaluate_genome(materials, thicknesses):
        # Constrain thicknesses
        ths = list(thicknesses)
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        for i in range(len(ths)):
            ths[i] = max(0.0, min(max_layer, ths[i]))
        # Re-check total
        total2 = sum(ths)
        if total2 > max_total:
            scale = max_total / total2
            ths = [t * scale for t in ths]
        
        score = _mean_reflectance(materials, ths, problem)
        return score, ths

    # Initialize Population
    population = []
    
    # 1. Quarter-wave stack (Alternating)
    init_mats = []
    init_ths = []
    for i in range(max_layers):
        if i % 2 == 0:
            init_mats.append('L')
            init_ths.append(wl_center / (4.0 * n_L))
        else:
            init_mats.append('H')
            init_ths.append(wl_center / (4.0 * n_H))
    population.append((init_mats, init_ths))
    
    # 2. Quarter-wave stack (Reversed)
    init_mats_rev = init_mats[::-1]
    init_ths_rev = init_ths[::-1]
    population.append((init_mats_rev, init_ths_rev))
    
    # 3. Random Graded Stack (approximating index gradient)
    # Air(1.0) -> Glass(1.52). L(1.38) is closer to Air, H(2.35) is far.
    # A gradient might be L...L...H...H...? Or mixed?
    # Let's just add random diverse individuals.
    for _ in range(pop_size - 2):
        mats = ['L' if random.random() < 0.5 else 'H' for _ in range(max_layers)]
        ths = [random.uniform(20, 100) for _ in range(max_layers)]
        population.append((mats, ths))
        
    # Evaluate initial population
    scored_population = []
    for mats, ths in population:
        score, constrained_ths = evaluate_genome(mats, ths)
        scored_population.append((score, mats, constrained_ths))
        
    scored_population.sort(key=lambda x: x[0])
    best_score, best_mats, best_ths = scored_population[0]
    
    # GA Loop
    generation = 0
    while time.time() < deadline - 20: # Leave 20s for final polish
        generation += 1
        new_population = []
        
        # Elitism: Keep top 10%
        elite_count = max(2, pop_size // 10)
        for i in range(elite_count):
            new_population.append((list(scored_population[i][1]), list(scored_population[i][2])))
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Selection: Tournament
            def select():
                a = random.randint(0, len(scored_population)-1)
                b = random.randint(0, len(scored_population)-1)
                if scored_population[a][0] < scored_population[b][0]:
                    return scored_population[a]
                else:
                    return scored_population[b]
            
            parent1 = select()
            parent2 = select()
            
            # Crossover
            if random.random() < crossover_rate:
                # Uniform crossover for sequence
                new_mats = []
                new_ths = []
                for i in range(max_layers):
                    if random.random() < 0.5:
                        new_mats.append(parent1[1][i])
                        new_ths.append(parent1[2][i])
                    else:
                        new_mats.append(parent2[1][i])
                        new_ths.append(parent2[2][i])
            else:
                new_mats = list(parent1[1])
                new_ths = list(parent1[2])
                
            # Mutation
            # Sequence Mutation
            for i in range(max_layers):
                if random.random() < mutation_rate_seq:
                    if new_mats[i] == 'L':
                        new_mats[i] = 'H'
                    else:
                        new_mats[i] = 'L'
                        
            # Thickness Mutation
            for i in range(max_layers):
                if random.random() < mutation_rate_thick:
                    # Gaussian noise scaled by current thickness or fixed?
                    # Fixed scale is safer for exploration
                    noise = random.gauss(0, 20.0)
                    new_ths[i] += noise
                    if new_ths[i] < 0:
                        new_ths[i] = 0
                    if new_ths[i] > max_layer:
                        new_ths[i] = max_layer
            
            new_population.append((new_mats, new_ths))
            
        # Evaluate new population
        scored_new_population = []
        for mats, ths in new_population:
            score, constrained_ths = evaluate_genome(mats, ths)
            scored_new_population.append((score, mats, constrained_ths))
            
        # Combine and sort
        all_scored = scored_population + scored_new_population
        all_scored.sort(key=lambda x: x[0])
        scored_population = all_scored[:pop_size]
        
        # Update best
        if scored_population[0][0] < best_score:
            best_score = scored_population[0][0]
            best_mats = list(scored_population[0][1])
            best_ths = list(scored_population[0][2])
            
        # Check for convergence or time
        if scored_population[0][0] == scored_population[-1][0]:
            break

    # Final Polish: Nelder-Mead on the best GA result
    def nelder_mead(materials_pattern, init_ths, problem, deadline, max_iters=500):
        ths = list(init_ths)
        ths = [max(0.0, min(max_layer, t)) for t in ths]
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        
        best_score = _mean_reflectance(materials_pattern, ths, problem)
        best_ths = list(ths)
        n = len(ths)
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial step size
        step = 10.0
        simplex = [list(ths)]
        for i in range(n):
            pt = list(ths)
            pt[i] = max(0.0, min(max_layer, pt[i] + step))
            simplex.append(pt)
            
        scores = [_mean_reflectance(materials_pattern, simplex[i], problem) for i in range(n + 1)]
        
        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            if scores[0] < best_score:
                best_score = scores[0]
                best_ths = list(simplex[0])
                
            # Check convergence
            if max(scores) - min(scores) < 1e-6:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = [max(0.0, min(max_layer, x)) for x in xr]
            # Constrain total
            total = sum(xr)
            if total > max_total:
                scale = max_total / total
                xr = [x * scale for x in xr]
            fr = _mean_reflectance(materials_pattern, xr, problem)
            
            if fr < scores[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = [max(0.0, min(max_layer, x)) for x in xe]
                total = sum(xe)
                if total > max_total:
                    scale = max_total / total
                    xe = [x * scale for x in xe]
                fe = _mean_reflectance(materials_pattern, xe, problem)
                if fe < fr:
                    simplex[-1] = xe
                    scores[-1] = fe
                else:
                    simplex[-1] = xr
                    scores[-1] = fr
            elif fr < scores[-2]:
                simplex[-1] = xr
                scores[-1] = fr
            else:
                if fr < scores[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = [max(0.0, min(max_layer, x)) for x in xc]
                    total = sum(xc)
                    if total > max_total:
                        scale = max_total / total
                        xc = [x * scale for x in xc]
                    fc = _mean_reflectance(materials_pattern, xc, problem)
                    if fc <= fr:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        break
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = [max(0.0, min(max_layer, x)) for x in simplex[i]]
                        total = sum(simplex[i])
                        if total > max_total:
                            scale = max_total / total
                            simplex[i] = [x * scale for x in simplex[i]]
                        scores[i] = _mean_reflectance(materials_pattern, simplex[i], problem)
                        
        return best_ths, best_score

    final_ths, final_score = nelder_mead(best_mats, best_ths, problem, deadline, max_iters=1000)
    
    if final_score < best_score:
        best_ths = final_ths
        best_score = final_score
        
    return list(best_mats), list(best_ths)
