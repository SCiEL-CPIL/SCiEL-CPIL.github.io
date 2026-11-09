"""Broadband antireflection coating design: Differential Evolution over sequence space with local polish."""
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
    
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)
    
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

def _clamp_thicknesses(ths, max_layer, max_total):
    """Clamp thicknesses to valid range and total."""
    ths = [max(0.0, min(max_layer, t)) for t in ths]
    total = sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = [t * scale for t in ths]
    return ths

def _nelder_mead_local(pattern, init_ths, problem, max_iters=150, deadline=None):
    """Fast local polish using Nelder-Mead on fixed pattern."""
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    ths = _clamp_thicknesses(list(init_ths), max_layer, max_total)
    best_score = _mean_reflectance(pattern, ths, problem)
    best_ths = list(ths)
    n = len(ths)
    if n == 0:
        return best_ths, best_score
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    step = 5.0
    simplex = [list(ths)]
    for i in range(n):
        pt = list(ths)
        pt[i] = max(0.0, min(max_layer, pt[i] + step))
        simplex.append(pt)
    scores = [_mean_reflectance(pattern, simplex[i], problem) for i in range(n + 1)]
    
    it = 0
    while it < max_iters and (deadline is None or time.time() < deadline):
        it += 1
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        if scores[0] < best_score:
            best_score = scores[0]
            best_ths = list(simplex[0])
            
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = _clamp_thicknesses(list(xr), max_layer, max_total)
        fr = _mean_reflectance(pattern, xr, problem)
        
        if fr < scores[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            xe = _clamp_thicknesses(list(xe), max_layer, max_total)
            fe = _mean_reflectance(pattern, xe, problem)
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
                xc = _clamp_thicknesses(list(xc), max_layer, max_total)
                fc = _mean_reflectance(pattern, xc, problem)
                if fc <= fr:
                    simplex[-1] = xc
                    scores[-1] = fc
                else:
                    break
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    simplex[i] = _clamp_thicknesses(simplex[i], max_layer, max_total)
                    scores[i] = _mean_reflectance(pattern, simplex[i], problem)
                if np.max(scores) - np.min(scores) < 1e-7:
                    break
    return best_ths, best_score

def design(problem):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    qw = {m: wl_center / (4.0 * n) for m, n in materials_dict.items()}
    
    # DE population size and generations
    pop_size = 40
    max_gens = 60
    random.seed(123)
    
    # Initial population: mix of structured and random patterns
    def random_pattern():
        n = random.randint(3, max_layers)
        return tuple(random.choice(['L', 'H']) for _ in range(n))
    
    def random_thicknesses(n):
        return [random.uniform(20, max_layer) for _ in range(n)]
    
    # Seed with known good structures
    pop = []
    # Quarter-wave alternating
    for n in [4, 6, 8, 10, 12]:
        pat = tuple('L' if i % 2 == 0 else 'H' for i in range(n))
        ths = [qw[m] for m in pat]
        ths = _clamp_thicknesses(ths, max_layer, max_total)
        pop.append((pat, ths))
    # Symmetric patterns
    for n in [6, 8, 10]:
        half = n // 2
        left = tuple(random.choice(['L', 'H']) for _ in range(half))
        if n % 2 == 0:
            pat = left + left[::-1]
        else:
            pat = left + ('L',) + left[::-1]
        ths = [qw[m] for m in pat]
        ths = _clamp_thicknesses(ths, max_layer, max_total)
        pop.append((pat, ths))
    # Random fills
    while len(pop) < pop_size:
        pat = random_pattern()
        ths = random_thicknesses(len(pat))
        ths = _clamp_thicknesses(ths, max_layer, max_total)
        pop.append((pat, ths))
    
    # Evaluate initial population
    scores = [_mean_reflectance(pat, ths, problem) for pat, ths in pop]
    best_idx = np.argmin(scores)
    best_pat = pop[best_idx][0]
    best_ths = pop[best_idx][1]
    best_score = scores[best_idx]
    
    # Differential Evolution
    F = 0.8  # scale factor
    CR = 0.9  # crossover rate
    
    gen = 0
    while gen < max_gens and time.time() < deadline - 20:  # leave 20s for final polish
        gen += 1
        # Sort population
        order = np.argsort(scores)
        pop = [pop[i] for i in order]
        scores = [scores[i] for i in order]
        
        if scores[0] < best_score:
            best_score = scores[0]
            best_pat = pop[0][0]
            best_ths = pop[0][1]
        
        # Early stopping if convergence
        if scores[0] < 0.5:
            break
            
        new_pop = [pop[0]]  # keep best
        new_scores = [scores[0]]
        
        for i in range(1, pop_size):
            # Select three distinct random individuals
            r1, r2, r3 = random.sample(range(1, pop_size), 3)
            pat_i = pop[i][0]
            ths_i = pop[i][1]
            n_i = len(ths_i)
            
            # Mutate thicknesses
            mutant_ths = list(ths_i)
            for k in range(n_i):
                if random.random() < CR:
                    mutant_ths[k] = ths_i[k] + F * (pop[r1][1][k] - pop[r2][1][k])
            mutant_ths = _clamp_thicknesses(mutant_ths, max_layer, max_total)
            
            # Occasionally mutate pattern (structural change)
            mutant_pat = pat_i
            if random.random() < 0.15:
                # Change one layer
                pos = random.randint(0, len(mutant_pat) - 1)
                new_mat = 'H' if mutant_pat[pos] == 'L' else 'L'
                mutant_pat = mutant_pat[:pos] + (new_mat,) + mutant_pat[pos+1:]
            elif random.random() < 0.05:
                # Add or remove a layer
                if len(mutant_pat) < max_layers and random.random() < 0.5:
                    pos = random.randint(0, len(mutant_pat))
                    new_mat = random.choice(['L', 'H'])
                    mutant_pat = mutant_pat[:pos] + (new_mat,) + mutant_pat[pos:]
                    mutant_ths = mutant_ths[:pos] + [random.uniform(30, 200)] + mutant_ths[pos:]
                elif len(mutant_pat) > 3:
                    pos = random.randint(0, len(mutant_pat) - 1)
                    mutant_pat = mutant_pat[:pos] + mutant_pat[pos+1:]
                    mutant_ths = mutant_ths[:pos] + mutant_ths[pos+1:]
            
            mutant_ths = _clamp_thicknesses(mutant_ths, max_layer, max_total)
            
            # Evaluate mutant
            if time.time() > deadline - 25:
                break
                
            mutant_score = _mean_reflectance(mutant_pat, mutant_ths, problem)
            
            # Selection
            if mutant_score <= scores[i]:
                new_pop.append((mutant_pat, mutant_ths))
                new_scores.append(mutant_score)
            else:
                new_pop.append(pop[i])
                new_scores.append(scores[i])
        
        pop = new_pop
        scores = new_scores
        
        # Adaptive F and CR
        if gen % 10 == 0:
            F = max(0.4, F - 0.05)
            CR = max(0.6, CR - 0.05)
    
    # Final local polish on the best candidate
    if time.time() < deadline - 5:
        polished_ths, polished_score = _nelder_mead_local(best_pat, best_ths, problem, 
                                                          max_iters=200, deadline=deadline - 5)
        if polished_score < best_score:
            best_ths = polished_ths
            best_score = polished_score
    
    # Try a few more local polishes on near-best candidates
    if time.time() < deadline - 10:
        order = np.argsort(scores)
        for idx in order[1:4]:
            if time.time() > deadline - 10:
                break
            pat_try = pop[idx][0]
            ths_try = pop[idx][1]
            polished_ths, polished_score = _nelder_mead_local(pat_try, ths_try, problem,
                                                              max_iters=100, deadline=deadline - 10)
            if polished_score < best_score:
                best_ths = polished_ths
                best_score = polished_score
                best_pat = pat_try
    
    if not best_ths:
        return ["L"], [qw["L"]]
        
    return list(best_pat), list(best_ths)
