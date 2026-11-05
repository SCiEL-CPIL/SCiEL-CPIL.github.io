"""Broadband antireflection coating design via structured GA + Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 145.0
_MARGIN = 5.0

def _compute_reflectance(materials, thicknesses, wavelengths, n0, ns, nL, nH):
    """
    Compute reflectance at all wavelengths simultaneously.
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full(len(wavelengths), abs(r)**2)
    
    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)
    
    for i, (mat, d) in enumerate(zip(materials, thicknesses)):
        n = nL if mat == 'L' else nH
        beta = 2.0 * np.pi * n * d / wavelengths
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = 1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    B = M11 + M12 * ns
    C = M21 + M22 * ns
    Y = C / B
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r)**2
    return np.real(R)

def _mean_reflectance_pct(materials, thicknesses, problem, n_wl=401):
    """Mean reflectance in percent over the band."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    
    wls = np.linspace(band[0], band[1], n_wl)
    R_array = _compute_reflectance(materials, thicknesses, wls, n0, ns, nL, nH)
    return float(np.mean(R_array) * 100.0)

def _is_valid(materials, thicknesses, max_layers, max_total, max_layer):
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return False
    if sum(thicknesses) > max_total + 1e-6:
        return False
    return True

def _evaluate(materials, thicknesses, problem, n_wl=401):
    """Return mean reflectance in percent, or inf if invalid."""
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    if not _is_valid(materials, thicknesses, max_layers, max_total, max_layer):
        return float('inf')
    return _mean_reflectance_pct(materials, thicknesses, problem, n_wl)

def _nelder_mead(materials, thicknesses, problem, deadline, max_iter=500):
    """
    Nelder-Mead simplex optimization on thicknesses.
    Materials are fixed; only thicknesses are optimized.
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        return materials, thicknesses
    
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    def objective(thick_arr):
        if len(thick_arr) != len(materials):
            return float('inf')
        
        thks = np.clip(thick_arr, 0, max_layer_nm)
        
        total = np.sum(thks)
        if total > max_total_nm + 1e-6:
            scale = max_total_nm / total
            thks = thks * scale
            thks = np.clip(thks, 0, max_layer_nm)
        
        return _mean_reflectance_pct(materials, list(thks), problem)

    simplex = []
    scores = []
    
    init = np.array(thicknesses, dtype=float)
    init = np.clip(init, 0, max_layer_nm)
    if np.sum(init) > max_total_nm:
        init = init * (max_total_nm / np.sum(init))
        
    simplex.append(init)
    scores.append(objective(init))
    
    for i in range(n_layers):
        p = init.copy()
        step = 30.0
        p[i] += step
        p = np.clip(p, 0, max_layer_nm)
        if np.sum(p) > max_total_nm:
             p = p * (max_total_nm / np.sum(p))
             
        simplex.append(p)
        scores.append(objective(p))
    
    alpha = 0.5
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(max_iter):
        if time.time() > deadline - _MARGIN:
            break
        
        sorted_indices = np.argsort(scores)
        simplex = [simplex[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        if abs(scores[0] - scores[-1]) < 1e-12:
            break
        
        worst = simplex[-1]
        worst_score = scores[-1]
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        reflected = centroid + alpha * (centroid - worst)
        reflected = np.clip(reflected, 0, max_layer_nm)
        if np.sum(reflected) > max_total_nm:
             reflected = reflected * (max_total_nm / np.sum(reflected))
             
        ref_score = objective(reflected)
        
        if ref_score < scores[0]:
            expanded = centroid + gamma * (reflected - centroid)
            expanded = np.clip(expanded, 0, max_layer_nm)
            if np.sum(expanded) > max_total_nm:
                 expanded = expanded * (max_total_nm / np.sum(expanded))
                 
            exp_score = objective(expanded)
            if exp_score < ref_score:
                simplex[-1] = expanded
                scores[-1] = exp_score
            else:
                simplex[-1] = reflected
                scores[-1] = ref_score
        elif ref_score < worst_score:
            simplex[-1] = reflected
            scores[-1] = ref_score
        else:
            contracted = centroid + rho * (worst - centroid)
            contracted = np.clip(contracted, 0, max_layer_nm)
            if np.sum(contracted) > max_total_nm:
                 contracted = contracted * (max_total_nm / np.sum(contracted))
                 
            con_score = objective(contracted)
            if con_score < worst_score:
                simplex[-1] = contracted
                scores[-1] = con_score
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    scores[i] = objective(simplex[i])
        
        if time.time() > deadline - _MARGIN:
            break
            
    best_idx = np.argmin(scores)
    best_thks = simplex[best_idx]
    best_thks = [max(0.0, min(max_layer_nm, t)) for t in best_thks]
    if sum(best_thks) > max_total_nm:
        scale = max_total_nm / sum(best_thks)
        best_thks = [t * scale for t in best_thks]
        
    return materials, best_thks

def _coordinate_descent(materials, thicknesses, problem, deadline, max_iter=1000):
    """
    Coordinate descent with adaptive steps for local refinement.
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        return materials, thicknesses
    
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    current_thick = list(thicknesses)
    current_score = _mean_reflectance_pct(materials, current_thick, problem)
    
    step_sizes = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05]
    
    for step in step_sizes:
        if time.time() > deadline - _MARGIN:
            break
        
        improved = True
        while improved and time.time() < deadline - _MARGIN:
            improved = False
            
            for i in range(n_layers):
                if time.time() > deadline - _MARGIN:
                    break
                
                orig = current_thick[i]
                for sign in [1, -1]:
                    new_thick = current_thick[:]
                    new_thick[i] = orig + sign * step
                    
                    if new_thick[i] < 0 or new_thick[i] > max_layer_nm:
                        continue
                    if sum(new_thick) > max_total_nm:
                        continue
                    
                    trial_score = _mean_reflectance_pct(materials, new_thick, problem)
                    if trial_score < current_score - 1e-12:
                        current_thick = new_thick
                        current_score = trial_score
                        improved = True
                        break
    
    return materials, current_thick

def _generate_structured_candidates(problem):
    """Generate diverse structured initial candidates."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    candidates = []
    
    qw_L = centre / (4.0 * nL)
    qw_H = centre / (4.0 * nH)
    
    # Single quarter-wave layers
    candidates.append((["L"], [qw_L]))
    candidates.append((["H"], [qw_H]))
    
    # Two-layer quarter-wave pairs
    if qw_L + qw_H <= max_total:
        candidates.append((["L", "H"], [qw_L, qw_H]))
        candidates.append((["H", "L"], [qw_H, qw_L]))
        
    # Three-layer L-H-L
    if 2*qw_L + qw_H <= max_total:
        candidates.append((["L", "H", "L"], [qw_L, qw_H, qw_L]))
        
    # Alternating stacks of various lengths
    for n in range(4, 12):
        mats = []
        thks = []
        total = 0.0
        for i in range(n):
            if i % 2 == 0:
                mat = "L"
                thk = qw_L
            else:
                mat = "H"
                thk = qw_H
            if total + thk > max_total:
                break
            mats.append(mat)
            thks.append(thk)
            total += thk
        if len(mats) >= 2:
            candidates.append((mats, thks))
            
    # Non-alternating patterns (HH, LL blocks)
    patterns = [
        ["L", "H", "H", "L"],
        ["H", "L", "L", "H"],
        ["L", "H", "H", "L", "H"],
        ["H", "L", "L", "H", "L"],
        ["L", "L", "H", "H", "L", "L"],
        ["H", "H", "L", "L", "H", "H"],
    ]
    for pat in patterns:
        mats = list(pat)
        thks = []
        total = 0.0
        for mat in mats:
            thk = qw_L if mat == "L" else qw_H
            if total + thk > max_total:
                break
            thks.append(thk)
            total += thk
        if len(mats) == len(thks) and len(mats) >= 2:
            candidates.append((mats, thks))
            
    # Randomized structures
    rng = np.random.default_rng(42)
    for _ in range(40):
        n_layers = int(rng.integers(2, 16))
        mats = []
        thks = []
        total = 0.0
        for _ in range(n_layers):
            mat = 'L' if rng.random() > 0.5 else 'H'
            thk = float(rng.uniform(15, 160))
            if total + thk > max_total:
                break
            mats.append(mat)
            thks.append(thk)
            total += thk
        if len(mats) >= 2:
             candidates.append((mats, thks))
             
    return candidates

def _genetic_algorithm(problem, deadline, pop_size=60, max_generations=400):
    """
    Genetic algorithm optimizing both material sequence and thicknesses.
    """
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    
    rng = np.random.default_rng(123)
    
    def encode_individual():
        """Generate a random valid individual."""
        n = int(rng.integers(2, max_layers + 1))
        mats = []
        thks = []
        total = 0.0
        for _ in range(n):
            mat = 'L' if rng.random() < 0.5 else 'H'
            thk = float(rng.uniform(10, min(160, max_layer)))
            if total + thk > max_total:
                if mats:
                    break
                thk = min(thk, max_total - total)
                if thk < 1:
                    thk = 1.0
            mats.append(mat)
            thks.append(thk)
            total += thk
        return mats, thks
    
    def evaluate_individual(mats, thks):
        if not mats or len(mats) != len(thks):
            return float('inf')
        if len(mats) > max_layers:
            return float('inf')
        for t in thks:
            if t < 0 or t > max_layer + 1e-9:
                return float('inf')
        if sum(thks) > max_total + 1e-6:
            return float('inf')
        return _mean_reflectance_pct(mats, thks, problem)
    
    def crossover(parent1, parent2):
        mats1, thks1 = parent1
        mats2, thks2 = parent2
        
        len1 = len(mats1)
        len2 = len(mats2)
        
        if len1 == 0 or len2 == 0:
            return (list(mats1), list(thks1)) if len1 > 0 else (list(mats2), list(thks2))
        
        if rng.random() < 0.5:
            cut1 = int(rng.integers(1, len1 + 1))
            cut2 = int(rng.integers(1, len2 + 1))
            
            new_mats = mats1[:cut1] + mats2[cut2-1:]
            new_thks = thks1[:cut1] + thks2[cut2-1:]
            
            while len(new_mats) > max_layers:
                idx = int(rng.integers(len(new_mats)))
                new_mats.pop(idx)
                new_thks.pop(idx)
            
            total = sum(new_thks)
            if total > max_total and total > 0:
                scale = max_total / total
                new_thks = [t * scale for t in new_thks]
            
            new_thks = [max(1.0, min(max_layer, t)) for t in new_thks]
            
            return (new_mats, new_thks)
        else:
            if rng.random() < 0.5:
                new_mats = list(mats1)
            else:
                new_mats = list(mats2)
            
            n = len(new_mats)
            new_thks = []
            for i in range(n):
                if i < len(thks1) and i < len(thks2):
                    t = 0.5 * thks1[i] + 0.5 * thks2[i]
                elif i < len(thks1):
                    t = thks1[i]
                else:
                    t = thks2[i]
                new_thks.append(max(1.0, min(max_layer, t)))
            
            total = sum(new_thks)
            if total > max_total and total > 0:
                scale = max_total / total
                new_thks = [t * scale for t in new_thks]
            
            return (new_mats, new_thks)
    
    def mutate(individual):
        mats, thks = individual
        n = len(mats)
        if n == 0:
            return encode_individual()
        
        new_mats = list(mats)
        new_thks = list(thks)
        
        mut_type = rng.random()
        
        if mut_type < 0.3:
            i = int(rng.integers(n))
            new_thks[i] = max(1.0, min(max_layer, new_thks[i] * float(rng.uniform(0.5, 2.0))))
        elif mut_type < 0.5:
            i = int(rng.integers(n))
            new_mats[i] = 'H' if new_mats[i] == 'L' else 'L'
        elif mut_type < 0.7 and n < max_layers:
            pos = int(rng.integers(n + 1))
            mat = 'L' if rng.random() < 0.5 else 'H'
            thk = float(rng.uniform(10, min(100, max_layer)))
            new_mats.insert(pos, mat)
            new_thks.insert(pos, thk)
            total = sum(new_thks)
            if total > max_total and total > 0:
                scale = max_total / total
                new_thks = [t * scale for t in new_thks]
        elif mut_type < 0.85 and n > 2:
            i = int(rng.integers(n))
            new_mats.pop(i)
            new_thks.pop(i)
        else:
            for i in range(len(new_thks)):
                new_thks[i] = max(1.0, min(max_layer, new_thks[i] + float(rng.normal(0, 20))))
        
        total = sum(new_thks)
        if total > max_total and total > 0:
            scale = max_total / total
            new_thks = [t * scale for t in new_thks]
        
        valid_mats = []
        valid_thks = []
        for m, t in zip(new_mats, new_thks):
            if t >= 1.0:
                valid_mats.append(m)
                valid_thks.append(min(max_layer, t))
        
        if len(valid_mats) < 2:
            return encode_individual()
        
        return (valid_mats, valid_thks)
    
    # Initialize population
    population = []
    scores = []
    
    for mats, thks in _generate_structured_candidates(problem)[:20]:
        population.append((list(mats), list(thks)))
        scores.append(evaluate_individual(mats, thks))
    
    while len(population) < pop_size:
        ind = encode_individual()
        population.append(ind)
        scores.append(evaluate_individual(ind[0], ind[1]))
    
    best_mats, best_thks = population[0]
    best_score = scores[0]
    
    stagnant_gens = 0
    
    for gen in range(max_generations):
        if time.time() > deadline:
            break
        
        sorted_idx = np.argsort(scores)
        population = [population[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]
        
        if scores[0] < best_score:
            best_score = scores[0]
            best_mats, best_thks = population[0]
            stagnant_gens = 0
        else:
            stagnant_gens += 1
        
        elite_count = max(5, pop_size // 5)
        new_population = [population[i] for i in range(elite_count)]
        new_scores = [scores[i] for i in range(elite_count)]
        
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            def tournament_select():
                idxs = list(rng.integers(len(population), size=3))
                best_i = min(idxs, key=lambda i: scores[i])
                return population[best_i]
            
            p1 = tournament_select()
            p2 = tournament_select()
            
            if rng.random() < 0.7:
                child = crossover(p1, p2)
            else:
                child = p1 if rng.random() < 0.5 else p2
            
            if rng.random() < 0.3:
                child = mutate(child)
            
            mats, thks = child
            if not mats or len(mats) > max_layers:
                child = encode_individual()
                mats, thks = child
            
            score = evaluate_individual(mats, thks)
            if score == float('inf'):
                child = encode_individual()
                mats, thks = child
                score = evaluate_individual(mats, thks)
            
            new_population.append(child)
            new_scores.append(score)
        
        population = new_population
        scores = new_scores
        
        if len(scores) > 0 and (max(scores) - min(scores)) < 1e-8:
            break
        
        # If stagnant for many generations, inject random diversity
        if stagnant_gens > 20:
            for _ in range(5):
                new_ind = encode_individual()
                new_population.append(new_ind)
                new_scores.append(evaluate_individual(new_ind[0], new_ind[1]))
            # Trim back to pop_size
            if len(new_population) > pop_size:
                all_idx = np.argsort(new_scores)
                new_population = [new_population[i] for i in all_idx[:pop_size]]
                new_scores = [new_scores[i] for i in all_idx[:pop_size]]
            population = new_population
            scores = new_scores
            stagnant_gens = 0
    
    return best_mats, best_thks

def design(problem):
    """Multi-start optimization: structured GA + Nelder-Mead + coordinate descent."""
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN
    
    # Phase 1: Generate and evaluate initial candidates
    candidates = _generate_structured_candidates(problem)
    
    best_score = float('inf')
    best_mats = None
    best_thks = None
    
    for mats, thks in candidates:
        if time.time() > deadline:
            break
        pre_score = _evaluate(mats, thks, problem)
        if pre_score < best_score:
            best_score = pre_score
            best_mats = list(mats)
            best_thks = list(thks)
    
    # Phase 2: Run GA for global exploration (~45% of time)
    ga_deadline = start_time + (deadline - start_time) * 0.45
    if time.time() < ga_deadline:
        ga_mats, ga_thks = _genetic_algorithm(problem, ga_deadline)
        ga_score = _evaluate(ga_mats, ga_thks, problem)
        if ga_score < best_score:
            best_score = ga_score
            best_mats = ga_mats
            best_thks = ga_thks
    
    # Phase 3: Nelder-Mead local refinement
    if best_mats is not None and time.time() < deadline - 15:
        opt_mats, opt_thks = _nelder_mead(
            list(best_mats), list(best_thks), problem, deadline
        )
        opt_score = _evaluate(opt_mats, opt_thks, problem)
        if opt_score < best_score:
            best_score = opt_score
            best_mats = opt_mats
            best_thks = opt_thks
    
    # Phase 4: Coordinate descent fine-tuning
    if best_mats is not None and time.time() < deadline - 10:
        opt_mats, opt_thks = _coordinate_descent(
            list(best_mats), list(best_thks), problem, deadline
        )
        opt_score = _evaluate(opt_mats, opt_thks, problem)
        if opt_score < best_score:
            best_mats = opt_mats
            best_thks = opt_thks
            best_score = opt_score
            
    # Phase 5: Random restart local search
    if best_mats is not None and time.time() < deadline - 5:
        rng = np.random.default_rng(time.time_ns())
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            perturbed_thks = [max(1.0, min(problem["max_layer_nm"], t + rng.normal(0, 20))) for t in best_thks]
            total = sum(perturbed_thks)
            if total > problem["max_total_nm"]:
                scale = problem["max_total_nm"] / total
                perturbed_thks = [t * scale for t in perturbed_thks]
                
            opt_mats, opt_thks = _nelder_mead(
                list(best_mats), perturbed_thks, problem, deadline
            )
            opt_score = _evaluate(opt_mats, opt_thks, problem)
            if opt_score < best_score:
                best_mats = opt_mats
                best_thks = opt_thks
                best_score = opt_score

    # Ensure validity
    if best_mats is None or not _is_valid(best_mats, best_thks,
                                           problem["max_layers"],
                                           problem["max_total_nm"],
                                           problem["max_layer_nm"]):
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        best_mats = ["L"]
        best_thks = [centre / (4.0 * problem["materials"]["L"])]
    
    best_thks = [max(0.0, min(problem["max_layer_nm"], t)) for t in best_thks]
    
    return best_mats, best_thks
# EVOLVE-BLOCK-END
