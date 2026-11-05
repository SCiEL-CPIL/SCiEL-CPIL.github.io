"""Broadband antireflection coating design via vectorized evaluation + Nelder-Mead + GA."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import random

_DEADLINE = 145.0  # safety margin before 150s kill

def _reflectance_vectorized(materials, thicknesses, wls, n0, ns):
    """Compute reflectance (0..1) at all wavelengths vectorized."""
    n_layers = len(materials)
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full(len(wls), r * r)
    
    n_wl = len(wls)
    # Total matrix starts as identity for each wavelength
    M = np.ones((n_wl, 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for i in range(n_layers):
        n = materials[i]
        d = thicknesses[i]
        # beta = 2*pi*n*d/wl
        beta = 2.0 * math.pi * n * d / wls
        cb = np.cos(beta)
        sb = np.sin(beta)
        # Characteristic matrix for this layer
        Mi = np.ones((n_wl, 2, 2), dtype=complex)
        Mi[:, 0, 0] = cb
        Mi[:, 0, 1] = 1j * sb / n
        Mi[:, 1, 0] = 1j * n * sb
        Mi[:, 1, 1] = cb
        # M = M @ Mi (elementwise matrix multiply)
        M = M @ Mi
    
    # Admittance Y = (M10 + M11*ns) / (M00 + M01*ns)
    Y = (M[:, 1, 0] + M[:, 1, 1] * ns) / (M[:, 0, 0] + M[:, 0, 1] * ns)
    r = (n0 - Y) / (n0 + Y)
    return np.real(r * np.conj(r))

def _mean_reflectance_pct(materials, thicknesses, band, n0, ns, n_wl=401):
    """Mean reflectance in percent over the band."""
    wls = np.linspace(band[0], band[1], n_wl)
    refl = _reflectance_vectorized(materials, thicknesses, wls, n0, ns)
    return 100.0 * np.mean(refl)

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
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    return _mean_reflectance_pct(materials, thicknesses, band, n0, ns, n_wl)

def _nelder_mead_optimize(materials, thicknesses, problem, max_iters=200, n_wl_eval=60):
    """Nelder-Mead simplex optimization on thicknesses."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    wls_coarse = np.linspace(band[0], band[1], n_wl_eval)
    
    def eval_coarse(thks):
        if len(thks) != len(materials):
            return float('inf')
        for t in thks:
            if t < -1e-9 or t > max_layer + 1e-9:
                return float('inf')
        if sum(thks) > max_total + 1e-6:
            return float('inf')
        refl = _reflectance_vectorized(materials, thks, wls_coarse, n0, ns)
        return 100.0 * np.mean(refl)
    
    n = len(thicknesses)
    if n == 0:
        return list(thicknesses)
    
    # Initialize simplex
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    # Create simplex: n+1 points
    simplex = []
    fvals = []
    
    # First point is the current
    thks0 = list(thicknesses)
    simplex.append(thks0)
    fvals.append(eval_coarse(thks0))
    
    # Other points perturbed
    for i in range(n):
        thks_i = list(thicknesses)
        step = max(10.0, 0.1 * abs(thicknesses[i]))
        thks_i[i] += step
        if thks_i[i] < 0:
            thks_i[i] = 0
        if thks_i[i] > max_layer:
            thks_i[i] = max_layer
        simplex.append(thks_i)
        fvals.append(eval_coarse(thks_i))
    
    best_idx = 0
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        
        # Sort by function value
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if fvals[0] < 0.1:
            break
        
        # Centroid of all but worst
        centroid = [np.mean([simplex[i][j] for i in range(n)]) for j in range(n)]
        
        # Reflection
        xr = [centroid[j] + alpha * (centroid[j] - simplex[n][j]) for j in range(n)]
        fr = eval_coarse(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
            fe = eval_coarse(xe)
            if fe < fr:
                simplex[n] = xe
                fvals[n] = fe
            else:
                simplex[n] = xr
                fvals[n] = fr
        elif fr < fvals[n - 1]:
            simplex[n] = xr
            fvals[n] = fr
        else:
            # Contraction
            if fr < fvals[n]:
                xc = [centroid[j] + rho * (simplex[n][j] - centroid[j]) for j in range(n)]
                fc = eval_coarse(xc)
                if fc <= fr:
                    simplex[n] = xc
                    fvals[n] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        fvals[i] = eval_coarse(simplex[i])
            else:
                xc = [centroid[j] - rho * (centroid[j] - simplex[n][j]) for j in range(n)]
                fc = eval_coarse(xc)
                if fc <= fvals[n]:
                    simplex[n] = xc
                    fvals[n] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        fvals[i] = eval_coarse(simplex[i])
        
        # Clamp
        for i in range(n + 1):
            for j in range(n):
                if simplex[i][j] < 0:
                    simplex[i][j] = 0
                if simplex[i][j] > max_layer:
                    simplex[i][j] = max_layer
    
    # Return best
    best_idx = int(np.argmin(fvals))
    return simplex[best_idx]

def _genetic_algorithm(problem, pop_size=30, n_gen=50, n_wl_eval=50):
    """Simple genetic algorithm for layer structure and thicknesses."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    wls_coarse = np.linspace(band[0], band[1], n_wl_eval)
    
    def eval_ind(mats, thks):
        if not _is_valid(mats, thks, max_layers, max_total, max_layer):
            return float('inf')
        refl = _reflectance_vectorized(mats, thks, wls_coarse, n0, ns)
        return 100.0 * np.mean(refl)
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        n_layers = random.randint(1, min(8, max_layers))
        mats = []
        thks = []
        total = 0.0
        for _ in range(n_layers):
            mat = random.choice(['L', 'H'])
            n_mat = problem["materials"][mat]
            thk = random.uniform(20, min(200, max_layer))
            if total + thk > max_total:
                thk = max(0, max_total - total)
            if thk <= 0:
                break
            mats.append(mat)
            thks.append(thk)
            total += thk
        if len(mats) > 0:
            population.append((mats, thks))
    
    # Evaluate
    scores = [eval_ind(m, t) for m, t in population]
    
    best_score = min(scores)
    best_idx = scores.index(best_score)
    best_mats = list(population[best_idx][0])
    best_thks = list(population[best_idx][1])
    
    for gen in range(n_gen):
        if time.time() > _DEADLINE - 15:
            break
        
        # Select parents (tournament)
        def select():
            candidates = random.sample(range(len(population)), min(3, len(population)))
            best_c = min(candidates, key=lambda i: scores[i])
            return population[best_c]
        
        # Create new population
        new_pop = []
        # Elitism
        best_idx = scores.index(min(scores))
        new_pop.append((list(population[best_idx][0]), list(population[best_idx][1])))
        
        while len(new_pop) < pop_size:
            if time.time() > _DEADLINE - 15:
                break
            p1 = select()
            p2 = select()
            
            # Crossover
            if random.random() < 0.7:
                # Uniform crossover on structure
                max_len = max(len(p1[0]), len(p2[0]))
                child_mats = []
                child_thks = []
                total = 0.0
                for i in range(max_len):
                    if i < len(p1[0]) and random.random() < 0.5:
                        mat = p1[0][i]
                        thk = p1[1][i]
                    else:
                        mat = p2[0][i]
                        thk = p2[1][i]
                    if total + thk > max_total:
                        thk = max(0, max_total - total)
                    if thk > 0:
                        child_mats.append(mat)
                        child_thks.append(thk)
                        total += thk
                if len(child_mats) > 0:
                    new_pop.append((child_mats, child_thks))
            else:
                new_pop.append((list(p1[0]), list(p1[1])))
            
            # Mutation
            if new_pop and random.random() < 0.5:
                mats, thks = new_pop[-1]
                mut_type = random.random()
                if mut_type < 0.3 and len(thks) > 1:
                    # Remove a layer
                    idx = random.randint(0, len(thks) - 1)
                    thks.pop(idx)
                    mats.pop(idx)
                elif mut_type < 0.5 and len(mats) < max_layers:
                    # Add a layer
                    pos = random.randint(0, len(mats))
                    mat = random.choice(['L', 'H'])
                    thk = random.uniform(20, min(150, max_layer))
                    if sum(thks) + thk <= max_total:
                        mats.insert(pos, mat)
                        thks.insert(pos, thk)
                elif thks:
                    # Mutate thickness
                    idx = random.randint(0, len(thks) - 1)
                    delta = random.uniform(-30, 30)
                    thks[idx] = max(0, min(max_layer, thks[idx] + delta))
                    if sum(thks) > max_total:
                        # Scale down
                        scale = max_total / sum(thks)
                        thks = [t * scale for t in thks]
            
            if new_pop and len(new_pop[-1][0]) == 0:
                new_pop.pop()
        
        # Evaluate new population
        scores = [eval_ind(m, t) for m, t in new_pop]
        
        best_score = min(scores)
        best_idx = scores.index(best_score)
        if best_score < _best_global_score[0]:
            _best_global_score[0] = best_score
            _best_global_mats[0] = list(new_pop[best_idx][0])
            _best_global_thks[0] = list(new_pop[best_idx][1])
        
        population = new_pop
    
    return _best_global_mats[0], _best_global_thks[0]

# Global best tracking for GA
_best_global_score = [float('inf')]
_best_global_mats = [[]]
_best_global_thks = [[]]

def _generate_candidates(problem):
    """Generate diverse initial candidates."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    candidates = []
    
    # Quarter-wave L
    qw_L = centre / (4.0 * nL)
    candidates.append((["L"], [qw_L]))
    
    # Quarter-wave H
    qw_H = centre / (4.0 * nH)
    candidates.append((["H"], [qw_H]))
    
    # L-H quarter-wave pair
    if qw_L + qw_H <= max_total:
        candidates.append((["L", "H"], [qw_L, qw_H]))
        candidates.append((["H", "L"], [qw_H, qw_L]))
    
    # Multi-layer quarter-wave stacks
    for n in range(3, 9):
        total = 0.0
        mats = []
        thks = []
        for i in range(n):
            if i % 2 == 0:
                mat = "L"
                thk = centre / (4.0 * nL)
            else:
                mat = "H"
                thk = centre / (4.0 * nH)
            if total + thk > max_total:
                break
            mats.append(mat)
            thks.append(thk)
            total += thk
        if len(mats) >= 2:
            candidates.append((mats, thks))
    
    # Non-quarter-wave starting points
    for factor in [0.7, 0.8, 1.2, 1.3]:
        t = centre * factor / (4.0 * nL)
        if t <= max_layer:
            candidates.append((["L"], [t]))
        t = centre * factor / (4.0 * nH)
        if t <= max_layer:
            candidates.append((["H"], [t]))
    
    # Two-layer non-quarter-wave
    for f1 in [0.8, 1.0, 1.2]:
        for f2 in [0.8, 1.0, 1.2]:
            t1 = centre * f1 / (4.0 * nL)
            t2 = centre * f2 / (4.0 * nH)
            if t1 <= max_layer and t2 <= max_layer and t1 + t2 <= max_total:
                candidates.append((["L", "H"], [t1, t2]))
                candidates.append((["H", "L"], [t2, t1]))
    
    # Three-layer
    for f1 in [0.8, 1.0]:
        for f2 in [0.8, 1.0]:
            for f3 in [0.8, 1.0]:
                t1 = centre * f1 / (4.0 * nL)
                t2 = centre * f2 / (4.0 * nH)
                t3 = centre * f3 / (4.0 * nL)
                if (t1 <= max_layer and t2 <= max_layer and t3 <= max_layer and
                    t1 + t2 + t3 <= max_total):
                    candidates.append((["L", "H", "L"], [t1, t2, t3]))
    
    # Random candidates
    random.seed(42)
    for _ in range(10):
        n_layers = random.randint(2, 6)
        mats = []
        thks = []
        total = 0.0
        for _ in range(n_layers):
            mat = random.choice(['L', 'H'])
            n_mat = problem["materials"][mat]
            thk = random.uniform(30, min(200, max_layer))
            if total + thk > max_total:
                thk = max(0, max_total - total)
            if thk <= 0:
                break
            mats.append(mat)
            thks.append(thk)
            total += thk
        if len(mats) > 0:
            candidates.append((mats, thks))
    
    return candidates

def design(problem):
    """Multi-start with Nelder-Mead + GA for broadband antireflection coating."""
    start_time = time.time()
    random.seed(int(start_time * 1000) % 100000)
    
    candidates = _generate_candidates(problem)
    
    best_score = float('inf')
    best_mats = None
    best_thks = None
    
    # Phase 1: Evaluate candidates and run Nelder-Mead on top ones
    # First, quick pre-evaluation
    pre_scores = []
    for mats, thks in candidates:
        if time.time() > _DEADLINE - 30:
            break
        pre_score = _evaluate(mats, thks, problem, n_wl=100)
        pre_scores.append((pre_score, mats, thks))
    
    pre_scores.sort(key=lambda x: x[0])
    
    # Run Nelder-Mead on top candidates
    n_optimize = min(8, len(pre_scores))
    for i in range(n_optimize):
        if time.time() > _DEADLINE - 20:
            break
        pre_score, mats, thks = pre_scores[i]
        
        # Nelder-Mead optimization
        opt_thks = _nelder_mead_optimize(mats, thks, problem, max_iters=150, n_wl_eval=50)
        
        # Check validity
        if _is_valid(mats, opt_thks, problem["max_layers"], problem["max_total_nm"], problem["max_layer_nm"]):
            opt_score = _evaluate(mats, opt_thks, problem, n_wl=100)
            if opt_score < best_score:
                best_score = opt_score
                best_mats = list(mats)
                best_thks = list(opt_thks)
    
    # Phase 2: Genetic Algorithm if time permits
    if time.time() < _DEADLINE - 25:
        # Initialize global best
        _best_global_score[0] = best_score
        if best_mats is not None:
            _best_global_mats[0] = list(best_mats)
            _best_global_thks[0] = list(best_thks)
        
        ga_mats, ga_thks = _genetic_algorithm(problem, pop_size=25, n_gen=40, n_wl_eval=50)
        
        if ga_mats and _best_global_score[0] < best_score:
            best_score = _best_global_score[0]
            best_mats = list(ga_mats)
            best_thks = list(ga_thks)
    
    # Phase 3: Final refinement with Nelder-Mead on the best
    if best_mats is not None and time.time() < _DEADLINE - 10:
        final_thks = _nelder_mead_optimize(best_mats, best_thks, problem, max_iters=100, n_wl_eval=60)
        if _is_valid(best_mats, final_thks, problem["max_layers"], problem["max_total_nm"], problem["max_layer_nm"]):
            final_score = _evaluate(best_mats, final_thks, problem, n_wl=100)
            if final_score < best_score:
                best_thks = list(final_thks)
                best_score = final_score
    
    # Ensure validity
    if best_mats is None or not _is_valid(best_mats, best_thks, 
                                           problem["max_layers"],
                                           problem["max_total_nm"],
                                           problem["max_layer_nm"]):
        # Fallback: single quarter-wave L
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        best_mats = ["L"]
        best_thks = [centre / (4.0 * problem["materials"]["L"])]
    
    # Clamp thicknesses
    best_thks = [max(0.0, min(problem["max_layer_nm"], t)) for t in best_thks]
    
    return best_mats, best_thks
# EVOLVE-BLOCK-END
