"""Broadband antireflection coating design: multi-start DE + SA + L-BFGS-B + Nelder-Mead + coordinate descent.

Strategy:
1. Generate diverse initial solutions (quarter-wave stacks, random designs)
2. Differential Evolution for global search over variable-length layer sequences
3. Simulated Annealing for escaping local minima
4. L-BFGS-B continuous optimization on top solutions (fixed material sequence)
5. Nelder-Mead local refinement
6. Final coordinate-descent polish
"""
import time
import numpy as np
import random
import math

# EVOLVE-BLOCK-START

def _reflectance_vectorized(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """Compute reflectance at all wavelengths using characteristic matrix method (vectorized)."""
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    n_layers = len(materials)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wls, r * r, dtype=np.float64)
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    for mat, thick in zip(materials, thicknesses):
        n = materials_dict[mat]
        if thick < 1e-9:
            continue
        beta = 2.0 * np.pi * n * thick / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        L11 = cos_b
        L12 = 1.0j * sin_b / n
        L21 = n * 1.0j * sin_b
        L22 = cos_b
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    Y_s = substrate_index
    Y = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    Y_0 = ambient_index
    r = (Y_0 - Y) / (Y_0 + Y)
    R = np.abs(r) ** 2
    return np.real(R)

def _evaluate(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict):
    """Compute mean reflectance in percent."""
    refls = _reflectance_vectorized(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    return True

def _clamp_thick(t, max_layer_nm):
    return max(0.0, min(t, max_layer_nm))

def _nelder_mead(materials, thicknesses, problem, wls, materials_dict,
                 ambient_index, substrate_index, deadline, max_iter=3000):
    """Nelder-Mead simplex optimization for thicknesses with fixed material sequence."""
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    # Build initial simplex
    simplex = [list(thicknesses)]
    rng = random.Random(12345)
    for i in range(n_layers):
        perturbed = list(thicknesses)
        perturbed[i] = _clamp_thick(perturbed[i] + 10.0 * (1 if rng.random() < 0.5 else -1), max_layer_nm)
        simplex.append(perturbed)
    
    # Also add a few random perturbations
    for _ in range(min(3, n_layers)):
        perturbed = list(thicknesses)
        for j in range(n_layers):
            perturbed[j] = _clamp_thick(perturbed[j] + rng.uniform(-20, 20), max_layer_nm)
        simplex.append(perturbed)
    
    # Evaluate
    scores = []
    for sp in simplex:
        if _is_valid(materials, sp, problem):
            scores.append(_evaluate(materials, sp, wls, ambient_index, substrate_index, materials_dict))
        else:
            scores.append(1e6)
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        if abs(scores[-1] - scores[0]) < 1e-14:
            break
        
        centroid = np.mean([np.array(simplex[i]) for i in range(len(simplex) - 1)], axis=0)
        worst = np.array(simplex[-1])
        
        # Reflection
        reflected = centroid + alpha * (centroid - worst)
        reflected = np.array([_clamp_thick(v, max_layer_nm) for v in reflected])
        if _is_valid(materials, reflected, problem):
            ref_score = _evaluate(materials, reflected, wls, ambient_index, substrate_index, materials_dict)
        else:
            ref_score = 1e6
        
        if ref_score < scores[0]:
            expanded = centroid + gamma * (reflected - centroid)
            expanded = np.array([_clamp_thick(v, max_layer_nm) for v in expanded])
            if _is_valid(materials, expanded, problem):
                exp_score = _evaluate(materials, expanded, wls, ambient_index, substrate_index, materials_dict)
            else:
                exp_score = 1e6
            if exp_score < ref_score:
                simplex[-1] = expanded.tolist()
                scores[-1] = exp_score
            else:
                simplex[-1] = reflected.tolist()
                scores[-1] = ref_score
        elif ref_score < scores[-2]:
            simplex[-1] = reflected.tolist()
            scores[-1] = ref_score
        else:
            contracted = centroid + rho * (worst - centroid)
            contracted = np.array([_clamp_thick(v, max_layer_nm) for v in contracted])
            if _is_valid(materials, contracted, problem):
                con_score = _evaluate(materials, contracted, wls, ambient_index, substrate_index, materials_dict)
            else:
                con_score = 1e6
            if con_score < scores[-1]:
                simplex[-1] = contracted.tolist()
                scores[-1] = con_score
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n_layers)]
                    if _is_valid(materials, simplex[i], problem):
                        scores[i] = _evaluate(materials, simplex[i], wls, ambient_index, substrate_index, materials_dict)
                    else:
                        scores[i] = 1e6
    
    best_idx = int(np.argmin(scores))
    return materials, simplex[best_idx]

def _coordinate_descent(materials, thicknesses, problem, wls, materials_dict,
                        ambient_index, substrate_index, deadline, max_rounds=150):
    """Coordinate descent with adaptive step for final polish."""
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    max_layer_nm = problem.get("max_layer_nm", 400)
    current_thicks = list(thicknesses)
    current_score = _evaluate(materials, current_thicks, wls, ambient_index, substrate_index, materials_dict)
    step_sizes = [5.0] * n_layers
    min_step = 0.005
    max_step = 30.0
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n_layers):
            best_local_score = current_score
            best_local_thick = current_thicks[i]
            for factor in [1.0, 0.5, 0.2, 0.1, 0.05, 2.0, 4.0]:
                step = step_sizes[i] * factor
                for sign in [1, -1]:
                    new_thick = _clamp_thick(current_thicks[i] + sign * step, max_layer_nm)
                    new_thicks = list(current_thicks)
                    new_thicks[i] = new_thick
                    if _is_valid(materials, new_thicks, problem):
                        new_score = _evaluate(materials, new_thicks, wls, ambient_index, substrate_index, materials_dict)
                        if new_score < best_local_score:
                            best_local_score = new_score
                            best_local_thick = new_thick
            if best_local_score < current_score - 1e-13:
                current_thicks[i] = best_local_thick
                current_score = best_local_score
                improved = True
        if not improved:
            step_sizes = [s * 0.6 for s in step_sizes]
            if min(step_sizes) < min_step:
                break
        else:
            step_sizes = [min(s * 1.3, max_step) for s in step_sizes]
    return materials, current_thicks

def _sa_optimize(materials, thicknesses, problem, wls, materials_dict,
                 ambient_index, substrate_index, deadline, rng,
                 T0=5.0, T1=0.001, n_steps=50000):
    """Simulated annealing with variable-length layer sequences."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    nL = materials_dict['L']
    nH = materials_dict['H']
    
    current_mats = list(materials)
    current_thicks = list(thicknesses)
    current_score = _evaluate(current_mats, current_thicks, wls, ambient_index, substrate_index, materials_dict)
    
    best_mats = list(current_mats)
    best_thicks = list(current_thicks)
    best_score = current_score
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        T = T0 * (T1 / T0) ** (step / max(1, n_steps - 1))
        
        # Generate neighbor
        new_mats = list(current_mats)
        new_thicks = list(current_thicks)
        
        move_type = rng.random()
        
        if move_type < 0.3 and len(new_mats) > 1:
            # Perturb a random thickness
            i = rng.randint(0, len(new_mats) - 1)
            delta = rng.gauss(0, 15)
            new_thicks[i] = _clamp_thick(new_thicks[i] + delta, max_layer_nm)
        elif move_type < 0.4 and len(new_mats) > 0:
            # Flip a random material
            i = rng.randint(0, len(new_mats) - 1)
            new_mats[i] = 'H' if new_mats[i] == 'L' else 'L'
        elif move_type < 0.5 and len(new_mats) < max_layers:
            # Add a layer
            pos = rng.randint(0, len(new_mats))
            mat = rng.choice(['L', 'H'])
            thick = rng.uniform(20, 100)
            new_mats.insert(pos, mat)
            new_thicks.insert(pos, thick)
        elif move_type < 0.6 and len(new_mats) > 1:
            # Remove a layer
            pos = rng.randint(0, len(new_mats) - 1)
            new_mats.pop(pos)
            new_thicks.pop(pos)
        else:
            # Perturb multiple thicknesses
            n_perturb = rng.randint(1, min(3, len(new_mats)))
            indices = rng.sample(range(len(new_mats)), n_perturb)
            for i in indices:
                delta = rng.gauss(0, 10)
                new_thicks[i] = _clamp_thick(new_thicks[i] + delta, max_layer_nm)
        
        # Check validity
        if not _is_valid(new_mats, new_thicks, problem):
            # Try to repair
            total = sum(new_thicks)
            if total > max_total_nm:
                scale = max_total_nm / total
                new_thicks = [t * scale for t in new_thicks]
            if len(new_mats) > max_layers:
                new_mats = new_mats[:max_layers]
                new_thicks = new_thicks[:max_layers]
            if not _is_valid(new_mats, new_thicks, problem):
                continue
        
        new_score = _evaluate(new_mats, new_thicks, wls, ambient_index, substrate_index, materials_dict)
        
        # Accept or reject
        delta_score = new_score - current_score
        if delta_score < 0 or rng.random() < math.exp(-delta_score / max(T, 1e-10)):
            current_mats = new_mats
            current_thicks = new_thicks
            current_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_mats = list(new_mats)
                best_thicks = list(new_thicks)
    
    return best_mats, best_thicks

def _differential_evolution(initial_solutions, problem, wls, materials_dict,
                            ambient_index, substrate_index, deadline, 
                            pop_size=30, max_generations=500, 
                            F=0.8, CR=0.9, seed=42):
    """Differential Evolution for variable-length layer sequences."""
    rng = random.Random(seed)
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # Initialize population from initial solutions + random perturbations
    population = []
    for mats, thicks in initial_solutions:
        population.append((list(mats), list(thicks)))
    
    # Fill up to pop_size with random or perturbed solutions
    while len(population) < pop_size:
        if initial_solutions:
            base_mats, base_thicks = rng.choice(initial_solutions)
            mats = list(base_mats)
            thicks = list(base_thicks)
            # Perturb
            if mats:
                for i in range(len(mats)):
                    thicks[i] = _clamp_thick(thicks[i] + rng.uniform(-20, 20), max_layer_nm)
                    if rng.random() < 0.1:
                        mats[i] = 'H' if mats[i] == 'L' else 'L'
            # Check validity and repair
            if not _is_valid(mats, thicks, problem):
                # Simple repair: scale down if total too high
                total = sum(thicks)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    thicks = [t * scale for t in thicks]
                if not _is_valid(mats, thicks, problem):
                    continue
        else:
            continue
        population.append((mats, thicks))
        
    # If still not enough, generate random
    while len(population) < pop_size:
        n_layers = rng.randint(2, min(10, max_layers))
        mats = [rng.choice(['L', 'H']) for _ in range(n_layers)]
        thicks = [rng.uniform(20, 100) for _ in range(n_layers)]
        if not _is_valid(mats, thicks, problem):
            total = sum(thicks)
            if total > max_total_nm:
                scale = max_total_nm / total
                thicks = [t * scale for t in thicks]
        if _is_valid(mats, thicks, problem):
            population.append((mats, thicks))

    # Evaluate initial population
    scores = []
    for mats, thicks in population:
        if _is_valid(mats, thicks, problem):
            scores.append(_evaluate(mats, thicks, wls, ambient_index, substrate_index, materials_dict))
        else:
            scores.append(1e6)
            
    best_idx = int(np.argmin(scores))
    best_mats = list(population[best_idx][0])
    best_thicks = list(population[best_idx][1])
    best_score = scores[best_idx]
    
    for gen in range(max_generations):
        if time.time() > deadline:
            break
        
        for i in range(len(population)):
            if time.time() > deadline:
                break
            
            # Select three distinct individuals
            a, b, c = rng.sample(range(len(population)), 3)
            
            # Mutation: v = x_a + F * (x_b - x_c)
            # Handle variable lengths by taking max length and padding
            len_a = len(population[a][0])
            len_b = len(population[b][0])
            len_c = len(population[c][0])
            max_len = max(len_a, len_b, len_c)
            
            # Pad with zeros for calculation, then truncate later
            # Materials: Use x_a's materials as base, flip if mutation suggests?
            # Simpler approach: 
            # 1. Determine new material sequence (mostly from x_a, some from x_b/x_c)
            # 2. Calculate thicknesses
            
            # Material selection: 
            # With prob 0.5, use x_a materials. 
            # Else, mix.
            new_mats = []
            for k in range(max_len):
                if k < len_a:
                    mat_a = population[a][0][k]
                else:
                    mat_a = 'L' # Default padding
                if k < len_b:
                    mat_b = population[b][0][k]
                else:
                    mat_b = 'L'
                if k < len_c:
                    mat_c = population[c][0][k]
                else:
                    mat_c = 'L'
                
                # Simple strategy: Keep x_a material, but occasionally flip or take from b/c
                if rng.random() < 0.1:
                    new_mats.append(mat_b)
                elif rng.random() < 0.1:
                    new_mats.append(mat_c)
                else:
                    new_mats.append(mat_a)
            
            # Thickness mutation: v_k = x_a_k + F * (x_b_k - x_c_k)
            new_thicks = []
            for k in range(max_len):
                t_a = population[a][1][k] if k < len_a else 0.0
                t_b = population[b][1][k] if k < len_b else 0.0
                t_c = population[c][1][k] if k < len_c else 0.0
                
                t_mut = t_a + F * (t_b - t_c)
                t_mut = _clamp_thick(t_mut, max_layer_nm)
                new_thicks.append(t_mut)
            
            # Crossover: Uniform crossover with CR
            # Target is x_i
            len_i = len(population[i][0])
            # If lengths differ, we need to handle carefully.
            # Strategy: The new individual length is max_len.
            # We decide for each layer if we take from mutant or target.
            # If target is shorter, we "extend" it.
            
            trial_mats = []
            trial_thicks = []
            
            for k in range(max_len):
                if rng.random() < CR or k == rng.randint(0, max_len - 1):
                    # Take from mutant
                    trial_mats.append(new_mats[k])
                    trial_thicks.append(new_thicks[k])
                else:
                    # Take from target
                    if k < len_i:
                        trial_mats.append(population[i][0][k])
                        trial_thicks.append(population[i][1][k])
                    else:
                        # Target didn't have this layer. Mutant does.
                        # Keep mutant's layer? Or drop?
                        # If we drop, we might lose info. Let's keep mutant's.
                        trial_mats.append(new_mats[k])
                        trial_thicks.append(new_thicks[k])
            
            # Prune trailing zero-thickness layers to keep representation compact
            # Also remove any layer with thickness < 1nm
            cleaned_mats = []
            cleaned_thicks = []
            for m, t in zip(trial_mats, trial_thicks):
                if t > 1.0:
                    cleaned_mats.append(m)
                    cleaned_thicks.append(t)
            
            if len(cleaned_mats) == 0:
                continue
                
            # Check constraints and repair
            if not _is_valid(cleaned_mats, cleaned_thicks, problem):
                # Try to repair total thickness
                total = sum(cleaned_thicks)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    cleaned_thicks = [t * scale for t in cleaned_thicks]
                
                if not _is_valid(cleaned_mats, cleaned_thicks, problem):
                    continue
                    
            trial_score = _evaluate(cleaned_mats, cleaned_thicks, wls, ambient_index, substrate_index, materials_dict)
            
            # Selection
            if trial_score < scores[i]:
                population[i] = (cleaned_mats, cleaned_thicks)
                scores[i] = trial_score
                if trial_score < best_score:
                    best_score = trial_score
                    best_mats = list(cleaned_mats)
                    best_thicks = list(cleaned_thicks)
                    
    return best_mats, best_thicks

def _lbfgsb_optimize(materials, thicknesses, problem, wls, materials_dict,
                     ambient_index, substrate_index, deadline, max_iter=1000):
    """L-BFGS-B continuous optimization for thicknesses with fixed material sequence."""
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    # Bounds: each thickness in [0, max_layer_nm]
    bounds = [(0.0, max_layer_nm)] * n_layers
    
    # Objective function with penalty for total thickness constraint
    def objective(x):
        if sum(x) > max_total_nm + 1e-9:
            return 1e6 + (sum(x) - max_total_nm) * 10.0
        if any(t < 0 for t in x):
            return 1e6
        try:
            return _evaluate(materials, x, wls, ambient_index, substrate_index, materials_dict)
        except Exception:
            return 1e6
    
    # Initial point
    x0 = np.array(thicknesses, dtype=float)
    
    # Use scipy if available, otherwise fall back to Nelder-Mead
    try:
        from scipy.optimize import minimize
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        best_x = result.x
        best_score = result.fun
    except ImportError:
        # Fallback to Nelder-Mead
        _, best_x = _nelder_mead(materials, thicknesses, problem, wls, materials_dict,
                                  ambient_index, substrate_index, deadline, max_iter=1000)
        best_score = _evaluate(materials, best_x, wls, ambient_index, substrate_index, materials_dict)
    
    # Ensure validity
    best_x = np.array([_clamp_thick(t, max_layer_nm) for t in best_x])
    if sum(best_x) > max_total_nm:
        scale = max_total_nm / sum(best_x)
        best_x *= scale
        best_x = np.array([_clamp_thick(t, max_layer_nm) for t in best_x])
    
    if not _is_valid(materials, best_x.tolist(), problem):
        return materials, thicknesses
    
    return materials, best_x.tolist()

def _generate_initial_solutions(problem, wls, materials_dict, band_nm, n_solutions=20, seed=42):
    """Generate diverse initial solutions."""
    rng = random.Random(seed)
    solutions = []
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    nL = materials_dict['L']
    nH = materials_dict['H']
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    
    # Quarter-wave stacks of various lengths
    for n in [1, 3, 5, 7, 9, 11, 13]:
        if n % 2 == 1:
            layers = ['L' if (i % 2 == 0) else 'H' for i in range(n)]
        else:
            layers = ['L' if (i % 2 == 0) else 'H' for i in range(n)]
        thicks = [wl_center / (4.0 * (nL if m == 'L' else nH)) for m in layers]
        if sum(thicks) <= max_total_nm:
            solutions.append((layers, thicks))
    
    # H-start quarter-wave
    for n in [3, 5, 7]:
        layers = ['H' if (i % 2 == 0) else 'L' for i in range(n)]
        thicks = [wl_center / (4.0 * (nH if m == 'H' else nL)) for m in layers]
        if sum(thicks) <= max_total_nm:
            solutions.append((layers, thicks))
    
    # Random designs
    for _ in range(10):
        n_layers = rng.randint(2, 10)
        materials = []
        thicknesses = []
        total = 0.0
        for _ in range(n_layers):
            mat = rng.choice(['L', 'H'])
            n = nL if mat == 'L' else nH
            d = rng.uniform(30.0, min(120.0, max_layer_nm))
            if total + d > max_total_nm:
                d = max(0.0, max_total_nm - total)
            if d < 5.0:
                continue
            materials.append(mat)
            thicknesses.append(d)
            total += d
        if len(materials) >= 1:
            solutions.append((materials, thicknesses))
    
    valid = []
    for mats, thicks in solutions:
        if _is_valid(mats, thicks, problem):
            valid.append((mats, thicks))
    return valid

def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layers = problem.get("max_layers", 20)
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Generate initial solutions
    initial_solutions = _generate_initial_solutions(problem, wls, materials_dict, band_nm, n_solutions=20, seed=42)
    
    if not initial_solutions:
        d_q = wl_center / (4.0 * nL)
        return ['L'], [d_q]
    
    # Evaluate and sort
    scored = []
    for mats, thicks in initial_solutions:
        score = _evaluate(mats, thicks, wls, n_amb, n_sub, materials_dict)
        scored.append((score, mats, thicks))
    scored.sort(key=lambda x: x[0])
    
    best_score = scored[0][0]
    best_materials = list(scored[0][1])
    best_thicknesses = list(scored[0][2])
    
    # Differential Evolution on top solutions
    # Use the top few solutions to seed the DE population
    top_solutions = [(mats, thicks) for score, mats, thicks in scored[:10]]
    
    if time.time() < deadline - 20:
        de_mats, de_thicks = _differential_evolution(
            top_solutions, problem, wls, materials_dict,
            n_amb, n_sub, deadline - 15,
            pop_size=40, max_generations=1000,
            F=0.8, CR=0.9, seed=123)
        
        if _is_valid(de_mats, de_thicks, problem):
            de_score = _evaluate(de_mats, de_thicks, wls, n_amb, n_sub, materials_dict)
            if de_score < best_score:
                best_score = de_score
                best_materials = list(de_mats)
                best_thicknesses = list(de_thicks)
    
    # Simulated Annealing on best
    if time.time() < deadline - 15 and len(best_materials) > 0:
        rng = random.Random(777)
        sa_mats, sa_thicks = _sa_optimize(
            list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
            n_amb, n_sub, deadline - 10, rng,
            T0=5.0, T1=0.001, n_steps=30000)
        if _is_valid(sa_mats, sa_thicks, problem):
            sa_score = _evaluate(sa_mats, sa_thicks, wls, n_amb, n_sub, materials_dict)
            if sa_score < best_score:
                best_score = sa_score
                best_materials = list(sa_mats)
                best_thicknesses = list(sa_thicks)
    
    # L-BFGS-B refinement on best (continuous optimization)
    if time.time() < deadline - 15 and len(best_materials) > 0:
        try:
            lbfgs_mats, lbfgs_thicks = _lbfgsb_optimize(
                list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
                n_amb, n_sub, deadline - 10, max_iter=500)
            if _is_valid(lbfgs_mats, lbfgs_thicks, problem):
                lbfgs_score = _evaluate(lbfgs_mats, lbfgs_thicks, wls, n_amb, n_sub, materials_dict)
                if lbfgs_score < best_score:
                    best_score = lbfgs_score
                    best_materials = list(lbfgs_mats)
                    best_thicknesses = list(lbfgs_thicks)
        except Exception:
            pass
    
    # Nelder-Mead refinement on best
    if time.time() < deadline - 10 and len(best_materials) > 0:
        nm_mats, nm_thicks = _nelder_mead(
            list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
            n_amb, n_sub, deadline - 5, max_iter=2000)
        if _is_valid(nm_mats, nm_thicks, problem):
            nm_score = _evaluate(nm_mats, nm_thicks, wls, n_amb, n_sub, materials_dict)
            if nm_score < best_score:
                best_score = nm_score
                best_materials = list(nm_mats)
                best_thicknesses = list(nm_thicks)
    
    # Coordinate descent polish
    if time.time() < deadline - 3 and len(best_materials) > 0:
        try:
            cd_mats, cd_thicks = _coordinate_descent(
                list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
                n_amb, n_sub, deadline - 1, max_rounds=80)
            if _is_valid(cd_mats, cd_thicks, problem):
                cd_score = _evaluate(cd_mats, cd_thicks, wls, n_amb, n_sub, materials_dict)
                if cd_score < best_score:
                    best_score = cd_score
                    best_materials = list(cd_mats)
                    best_thicknesses = list(cd_thicks)
        except Exception:
            pass
    
    # Ensure validity
    if not _is_valid(best_materials, best_thicknesses, problem):
        d_q = wl_center / (4.0 * nL)
        best_materials = ["L"]
        best_thicknesses = [d_q]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
