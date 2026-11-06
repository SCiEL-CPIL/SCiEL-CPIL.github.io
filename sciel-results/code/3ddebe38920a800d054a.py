"""Broadband antireflection coating design using characteristic-matrix method."""
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
    
    n_layers = len(materials)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wls, r*r)
        
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
    return R

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

def _nelder_mead(objective, x0, bounds, max_iter=500, tol=1e-10, deadline=None):
    """
    Nelder-Mead simplex optimization with bounds.
    """
    n = len(x0)
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x = np.array(x0, dtype=float)
        step = 0.05 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] - bounds[i][0] > 0 else 0.05
        x[i] += step
        simplex.append(x)
    
    # Evaluate initial simplex
    f_values = []
    for x in simplex:
        # Clip to bounds
        x_clipped = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
        f = objective(x_clipped)
        f_values.append(f)
    
    # Order simplex
    order = np.argsort(f_values)
    simplex = [simplex[i] for i in order]
    f_values = [f_values[i] for i in order]
    
    alpha = 1.0   # Reflection
    gamma = 2.0   # Expansion
    rho = 0.5     # Contraction
    sigma = 0.5   # Shrink
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Check convergence
        if abs(f_values[0] - f_values[-1]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = objective(x_r)
        
        if f_r < f_values[0]:
            # Try expansion
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = objective(x_e)
            
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
        elif f_r < f_values[-2]:
            simplex[-1] = x_r
            f_values[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = objective(x_c)
            
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    f_values[i] = objective(simplex[i])
        
        # Reorder
        order = np.argsort(f_values)
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
    
    return simplex[0], f_values[0]

def _sa_step(materials, thicknesses, problem, wls, materials_dict, 
             ambient_index, substrate_index, band_nm, T, rng, deadline):
    """
    Perform one simulated annealing step.
    """
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    n_layers = len(materials)
    
    # Choose a move type
    if n_layers < 2:
        move_type = rng.choice([1, 1, 1, 2, 3])
    elif n_layers >= max_layers:
        move_type = rng.choice([1, 1, 2, 4, 5])
    else:
        move_type = rng.choice([1, 1, 1, 2, 3, 4, 5])
    
    # Current score
    current_score = _mean_reflectance(materials, thicknesses, band_nm, 401, 
                                       ambient_index, substrate_index, materials_dict)
    
    # Apply move
    new_materials = list(materials)
    new_thicknesses = list(thicknesses)
    
    if move_type == 1:  # Perturb a random thickness
        if n_layers == 0:
            return materials, thicknesses, False
        idx = rng.randint(0, n_layers - 1)
        base = max(thicknesses[idx], 50.0)
        delta = rng.gauss(0, base * 0.3)
        new_thicknesses[idx] = max(0.0, min(max_layer_nm, thicknesses[idx] + delta))
        
    elif move_type == 2:  # Flip a random layer type
        if n_layers == 0:
            return materials, thicknesses, False
        idx = rng.randint(0, n_layers - 1)
        new_materials[idx] = 'H' if materials[idx] == 'L' else 'L'
        
    elif move_type == 3:  # Add a layer at a random position
        if n_layers >= max_layers:
            return materials, thicknesses, False
        pos = rng.randint(0, n_layers)
        mat = rng.choice(['L', 'H'])
        n_mat = materials_dict[mat]
        wl_center = 0.5 * (band_nm[0] + band_nm[1])
        thick = wl_center / (4.0 * n_mat)
        thick = max(10.0, min(max_layer_nm, thick * rng.uniform(0.5, 1.5)))
        new_materials.insert(pos, mat)
        new_thicknesses.insert(pos, thick)
        
    elif move_type == 4:  # Remove a random layer
        if n_layers <= 1:
            return materials, thicknesses, False
        idx = rng.randint(0, n_layers - 1)
        new_materials.pop(idx)
        new_thicknesses.pop(idx)
        
    elif move_type == 5:  # Swap adjacent layers
        if n_layers < 2:
            return materials, thicknesses, False
        idx = rng.randint(0, n_layers - 2)
        new_materials[idx], new_materials[idx+1] = new_materials[idx+1], new_materials[idx]
        new_thicknesses[idx], new_thicknesses[idx+1] = new_thicknesses[idx+1], new_thicknesses[idx]
    
    # New score
    new_score = _mean_reflectance(new_materials, new_thicknesses, band_nm, 401, 
                                   ambient_index, substrate_index, materials_dict)
    
    # Acceptance criterion
    if new_score < current_score:
        return new_materials, new_thicknesses, True
    else:
        delta_e = new_score - current_score
        if delta_e == 0:
            return new_materials, new_thicknesses, True
        if T > 1e-8:
            prob = np.exp(-delta_e / T)
            if rng.random() < prob:
                return new_materials, new_thicknesses, True
    
    return materials, thicknesses, False

def _local_refine(materials, thicknesses, problem, wls, materials_dict, 
                  ambient_index, substrate_index, band_nm, deadline, max_iter=100):
    """
    Nelder-Mead refinement for a fixed material sequence.
    """
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    x0 = np.array(thicknesses, dtype=float)
    bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
    
    def objective(x):
        x_clipped = np.clip(x, 0.0, max_layer_nm)
        if np.sum(x_clipped) > max_total_nm:
            return 100.0 + 10.0 * (np.sum(x_clipped) - max_total_nm)
        t_list = [float(t) for t in x_clipped]
        return _mean_reflectance(materials, t_list, band_nm, 401, 
                                  ambient_index, substrate_index, materials_dict)
    
    x_opt, f_opt = _nelder_mead(objective, x0, bounds, max_iter=max_iter, 
                                  tol=1e-10, deadline=deadline)
    x_opt = np.clip(x_opt, 0.0, max_layer_nm)
    
    # Ensure total thickness constraint
    if np.sum(x_opt) > max_total_nm:
        scale = max_total_nm / np.sum(x_opt)
        x_opt = x_opt * scale
    
    refined_thicknesses = [float(t) for t in x_opt]
    return materials, refined_thicknesses

def design(problem):
    """
    Design using simulated annealing over both layer sequence and thicknesses,
    with Nelder-Mead local refinement.
    """
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
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    rng = random.Random(42)
    
    # Initial solution: quarter-wave layers
    init_materials = []
    init_thicknesses = []
    for mat in ['L', 'H', 'L']:
        n = nL if mat == 'L' else nH
        d = wl_center / (4.0 * n)
        init_materials.append(mat)
        init_thicknesses.append(d)
    
    best_materials = list(init_materials)
    best_thicknesses = list(init_thicknesses)
    best_score = _mean_reflectance(best_materials, best_thicknesses, band_nm, 401, 
                                    n_amb, n_sub, materials_dict)
    
    # Simulated annealing
    T_start = 5.0
    T_end = 0.01
    n_iterations = 2000
    
    current_materials = list(init_materials)
    current_thicknesses = list(init_thicknesses)
    current_score = best_score
    
    for i in range(n_iterations):
        if time.time() > deadline - 30:
            break
        
        T = T_start * (T_end / T_start) ** (i / n_iterations)
        
        new_materials, new_thicknesses, accepted = _sa_step(
            current_materials, current_thicknesses, problem, wls, materials_dict,
            n_amb, n_sub, band_nm, T, rng, deadline)
        
        if accepted:
            current_materials = new_materials
            current_thicknesses = new_thicknesses
            current_score = _mean_reflectance(current_materials, current_thicknesses, band_nm, 
                                              401, n_amb, n_sub, materials_dict)
            
            if current_score < best_score:
                best_score = current_score
                best_materials = list(current_materials)
                best_thicknesses = list(current_thicknesses)
    
    # Final local refinement on the best solution
    if time.time() < deadline - 5 and len(best_materials) > 0:
        try:
            refined_materials, refined_thicknesses = _local_refine(
                list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
                n_amb, n_sub, band_nm, deadline, max_iter=200)
            
            if _is_valid(refined_materials, refined_thicknesses, problem):
                refined_score = _mean_reflectance(refined_materials, refined_thicknesses, 
                                                   band_nm, 401, n_amb, n_sub, materials_dict)
                if refined_score < best_score:
                    best_score = refined_score
                    best_materials = refined_materials
                    best_thicknesses = refined_thicknesses
        except Exception:
            pass
    
    # Ensure validity
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [wl_center / (4.0 * nL)]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
