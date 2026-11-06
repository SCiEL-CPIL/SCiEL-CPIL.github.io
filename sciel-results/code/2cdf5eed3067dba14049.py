"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _reflectance_vectorized(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """Compute reflectance at all wavelengths using characteristic matrix method (vectorized)."""
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    if len(materials) == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wls, r * r, dtype=float)
    
    n_layers = len(materials)
    
    # Start with identity matrix for all wavelengths
    B = np.ones_like(wls, dtype=complex)
    C = np.zeros_like(wls, dtype=complex)
    D = np.zeros_like(wls, dtype=complex)
    E = np.ones_like(wls, dtype=complex)
    
    for mat, thick in zip(materials, thicknesses):
        n = materials_dict[mat]
        if thick < 1e-9:
            continue
        beta = 2.0 * np.pi * n * thick / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Layer matrix: [[cos, i*sin/n], [i*n*sin, cos]]
        m11 = cos_b
        m12 = 1.0j * sin_b / n
        m21 = 1.0j * n * sin_b
        m22 = cos_b
        
        # Multiply total = total @ layer_matrix
        new_B = B * m11 + C * m21
        new_C = B * m12 + C * m22
        new_D = D * m11 + E * m21
        new_E = D * m12 + E * m22
        B, C, D, E = new_B, new_C, new_D, new_E
    
    # Substrate admittance
    Y_s = substrate_index
    Y_0 = ambient_index
    
    # Input admittance
    Y = (D * Y_s + E) / (B * Y_s + C)
    
    r = (Y_0 - Y) / (Y_0 + Y)
    return np.abs(r) ** 2


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


def _nelder_mead(f, x0, bounds, max_iter=200, tol=1e-6):
    """Simple Nelder-Mead optimization with bounds."""
    n = len(x0)
    # Initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x_trial = np.array(x0, dtype=float)
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] - bounds[i][0] > 0 else 0.1
        x_trial[i] += step
        x_trial = np.clip(x_trial, bounds, bounds)
        simplex.append(x_trial)
    
    values = [f(x) for x in simplex]
    
    for iteration in range(max_iter):
        # Sort simplex
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[-1] - values[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, bounds, bounds)
        f_r = f(x_r)
        
        if f_r < values[0]:
            # Expansion
            gamma = 2.0
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, bounds, bounds)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        elif f_r < values[-2]:
            simplex[-1] = x_r
            values[-1] = f_r
        else:
            # Contraction
            rho = 0.5
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, bounds, bounds)
            f_c = f(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds, bounds)
                    values[i] = f(simplex[i])
    
    # Return best
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]


def design(problem):
    """Design antireflection coating using Nelder-Mead optimization with restarts."""
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layers = problem["max_layers"]
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    def objective(thicknesses, materials):
        if not _is_valid(materials, thicknesses, problem):
            return 100.0
        return _mean_reflectance(materials, thicknesses, band_nm, 401, n_amb, n_sub, materials_dict)
    
    best_materials = ["L"]
    best_thicknesses = [wl_center / (4.0 * nL)]
    best_score = objective(best_thicknesses, best_materials)
    
    # Define candidate patterns with initial thickness estimates
    def make_initial_thicknesses(pattern, wl_ref):
        thicks = []
        for m in pattern:
            n = nL if m == 'L' else nH
            thicks.append(wl_ref / (4.0 * n))
        return thicks
    
    # Generate diverse starting patterns
    patterns = []
    
    # Single layer
    patterns.append(('L',))
    patterns.append(('H',))
    
    # Two layers
    patterns.append(('L', 'H'))
    patterns.append(('H', 'L'))
    
    # Three layers
    patterns.append(('L', 'H', 'L'))
    patterns.append(('H', 'L', 'H'))
    patterns.append(('L', 'L', 'H'))
    patterns.append(('H', 'H', 'L'))
    patterns.append(('L', 'H', 'H'))
    patterns.append(('H', 'L', 'L'))
    
    # Four layers
    patterns.append(('L', 'H', 'L', 'H'))
    patterns.append(('H', 'L', 'H', 'L'))
    patterns.append(('L', 'H', 'H', 'L'))
    patterns.append(('H', 'L', 'L', 'H'))
    
    # Five layers
    patterns.append(('L', 'H', 'L', 'H', 'L'))
    patterns.append(('H', 'L', 'H', 'L', 'H'))
    patterns.append(('L', 'H', 'H', 'L', 'H'))
    patterns.append(('H', 'L', 'L', 'H', 'L'))
    
    # Six layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L'))
    
    # Seven layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H'))
    
    # Eight layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'))
    
    # Nine layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'))
    
    # Ten layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'))
    
    # Eleven layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'))
    
    # Twelve layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'))
    
    # Thirteen layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'))
    
    # Fourteen layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'))
    
    # Fifteen layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'))
    
    # Sixteen layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'))
    
    # Seventeen layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'))
    
    # Eighteen layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'))
    
    # Nineteen layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'))
    
    # Twenty layers
    patterns.append(('L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'))
    patterns.append(('H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'))
    
    # Some non-alternating patterns for variety
    patterns.append(('L', 'L', 'H', 'H'))
    patterns.append(('H', 'H', 'L', 'L'))
    patterns.append(('L', 'H', 'H', 'L'))
    patterns.append(('H', 'L', 'L', 'H'))
    patterns.append(('L', 'L', 'H', 'L', 'H'))
    patterns.append(('H', 'H', 'L', 'H', 'L'))
    patterns.append(('L', 'H', 'L', 'L', 'H'))
    patterns.append(('H', 'L', 'H', 'H', 'L'))
    
    # Sort patterns by number of layers (fewer first for quick wins)
    patterns.sort(key=len)
    
    # Reference wavelengths for initial estimates
    wl_refs = [450, 500, 550, 600, 650, 700]
    
    np.random.seed(42)
    
    # For each pattern, try a few initial conditions and optimize
    for pattern in patterns:
        if time.time() - start_time > deadline:
            break
        
        n_layers = len(pattern)
        if n_layers > max_layers:
            continue
        
        # Determine bounds
        bounds = []
        for m in pattern:
            n = nL if m == 'L' else nH
            # Minimum reasonable thickness
            d_min = 1.0
            # Maximum: constrained by max_layer_nm and total budget
            d_max = min(max_layer_nm, max_total_nm / n_layers)
            bounds.append((d_min, d_max))
        
        # Try multiple reference wavelengths
        for wl_ref in wl_refs:
            if time.time() - start_time > deadline:
                break
            
            # Initial guess: quarter-wave at reference wavelength
            x0 = make_initial_thicknesses(pattern, wl_ref)
            
            # Check if valid
            if not _is_valid(list(pattern), x0, problem):
                # Scale down to fit
                total = sum(x0)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    x0 = [t * scale for t in x0]
                # Clip to bounds
                x0 = [min(max(t, b[0]), b[1]) for t, b in zip(x0, bounds)]
            
            if not _is_valid(list(pattern), x0, problem):
                continue
            
            # Evaluate initial
            f0 = objective(x0, list(pattern))
            if f0 < best_score:
                best_score = f0
                best_materials = list(pattern)
                best_thicknesses = x0[:]
            
            # Run Nelder-Mead optimization
            if time.time() - start_time > deadline - 5:
                continue
            
            def obj_func(x):
                x_clipped = np.clip(x, bounds, bounds)
                return objective(list(x_clipped), list(pattern))
            
            try:
                x_opt, f_opt = _nelder_mead(obj_func, x0, bounds, max_iter=100, tol=1e-5)
            except Exception:
                continue
            
            if f_opt < best_score:
                best_score = f_opt
                best_materials = list(pattern)
                best_thicknesses = list(x_opt)
        
        # Also try a random perturbation of the best found for this pattern
        if time.time() - start_time > deadline - 5:
            continue
        
        if _is_valid(best_materials, best_thicknesses, problem) and len(best_materials) == n_layers:
            x0 = best_thicknesses[:]
            for _ in range(3):
                x_perturbed = [t * np.random.uniform(0.7, 1.3) for t in x0]
                x_perturbed = [min(max(t, b[0]), b[1]) for t, b in zip(x_perturbed, bounds)]
                if not _is_valid(list(pattern), x_perturbed, problem):
                    continue
                
                f_pert = objective(x_perturbed, list(pattern))
                if f_pert < best_score:
                    best_score = f_pert
                    best_materials = list(pattern)
                    best_thicknesses = x_perturbed[:]
                
                if time.time() - start_time > deadline - 5:
                    break
                
                def obj_func2(x):
                    x_clipped = np.clip(x, bounds, bounds)
                    return objective(list(x_clipped), list(pattern))
                
                try:
                    x_opt, f_opt = _nelder_mead(obj_func2, x_perturbed, bounds, max_iter=80, tol=1e-5)
                except Exception:
                    continue
                
                if f_opt < best_score:
                    best_score = f_opt
                    best_materials = list(pattern)
                    best_thicknesses = list(x_opt)
    
    # Final polish: run Nelder-Mead on the best solution with tighter tolerance
    if time.time() - start_time < deadline - 3:
        n_layers = len(best_materials)
        bounds = []
        for m in best_materials:
            d_min = 1.0
            d_max = min(max_layer_nm, max_total_nm / n_layers)
            bounds.append((d_min, d_max))
        
        def obj_final(x):
            x_clipped = np.clip(x, bounds, bounds)
            return objective(list(x_clipped), best_materials)
        
        try:
            x_final, f_final = _nelder_mead(obj_final, best_thicknesses, bounds, max_iter=200, tol=1e-7)
            if f_final < best_score:
                best_score = f_final
                best_thicknesses = list(x_final)
        except Exception:
            pass
    
    # Ensure final result is valid
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Fallback to a simple valid solution
        best_materials = ["L"]
        best_thicknesses = [wl_center / (4.0 * nL)]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
