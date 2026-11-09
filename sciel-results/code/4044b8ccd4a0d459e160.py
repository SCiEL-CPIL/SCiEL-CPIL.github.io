<<<<<<< SEARCH
def _objective(params, materials, problem):
    """Objective function: mean reflectance over the band."""
    thicknesses = np.array(params)
    
    # Check constraints
    if np.any(thicknesses < 0) or np.any(thicknesses > problem["max_layer_nm"]):
        return 1e6
    if np.sum(thicknesses) > problem["max_total_nm"]:
        return 1e6
    
    wavelengths = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    R = _reflectance_spectrum(materials, thicknesses, problem, wavelengths)
    return np.mean(R)

def design(problem):
    """Design an antireflection coating using optimization."""
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 150s limit
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    
    # Try different numbers of layers from 2 to max_layers
    # Start with a good initial guess based on quarter-wave stack
    
    best_score = 1e6
    best_materials = ["L"]
    best_thicknesses = [band[0] / (4.0 * n_L)]
    
    # Precompute wavelengths for evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Try various layer counts
    for n_layers in range(2, min(max_layers + 1, 11)):  # Try up to 10 layers first
        if time.time() > deadline:
            break
        
        # Generate a few candidate material patterns
        # Pattern 1: Alternating L, H, L, H, ...
        # Pattern 2: Start with L, end with H
        # Pattern 3: Various patterns
        
        candidate_patterns = []
        
        # Alternating starting with L
        pattern1 = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        candidate_patterns.append(pattern1)
        
        # Alternating starting with H
        pattern2 = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        candidate_patterns.append(pattern2)
        
        # For larger stacks, try symmetric patterns
        if n_layers >= 4:
            # L, H, H, L, H, H, L pattern (repeated)
            base = ["L", "H", "H", "L"]
            pattern3 = []
            for i in range(n_layers):
                pattern3.append(base[i % 4])
            candidate_patterns.append(pattern3)
        
        # L, L, H, H pattern
        if n_layers >= 4:
            base = ["L", "L", "H", "H"]
            pattern4 = []
            for i in range(n_layers):
                pattern4.append(base[i % 4])
            candidate_patterns.append(pattern4)
        
        # Unique patterns
        seen = set()
        unique_patterns = []
        for p in candidate_patterns:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                unique_patterns.append(p)
        
        for materials in unique_patterns:
            if time.time() > deadline:
                break
            
            # Initial guess: quarter-wave thicknesses at center wavelength
            centre = 0.5 * (band[0] + band[1])
            x0 = []
            for m in materials:
                n_m = problem["materials"][m]
                d0 = centre / (4.0 * n_m)
                x0.append(d0)
            
            # Ensure initial guess satisfies constraints
            if sum(x0) > max_total_nm:
                scale = max_total_nm / sum(x0)
                x0 = [d * scale for d in x0]
            
            # Bounds for each thickness
            bounds = [(0.0, max_layer_nm)] * n_layers
            
            # Use differential evolution for global optimization
            try:
                result = differential_evolution(
                    _objective,
                    bounds,
                    args=(materials, problem),
                    maxiter=100,
                    popsize=10,
                    tol=1e-8,
                    seed=42,
                    polish=True
                )
                
                if result.success or result.fun < best_score:
                    thicknesses = list(result.x)
                    # Verify constraints
                    if all(0 <= t <= max_layer_nm for t in thicknesses) and sum(thicknesses) <= max_total_nm:
                        score = _objective(thicknesses, materials, problem)
                        if score < best_score:
                            best_score = score
                            best_materials = materials
                            best_thicknesses = thicknesses
            except Exception:
                pass
        
        # If we have time, try a few more layers
        if time.time() > deadline - 10:
            break
    
    # If we still have time, try to refine the best solution with Nelder-Mead
    if time.time() < deadline - 5 and best_score < 1e5:
        try:
            materials = best_materials
            x0 = best_thicknesses
            bounds = [(0.0, max_layer_nm)] * len(materials)
            
            # Constrained Nelder-Mead
            from scipy.optimize import NonlinearConstraint
            
            # Constraint: sum of thicknesses <= max_total_nm
            def sum_constraint(x):
                return max_total_nm - sum(x)
            
            constraints = [NonlinearConstraint(sum_constraint, 0, np.inf)]
            
            result = minimize(
                _objective,
                x0,
                args=(materials, problem),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result.fun < best_score:
                thicknesses = list(result.x)
                if all(0 <= t <= max_layer_nm for t in thicknesses) and sum(thicknesses) <= max_total_nm:
                    best_score = result.fun
                    best_thicknesses = thicknesses
        except Exception:
            pass
    
    # Ensure final result is valid
    if not all(0 <= t <= max_layer_nm for t in best_thicknesses):
        best_thicknesses = [min(max(t, 0), max_layer_nm) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
=======
def _nelder_mead(objective, x0, bounds, maxiter=500, tol=1e-8):
    """Simple Nelder-Mead implementation with bounds."""
    n = len(x0)
    
    # Simplex initialization
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float)
        step = 0.05 * abs(point[i]) + 0.00025
        point[i] += step
        simplex.append(point)
    
    # Evaluate initial simplex
    values = [objective(p) for p in simplex]
    
    for it in range(maxiter):
        # Sort simplex
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[0] - values[-1]) < tol:
            break
        
        # Centroid (excluding worst point)
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < values[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            # Accept reflection
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = objective(simplex[i])
    
    # Return best point
    best_idx = np.argmin(values)
    return np.array(simplex[best_idx]), values[best_idx]

def design(problem):
    """Design an antireflection coating using optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    
    best_score = 1e6
    best_materials = ["L"]
    best_thicknesses = [band[0] / (4.0 * n_L)]
    
    # Precompute wavelengths for evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def _objective_constrained(params):
        """Objective with constraint handling."""
        thicknesses = np.array(params)
        
        # Check constraints
        if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
            return 1e6
        if np.sum(thicknesses) > max_total_nm:
            return 1e6
        
        R = _reflectance_spectrum(best_materials, thicknesses, problem, wavelengths)
        return np.mean(R)
    
    # Try various layer counts
    for n_layers in range(2, min(max_layers + 1, 11)):  # Try up to 10 layers first
        if time.time() > deadline:
            break
        
        # Generate a few candidate material patterns
        candidate_patterns = []
        
        # Alternating starting with L
        pattern1 = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        candidate_patterns.append(pattern1)
        
        # Alternating starting with H
        pattern2 = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        candidate_patterns.append(pattern2)
        
        # For larger stacks, try symmetric patterns
        if n_layers >= 4:
            base = ["L", "H", "H", "L"]
            pattern3 = []
            for i in range(n_layers):
                pattern3.append(base[i % 4])
            candidate_patterns.append(pattern3)
        
        if n_layers >= 4:
            base = ["L", "L", "H", "H"]
            pattern4 = []
            for i in range(n_layers):
                pattern4.append(base[i % 4])
            candidate_patterns.append(pattern4)
        
        # Unique patterns
        seen = set()
        unique_patterns = []
        for p in candidate_patterns:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                unique_patterns.append(p)
        
        for materials in unique_patterns:
            if time.time() > deadline:
                break
            
            best_materials = materials  # Update global reference for objective
            
            # Initial guess: quarter-wave thicknesses at center wavelength
            centre = 0.5 * (band[0] + band[1])
            x0 = []
            for m in materials:
                n_m = problem["materials"][m]
                d0 = centre / (4.0 * n_m)
                x0.append(d0)
            
            # Ensure initial guess satisfies constraints
            if sum(x0) > max_total_nm:
                scale = max_total_nm / sum(x0)
                x0 = [d * scale for d in x0]
            
            # Use Nelder-Mead for local optimization
            try:
                result_x, result_f = _nelder_mead(
                    _objective_constrained,
                    x0,
                    bounds=[(0.0, max_layer_nm)] * n_layers,
                    maxiter=300
                )
                
                # Verify constraints
                if all(0 <= t <= max_layer_nm for t in result_x) and sum(result_x) <= max_total_nm:
                    if result_f < best_score:
                        best_score = result_f
                        best_materials = materials
                        best_thicknesses = list(result_x)
            except Exception:
                pass
        
        # If we have time, try a few more layers
        if time.time() > deadline - 10:
            break
    
    # Ensure final result is valid
    if not all(0 <= t <= max_layer_nm for t in best_thicknesses):
        best_thicknesses = [min(max(t, 0), max_layer_nm) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
