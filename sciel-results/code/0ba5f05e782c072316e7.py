"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _reflectance_spectrum(materials, thicknesses, problem, wavelengths):
    """Compute reflectance spectrum using the characteristic matrix method."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_layers = len(materials)
    
    # Precompute refractive indices for each layer
    n_vals = np.array([problem["materials"][m] for m in materials])
    d_vals = np.array(thicknesses)
    
    n_waves = len(wavelengths)
    R = np.zeros(n_waves)
    
    for i, lam in enumerate(wavelengths):
        # Build the characteristic matrix
        M = np.array([[1.0, 0.0]], dtype=complex)  # Start with identity
        
        for j in range(n_layers):
            delta = 2.0 * np.pi * n_vals[j] * d_vals[j] / lam
            cd = np.cos(delta)
            sd = np.sin(delta)
            
            if abs(sd) < 1e-12:
                inv_n_sd = 0.0
            else:
                inv_n_sd = 1.0 / (n_vals[j] * sd)
            
            B = n_vals[j] * sd
            
            layer_M = np.array([[cd, 1j * inv_n_sd], [1j * B, cd]], dtype=complex)
            M = M @ layer_M
        
        M11, M12 = M[0, 0], M[0, 1]
        M21, M22 = M[1, 0], M[1, 1]
        
        Y_in = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
        
        r = (n_amb - Y_in) / (n_amb + Y_in)
        
        R[i] = np.abs(r) ** 2 * 100.0
    
    return R

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

def _binary_search_pattern(materials, problem, wavelengths_check):
    """
    For a given set of wavelengths, determine the best binary pattern (L/H)
    that minimizes reflectance. We do this by evaluating reflectance for
    each layer position and choosing the material that gives lower local
    reflectance contribution, or more simply, we just try a few smart patterns.
    This is a simplified version of the 'continuous profile' idea where we
    discretize the index profile.
    """
    # This function is a placeholder for the structural change.
    # Instead of a full binary search over 2^N patterns, we generate
    # patterns based on the target index at specific wavelengths.
    pass

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
    # Initialize with a valid single layer to ensure consistency
    best_materials = ["L"]
    best_thicknesses = [band[0] / (4.0 * n_L)]
    
    # Precompute wavelengths for evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def _objective_constrained(params, materials):
        """Objective with constraint handling."""
        thicknesses = np.array(params)
        
        # Check constraints
        if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
            return 1e6
        if np.sum(thicknesses) > max_total_nm:
            return 1e6
        
        R = _reflectance_spectrum(materials, thicknesses, problem, wavelengths)
        return np.mean(R)
    
    # Structural Change: Generate patterns based on "effective index" matching.
    # The ideal index for a single layer is sqrt(n_amb * n_sub).
    # For multiple layers, we can approximate a graded index by alternating
    # or grouping layers.
    
    # Strategy:
    # 1. Try standard quarter-wave stacks (alternating).
    # 2. Try "binary search" style patterns where we assign L or H based on
    #    which brings the local admittance closer to n_amb at key wavelengths.
    # 3. Optimize thicknesses for the best few patterns.
    
    centre = 0.5 * (band[0] + band[1])
    
    # Generate candidate patterns
    candidate_patterns = set()
    
    # 1. Alternating patterns
    for n_layers in range(2, min(max_layers + 1, 12)):
        pattern_L = tuple(["L" if i % 2 == 0 else "H" for i in range(n_layers)])
        pattern_H = tuple(["H" if i % 2 == 0 else "L" for i in range(n_layers)])
        candidate_patterns.add(pattern_L)
        candidate_patterns.add(pattern_H)
        
        # 2. Symmetric patterns (e.g., L H H L, L L H H L L)
        if n_layers >= 4:
            # Palindromic patterns
            half = n_layers // 2
            for start_mat in ["L", "H"]:
                # Simple symmetric: first half alternating, second half mirrored
                first_half = [start_mat if i % 2 == 0 else ("H" if start_mat == "L" else "L") for i in range(half)]
                if n_layers % 2 == 0:
                    second_half = first_half[::-1]
                else:
                    second_half = first_half[::-1] + [start_mat]
                pattern_sym = tuple(first_half + second_half)
                candidate_patterns.add(pattern_sym)
        
        # 3. "Binary Search" Pattern:
        # At the center wavelength, determine the target admittance.
        # We can create patterns that try to "step" the index from n_amb to n_sub.
        # For example, if we use 4 layers, we might use L, L, H, H or L, H, H, L.
        # Let's add a few heuristic patterns.
        if n_layers == 4:
            candidate_patterns.add(("L", "L", "H", "H"))
            candidate_patterns.add(("H", "H", "L", "L"))
            candidate_patterns.add(("L", "H", "H", "L"))
            candidate_patterns.add(("H", "L", "L", "H"))
        if n_layers == 6:
            candidate_patterns.add(("L", "L", "L", "H", "H", "H"))
            candidate_patterns.add(("H", "H", "H", "L", "L", "L"))
            candidate_patterns.add(("L", "H", "L", "H", "L", "H"))
            candidate_patterns.add(("H", "L", "H", "L", "H", "L"))
    
    # Sort patterns by length to try shorter ones first (faster to evaluate)
    patterns_list = sorted(list(candidate_patterns), key=lambda x: len(x))
    
    # Optimize thicknesses for each pattern
    for materials_tuple in patterns_list:
        if time.time() > deadline:
            break
        
        materials = list(materials_tuple)
        n_layers = len(materials)
        
        # Initial guess: quarter-wave thicknesses at center wavelength
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
            # Create a closure for the objective with fixed materials
            def _obj(params):
                return _objective_constrained(params, materials)
            
            result_x, result_f = _nelder_mead(
                _obj,
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
        
        # If we have time, we could also try a few random restarts for the best patterns
        # but for now, we stick to one optimization per pattern to stay within time.
    
    # Ensure final result is valid and consistent
    if len(best_materials) != len(best_thicknesses):
        # Fallback to the initial valid single layer if mismatch occurs
        best_materials = ["L"]
        best_thicknesses = [band[0] / (4.0 * n_L)]
        
    if not all(0 <= t <= max_layer_nm for t in best_thicknesses):
        best_thicknesses = [min(max(t, 0), max_layer_nm) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
