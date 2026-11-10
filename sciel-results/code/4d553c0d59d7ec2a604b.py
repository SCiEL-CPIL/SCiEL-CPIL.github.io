"""Combined design program: Expanded analytical candidates + Vectorized Nelder-Mead refinement."""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband AR coating using analytical stacks + Nelder-Mead refinement."""
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation (401 points from 400 to 800 nm)
    lambdas = np.linspace(band[0], band[1], 401)
    lambda_c = 0.5 * (band[0] + band[1])
    
    def mean_reflectance(materials, thicknesses):
        """
        Compute mean reflectance in percent over the band using characteristic matrix method.
        Vectorized over wavelengths for speed.
        """
        n_layers = [nL if m == 'L' else nH for m in materials]
        n_count = len(materials)
        
        if n_count == 0:
            r = (n0 - ns) / (n0 + ns)
            return (np.abs(r) ** 2) * 100.0
        
        # Pre-compute layer parameters
        # thicknesses is a list of floats
        # We need to compute matrices for each wavelength
        # M_total = M1 @ M2 ... @ Mn
        # For each wavelength wl, M_i depends on wl
        
        # To vectorize, we can compute the matrix for each layer for all wavelengths at once.
        # However, matrix multiplication is sequential across layers.
        # Let's iterate layers, updating a vector of matrices (one per wavelength).
        
        # Shape: (401, 2, 2)
        M_total = np.zeros((len(lambdas), 2, 2), dtype=complex)
        M_total[:, 0, 0] = 1.0
        M_total[:, 1, 1] = 1.0
        
        for i in range(n_count):
            ni = n_layers[i]
            di = thicknesses[i]
            
            # delta = 2 * pi * ni * di / wl
            delta = 2.0 * np.pi * ni * di / lambdas
            c = np.cos(delta)
            s = np.sin(delta)
            
            # M_layer shape (401, 2, 2)
            M_layer = np.zeros((len(lambdas), 2, 2), dtype=complex)
            M_layer[:, 0, 0] = c
            M_layer[:, 0, 1] = 1j * s / ni
            M_layer[:, 1, 0] = 1j * ni * s
            M_layer[:, 1, 1] = c
            
            # M_total = M_total @ M_layer
            # Batch matrix multiplication
            M_total = M_total @ M_layer
            
        A = M_total[:, 0, 0]
        B = M_total[:, 0, 1]
        C = M_total[:, 1, 0]
        D = M_total[:, 1, 1]
        
        # Admittance y_in = (C + D*ns) / (A + B*ns)
        y_in = (C + D * ns) / (A + B * ns)
        r = (n0 - y_in) / (n0 + y_in)
        reflectance = np.abs(r) ** 2
        
        return np.mean(reflectance) * 100.0

    def is_valid(materials, thicknesses):
        """Check constraints."""
        if len(materials) != len(thicknesses):
            return False
        if len(materials) > max_layers:
            return False
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm + 1e-9:
            return False
        return True

    # --- Generate candidate structures ---
    # We generate a diverse set of topologies.
    # Quarter-wave thicknesses are used as initial guesses.
    
    candidates = []
    
    # Helper to add candidate if valid
    def add_candidate(mat_str):
        materials = list(mat_str)
        if len(materials) > max_layers:
            return
        # Check if we can fit quarter-wave thicknesses within limits
        # d = lambda_c / (4 * n)
        # Sum d < max_total_nm?
        total_qw = 0
        for m in materials:
            n = nL if m == 'L' else nH
            total_qw += lambda_c / (4.0 * n)
        if total_qw <= max_total_nm * 1.5: # Allow some room for optimization
            thicknesses = [lambda_c / (4.0 * (nL if m == 'L' else nH)) for m in materials]
            if is_valid(materials, thicknesses):
                candidates.append((materials, thicknesses))
            else:
                # Scale down thicknesses to fit
                scale = max_total_nm / total_qw * 0.9
                thicknesses = [t * scale for t in thicknesses]
                if is_valid(materials, thicknesses):
                    candidates.append((materials, thicknesses))

    # Single layer
    add_candidate("L")
    add_candidate("H")
    
    # Two layers
    add_candidate("LH")
    add_candidate("HL")
    
    # Three layers
    add_candidate("LHL")
    add_candidate("HLH")
    add_candidate("LLH")
    add_candidate("HHL")
    
    # Four layers
    add_candidate("LHHL")
    add_candidate("HLHL")
    add_candidate("LHLH")
    add_candidate("HLHL")
    
    # Five layers
    add_candidate("LHLHL")
    add_candidate("HLHLH")
    add_candidate("LHLLH")
    add_candidate("HLHHL")
    
    # Six layers
    add_candidate("LHLHHL")
    add_candidate("HLHLHH")
    add_candidate("LHLHLH")
    add_candidate("HLHLHL")
    
    # Seven layers
    add_candidate("LHLHLHL")
    add_candidate("HLHLHLH")
    
    # Eight layers
    add_candidate("LHLHLHHL")
    add_candidate("HLHLHLHH")
    
    # Nine layers
    add_candidate("LHLHLHLHL")
    add_candidate("HLHLHLHLH")
    
    # Ten layers
    add_candidate("LHLHLHLHHL")
    add_candidate("HLHLHLHLHH")
    
    # Eleven layers
    add_candidate("LHLHLHLHLHL")
    add_candidate("HLHLHLHLHLH")
    
    # Twelve layers
    add_candidate("LHLHLHLHLHHL")
    add_candidate("HLHLHLHLHLHH")
    
    # Thirteen layers
    add_candidate("LHLHLHLHLHLHL")
    add_candidate("HLHLHLHLHLHLH")
    
    # Fourteen layers
    add_candidate("LHLHLHLHLHLHHL")
    add_candidate("HLHLHLHLHLHLHH")
    
    # Fifteen layers
    add_candidate("LHLHLHLHLHLHLHL")
    add_candidate("HLHLHLHLHLHLHLH")
    
    # Sixteen layers
    add_candidate("LHLHLHLHLHLHLHHL")
    add_candidate("HLHLHLHLHLHLHLHH")
    
    # Seventeen layers
    add_candidate("LHLHLHLHLHLHLHLHL")
    add_candidate("HLHLHLHLHLHLHLHLH")
    
    # Eighteen layers
    add_candidate("LHLHLHLHLHLHLHLHHL")
    add_candidate("HLHLHLHLHLHLHLHLHH")
    
    # Nineteen layers
    add_candidate("LHLHLHLHLHLHLHLHLHL")
    add_candidate("HLHLHLHLHLHLHLHLHLH")
    
    # Twenty layers
    add_candidate("LHLHLHLHLHLHLHLHLHHL")
    add_candidate("HLHLHLHLHLHLHLHLHLHH")

    if not candidates:
        # Fallback
        return ["L"], [lambda_c / (4.0 * nL)]

    # Evaluate all candidates to find the best starting point
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    for materials, thicknesses in candidates:
        score = mean_reflectance(materials, thicknesses)
        if score < best_score:
            best_score = score
            best_materials = materials
            best_thicknesses = list(thicknesses)
            
    # --- Local Refinement using Nelder-Mead ---
    # We will refine the top few candidates to ensure we don't get stuck in a poor local minimum.
    # Sort candidates by score
    scored_candidates = []
    for materials, thicknesses in candidates:
        score = mean_reflectance(materials, thicknesses)
        scored_candidates.append((score, materials, thicknesses))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top 5 candidates
    top_k = min(5, len(scored_candidates))
    
    for i in range(top_k):
        if (time.time() - start_time) > DEADLINE:
            break
            
        score, materials, thicknesses = scored_candidates[i]
        if score >= best_score:
            break # No point refining worse candidates if we have a good best
        
        # Define objective function for scipy
        def objective(x):
            # x is array of thicknesses
            # Check constraints inside objective to penalize invalid solutions
            if any(xi < 0 or xi > max_layer_nm for xi in x):
                return 1e6
            if sum(x) > max_total_nm + 1e-9:
                return 1e6
            return mean_reflectance(materials, list(x))
        
        # Initial guess
        x0 = np.array(thicknesses)
        
        # Bounds for Nelder-Mead (it doesn't natively support bounds well, 
        # so we rely on the penalty function and careful initial simplex)
        # We can use scipy.optimize.minimize with method='Nelder-Mead'
        
        try:
            result = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 0.01, 'fatol': 1e-6}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_materials = materials
                best_thicknesses = list(result.x)
                
        except Exception:
            pass
            
    # Final validation
    if not is_valid(best_materials, best_thicknesses):
        # Fallback to the best analytical candidate
        return scored_candidates[0][1], scored_candidates[0][2]
        
    # Ensure thicknesses are strictly within bounds
    final_thicknesses = []
    for t in best_thicknesses:
        if t < 0: t = 0.0
        if t > max_layer_nm: t = max_layer_nm
        final_thicknesses.append(t)
        
    if sum(final_thicknesses) > max_total_nm:
        # Scale down
        scale = max_total_nm / sum(final_thicknesses)
        final_thicknesses = [t * scale for t in final_thicknesses]
        
    return best_materials, final_thicknesses
# EVOLVE-BLOCK-END
