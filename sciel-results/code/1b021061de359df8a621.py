import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _reflectance_spectrum(materials, thicknesses, problem, wavelengths):
    """Compute reflectance spectrum using the characteristic matrix method."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_layers = len(materials)
    
    if n_layers == 0:
        # Bare substrate
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.full(len(wavelengths), np.abs(r)**2 * 100.0)
        
    n_vals = np.array([problem["materials"][m] for m in materials])
    d_vals = np.array(thicknesses)
    
    n_waves = len(wavelengths)
    R = np.zeros(n_waves)
    
    # Precompute layer matrices for each wavelength is not straightforward 
    # because delta depends on wavelength.
    # We iterate over wavelengths.
    
    for i, lam in enumerate(wavelengths):
        M = np.array([[1.0, 0.0]], dtype=complex)
        
        for j in range(n_layers):
            delta = 2.0 * np.pi * n_vals[j] * d_vals[j] / lam
            cd = np.cos(delta)
            sd = np.sin(delta)
            
            # Handle sin(delta) near 0
            if abs(sd) < 1e-12:
                inv_n_sd = 0.0
            else:
                inv_n_sd = 1.0 / (n_vals[j] * sd)
            
            B = n_vals[j] * sd
            
            layer_M = np.array([[cd, 1j * inv_n_sd], [1j * B, cd]], dtype=complex)
            M = M @ layer_M
        
        M11, M12 = M[0, 0], M[0, 1]
        M21, M22 = M[1, 0], M[1, 1]
        
        # Input admittance Y_in
        Y_in = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
        
        # Reflection coefficient
        r = (n_amb - Y_in) / (n_amb + Y_in)
        
        R[i] = np.abs(r) ** 2 * 100.0
    
    return R

def _objective(params, materials, problem, wavelengths, max_layer_nm, max_total_nm):
    """Objective function for the optimizer."""
    thicknesses = np.array(params)
    
    # Constraint check
    if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
        return 1e6
    if np.sum(thicknesses) > max_total_nm:
        return 1e6
        
    R = _reflectance_spectrum(materials, thicknesses, problem, wavelengths)
    return np.mean(R)

def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    
    # Wavelengths for evaluation (401 points)
    wavelengths = np.linspace(band[0], band[1], 401)
    
    best_score = 1e6
    best_materials = ["L"]
    best_thicknesses = [600 / (4.0 * n_L)] # Initial guess: 600nm center quarter wave L
    
    # Helper to create alternating pattern
    def get_pattern(n_layers):
        # Start with L (lower index) to match air better
        return ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        
    # Try different numbers of layers
    # Even numbers of layers often provide good broadband performance
    # We try up to 10 or 12 layers to stay within time and complexity limits
    layer_counts = [2, 4, 6, 8, 10, 12]
    
    for n_layers in layer_counts:
        if time.time() > deadline - 10.0: # Leave 10s for final checks
            break
            
        materials = get_pattern(n_layers)
        
        # Bounds for DE
        # Each layer thickness between 0 and max_layer_nm
        # But we also have a total sum constraint.
        # DE handles bounds easily. We penalize total sum in objective.
        bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
        
        # Initial population or just let DE decide
        # To speed up, we can provide a good initial guess if we had one, 
        # but DE is robust.
        
        # We want to limit the time per DE run.
        # maxiter is a proxy. Let's estimate.
        # If n_layers=10, 10D space.
        # Let's use a moderate population size and maxiter.
        # We will check time after DE.
        
        try:
            # Use DE for global search
            # tol and maxiter control convergence.
            # We'll use a tight maxiter to ensure we don't hang.
            de_result = differential_evolution(
                lambda x: _objective(x, materials, problem, wavelengths, max_layer_nm, max_total_nm),
                bounds,
                maxiter=50, # Limit iterations to control time
                popsize=10,  # Smaller population for speed
                tol=1e-4,
                seed=42,
                init='sobol' # Sobol sampling is more uniform
            )
            
            best_params = de_result.x
            best_val = de_result.fun
            
            # Polish with Nelder-Mead
            if time.time() < deadline - 5.0:
                nm_result = minimize(
                    lambda x: _objective(x, materials, problem, wavelengths, max_layer_nm, max_total_nm),
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6}
                )
                
                if nm_result.fun < best_val:
                    best_params = nm_result.x
                    best_val = nm_result.fun
            
            # Check if this is the best so far
            if best_val < best_score:
                # Ensure constraints are strictly met for the final answer
                # DE might have found a point slightly outside due to floating point
                # or the penalty was soft. We clip and re-evaluate.
                clean_params = np.clip(best_params, 0.0, max_layer_nm)
                if np.sum(clean_params) > max_total_nm:
                    # Scale down if necessary
                    scale = max_total_nm / np.sum(clean_params)
                    clean_params = clean_params * scale
                    clean_params = np.clip(clean_params, 0.0, max_layer_nm)
                
                final_score = _objective(clean_params, materials, problem, wavelengths, max_layer_nm, max_total_nm)
                
                if final_score < best_score:
                    best_score = final_score
                    best_materials = materials
                    best_thicknesses = clean_params.tolist()
                    
        except Exception:
            continue

    return best_materials, best_thicknesses
