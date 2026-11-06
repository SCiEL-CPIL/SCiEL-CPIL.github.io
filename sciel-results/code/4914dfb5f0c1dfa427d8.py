"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_reflectance(wavelengths, materials, thicknesses, ambient_index, substrate_index):
    """
    Compute mean reflectance in percent for a given stack.
    wavelengths: array of wavelengths in nm
    materials: list of 'L' or 'H'
    thicknesses: list of thicknesses in nm
    """
    if len(materials) == 0:
        # Bare substrate
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return (r**2) * 100.0
    
    n_l = len(materials)
    # Precompute indices and optical path lengths
    # delta = 2 * pi * n * d / lambda
    # We need to handle the matrix multiplication efficiently.
    
    # Vectorized computation for all wavelengths at once is tricky with matrix chains
    # because the matrices depend on wavelength.
    # However, for small n_l (<=20) and 401 wavelengths, a loop over wavelengths is fine.
    
    mean_r = 0.0
    n_waves = len(wavelengths)
    
    for i in range(n_waves):
        lam = wavelengths[i]
        
        # Build matrices for this wavelength
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
        # Note: Using complex numbers.
        
        M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_l):
            n_j = problem["materials"][materials[j]]
            d_j = thicknesses[j]
            delta_j = 2.0 * np.pi * n_j * d_j / lam
            
            c_j = np.cos(delta_j)
            s_j = np.sin(delta_j)
            
            # Matrix M_j
            m11 = c_j
            m12 = 1j * s_j / n_j
            m21 = 1j * n_j * s_j
            m22 = c_j
            
            M_j = np.array([[m11, m12], [m21, m22]], dtype=complex)
            
            M_total = M_total @ M_j
        
        # Substrate matrix
        # The characteristic matrix method usually includes the substrate as the final layer
        # or handles the boundary condition at the substrate.
        # Standard formula:
        # Y = (M11 * n_sub + M21) / (M12 * n_sub + M22) ? No.
        # Let's use the standard relation:
        # The admittance Y at the incident side is given by:
        # Y = (B / D) where [B, C; D, E] is the total matrix.
        # Actually, for normal incidence:
        # [E]   [M11 M12] [E_sub]
        # [H] = [M21 M22] [H_sub]
        # H_sub = E_sub / n_sub (for TE) or n_sub * E_sub (for TM)?
        # At normal incidence, H = E/n for the field ratio in the medium.
        # So H_sub = E_sub / n_sub.
        # Thus:
        # E = M11 * E_sub + M12 * (E_sub / n_sub) = E_sub * (M11 + M12/n_sub)
        # H = M21 * E_sub + M22 * (E_sub / n_sub) = E_sub * (M21 + M22/n_sub)
        # Y = H/E = (M21 + M22/n_sub) / (M11 + M12/n_sub)
        
        n_sub = substrate_index
        B = M_total[0, 0]
        C = M_total[0, 1]
        D = M_total[1, 0]
        E = M_total[1, 1]
        
        Y = (D + E / n_sub) / (B + C / n_sub)
        
        # Reflectance R = |(n_0 - Y)/(n_0 + Y)|^2
        n_0 = ambient_index
        r = (n_0 - Y) / (n_0 + Y)
        R = np.abs(r)**2
        mean_r += R
        
    return (mean_r / n_waves) * 100.0

def design(problem):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s budget
    
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials_map = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Initial guess: 4-layer quarter-wave stack HLHL
    # Center wavelength 600nm
    lam_c = 0.5 * (band_nm[0] + band_nm[1])
    n_l = materials_map['L']
    n_h = materials_map['H']
    
    d_l = lam_c / (4.0 * n_l)
    d_h = lam_c / (4.0 * n_h)
    
    best_materials = ["H", "L", "H", "L"]
    best_thicknesses = [d_h, d_l, d_h, d_l]
    best_score = _compute_reflectance(wavelengths, best_materials, best_thicknesses, ambient_index, substrate_index)
    
    # Global search over number of layers and rough thicknesses
    # We will try different numbers of layers from 1 to max_layers (capped for speed)
    # For each layer count, we can use a coarse grid or random sampling.
    # Given the time budget, let's do a structured search.
    
    # Strategy: 
    # 1. Try layer counts 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20
    # 2. For each count, generate several random candidates and evaluate.
    # 3. Keep track of best.
    
    np.random.seed(42)
    
    layer_counts_to_try = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
    
    # Pre-compute some statistics
    # We want to distribute thicknesses such that total <= max_total_nm
    
    for n_layers in layer_counts_to_try:
        if time.time() > deadline:
            break
            
        # Number of trials for this layer count
        # More trials for smaller layer counts as they are faster to evaluate?
        # Actually, evaluation cost is proportional to n_layers * n_wavelengths.
        # n_wavelengths is fixed at 401.
        # So cost is linear in n_layers.
        # Let's do ~50 trials per layer count, adjusted by remaining time.
        
        # Estimate time per eval:
        # If we assume ~1ms per eval for small stacks, 50 evals = 50ms.
        # 20 layers = ~20ms per eval? 50 evals = 1s.
        # Total time for all counts: roughly 10-20 seconds. Safe.
        
        num_trials = 50
        
        for t in range(num_trials):
            if time.time() > deadline:
                break
                
            # Generate random materials
            mats = []
            for _ in range(n_layers):
                # Bias towards alternating? Or random?
                # Random is safer for exploration.
                if np.random.random() < 0.5:
                    mats.append('L')
                else:
                    mats.append('H')
            
            # Generate random thicknesses
            # Uniform distribution in [0, max_layer_nm]
            # But we must ensure sum <= max_total_nm
            # Simple approach: generate random thicknesses, scale if sum > max_total_nm
            # Or generate from a Dirichlet-like distribution?
            # Let's just generate uniform and clip/scale.
            
            thick = np.random.uniform(0, max_layer_nm, n_layers)
            total_thick = np.sum(thick)
            if total_thick > max_total_nm:
                thick = thick * (max_total_nm / total_thick)
            
            # Evaluate
            try:
                score = _compute_reflectance(wavelengths, mats, list(thick), ambient_index, substrate_index)
            except Exception:
                continue
            
            if score < best_score:
                best_score = score
                best_materials = mats
                best_thicknesses = list(thick)
    
    # Local refinement of the best solution found
    # Use Nelder-Mead or simple coordinate descent?
    # Nelder-Mead is robust for small dimensions.
    # Max dimensions is 20. NM might be slow.
    # Let's do a simple gradient-free local search: perturb each thickness slightly.
    
    # Actually, let's just do a few iterations of random perturbation around the best solution.
    # This is a "hill climbing" step.
    
    current_materials = best_materials
    current_thicknesses = np.array(best_thicknesses)
    current_score = best_score
    
    # Define a neighborhood search
    # For each dimension, try +/- steps
    # Since we have a deadline, we can do this for a few iterations.
    
    # Better: Use scipy.optimize.minimize if available and fast enough?
    # SciPy is available.
    
    def objective(thick_array):
        # thick_array is 1D numpy array
        # Check constraints
        if np.any(thick_array < 0) or np.any(thick_array > max_layer_nm):
            return 100.0 # Penalty
        if np.sum(thick_array) > max_total_nm:
            return 100.0
        try:
            return _compute_reflectance(wavelengths, current_materials, list(thick_array), ambient_index, substrate_index)
        except:
            return 100.0
    
    # Try to optimize thicknesses for the fixed material stack
    # We can try this for the top few best material stacks if we stored them, but we only stored one.
    # Let's just optimize the current best.
    
    if time.time() < deadline - 5.0: # Need 5s margin
        try:
            from scipy.optimize import minimize
            
            # Bounds
            bounds = [(0, max_layer_nm) for _ in range(len(current_thicknesses))]
            
            # Initial guess
            x0 = current_thicknesses.copy()
            
            # Run Nelder-Mead
            # maxiter needs to be small to fit in time
            # Each function call takes ~1-10ms.
            # Let's allow 200 iterations.
            
            res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.success or (res.fun < current_score):
                # Check if result is valid
                if np.all(res.x >= 0) and np.all(res.x <= max_layer_nm) and np.sum(res.x) <= max_total_nm:
                    new_score = res.fun
                    if new_score < current_score:
                        current_score = new_score
                        current_thicknesses = res.x
        except Exception:
            pass
    
    return current_materials, list(current_thicknesses)

# EVOLVE-BLOCK-END
