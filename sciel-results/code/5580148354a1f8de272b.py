"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _compute_reflectance_vectorized(materials, thicknesses, n_air, n_sub, wavelengths):
    """
    Compute reflectance for a stack across all wavelengths in parallel.
    materials: list of 'L' or 'H'
    thicknesses: list of physical thicknesses in nm
    n_air: ambient index
    n_sub: substrate index
    wavelengths: array of wavelengths in nm
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_air - n_sub) / (n_air + n_sub)
        return np.full_like(wavelengths, abs(r)**2)
    
    # M_total is 2x2 matrix for each wavelength
    # Initialize as identity
    M_total = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M_total[:, 0, 0] = 1.0
    M_total[:, 1, 1] = 1.0
    
    # Pre-compute indices for speed
    n_vals = np.array([problem_global["materials"][m] for m in materials])
    
    # Iterate through layers
    for i in range(n_layers):
        n = n_vals[i]
        d = thicknesses[i]
        
        # delta = 2 * pi * n * d / lambda
        # wavelengths is shape (401,)
        delta = 2.0 * np.pi * n * d / wavelengths
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # M_i for each wavelength
        # M_i = [[cos, i*sin/n], [i*n*sin, cos]]
        M_i = np.zeros((len(wavelengths), 2, 2), dtype=complex)
        M_i[:, 0, 0] = cos_d
        M_i[:, 0, 1] = 1j * sin_d / n
        M_i[:, 1, 0] = 1j * n * sin_d
        M_i[:, 1, 1] = cos_d
        
        # M_total = M_total @ M_i
        M_total = np.einsum('nij,njk->nik', M_total, M_i)
        
    # Calculate Y_in
    m11 = M_total[:, 0, 0]
    m12 = M_total[:, 0, 1]
    m21 = M_total[:, 1, 0]
    m22 = M_total[:, 1, 1]
    
    Y_in = (m21 + m22 * n_sub) / (m11 + m12 * n_sub)
    
    # r = (n_air - Y_in) / (n_air + Y_in)
    r = (n_air - Y_in) / (n_air + Y_in)
    
    R = np.abs(r)**2
    return R

problem_global = None

def design(problem):
    global problem_global
    problem_global = problem
    
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    start_time = time.time()
    time_budget = 55.0 # Leave 5s margin
    
    best_fitness = float('inf')
    best_mats = []
    best_thicks = []
    
    # Objective function for scipy.optimize
    # x: array of thicknesses
    # mats: fixed list of materials for this optimization run
    def objective(x, mats):
        # Check constraints: total thickness
        total_t = np.sum(x)
        if total_t > max_total_nm:
            # Penalize heavily if constraint violated, or just project?
            # Better to project to feasible space or use bounds?
            # L-BFGS-B supports bounds. We can't easily bound the sum in simple bounds.
            # We can penalize the objective.
            penalty = 100.0 * (total_t - max_total_nm)
        else:
            penalty = 0.0
        
        # Check individual layer bounds (handled by scipy bounds, but just in case)
        if np.any(x < 0) or np.any(x > max_layer_nm):
             penalty += 1000.0
        
        # If layers are effectively zero, we can skip calculation for speed?
        # No, just calculate.
        R = _compute_reflectance_vectorized(mats, x, n_air, n_sub, wavelengths)
        return np.mean(R) * 100.0 + penalty

    # Generate candidate material patterns
    patterns = []
    
    # 1. Alternating patterns of various lengths
    for length in range(1, 9):
        for start in ['L', 'H']:
            pat = []
            for i in range(length):
                if i % 2 == 0:
                    pat.append(start)
                else:
                    pat.append('H' if start == 'L' else 'L')
            patterns.append(pat)
            
    # 2. Some random patterns
    random.seed(42)
    for _ in range(20):
        length = random.randint(2, 10)
        pat = [random.choice(['L', 'H']) for _ in range(length)]
        patterns.append(pat)

    # Deduplicate patterns
    unique_patterns = []
    seen = set()
    for p in patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
            
    # Optimize each pattern
    for mats in unique_patterns:
        if time.time() - start_time > time_budget:
            break
            
        n_layers = len(mats)
        
        # Initial guess: quarter wave at center wavelength
        lam_c = 0.5 * (band_nm[0] + band_nm[1])
        x0 = []
        for m in mats:
            n_m = problem_global["materials"][m]
            d = lam_c / (4.0 * n_m)
            x0.append(d)
            
        # Bounds for each thickness
        bounds = [(0, max_layer_nm) for _ in range(n_layers)]
        
        # Use Nelder-Mead as it is derivative-free and robust for this size
        # Or L-BFGS-B if we had gradients, but we don't easily.
        # Nelder-Mead is good for n < 20.
        try:
            res = minimize(
                objective, 
                x0, 
                args=(mats,), 
                method='Nelder-Mead', 
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 0.1, 'fatol': 1e-5}
            )
            
            x_opt = res.x
            f_opt = res.fun
            
            # Verify constraints strictly for the final candidate
            if np.sum(x_opt) <= max_total_nm + 1e-6:
                # Re-evaluate without penalty to get true score
                R = _compute_reflectance_vectorized(mats, x_opt, n_air, n_sub, wavelengths)
                true_score = np.mean(R) * 100.0
                
                if true_score < best_fitness:
                    best_fitness = true_score
                    best_mats = mats
                    best_thicks = x_opt
                    
        except Exception:
            continue

    # If no valid solution found, fallback to simple quarter wave
    if not best_mats:
        lam_c = 0.5 * (band_nm[0] + band_nm[1])
        n_L = problem_global["materials"]["L"]
        d_L = lam_c / (4.0 * n_L)
        best_mats = ["L"]
        best_thicks = [d_L]
        best_fitness = 4.0 # Just a placeholder, we return the design
        
    # Clean up output
    final_mats = []
    final_thicks = []
    for m, t in zip(best_mats, best_thicks):
        if t > 1e-6:
            final_mats.append(m)
            final_thicks.append(float(t))
            
    if not final_mats:
        # Fallback again if all zeroed out
        lam_c = 0.5 * (band_nm[0] + band_nm[1])
        n_L = problem_global["materials"]["L"]
        d_L = lam_c / (4.0 * n_L)
        final_mats = ["L"]
        final_thicks = [d_L]
        
    return final_mats, final_thicks
# EVOLVE-BLOCK-END
