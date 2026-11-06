"""Improved program using Global Optimization for broadband antireflection coating."""
import time
import numpy as np
from scipy.optimize import dual_annealing

# EVOLVE-BLOCK-START

def _mean_reflectance_vec(thicknesses, materials, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """
    Compute mean reflectance in percent over the band, vectorized over wavelengths.
    thicknesses: array of physical thicknesses in nm.
    materials: list of 'L'/'H'.
    wls: array of wavelengths in nm.
    """
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    n_layers = len(materials)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return 100.0 * r * r

    # Precompute indices
    n_vals = np.array([materials_dict[m] for m in materials])
    
    # Vectorize over wavelengths
    # thicknesses shape: (n_layers,)
    # wls shape: (n_points,)
    # We need to compute beta for each layer and wavelength
    # beta[i, j] = 2*pi * n[i] * d[i] / wls[j]
    
    # Reshape for broadcasting: n_vals (n_layers, 1), thicknesses (n_layers, 1), wls (1, n_points)
    n_col = n_vals.reshape(-1, 1)
    d_col = np.asarray(thicknesses).reshape(-1, 1)
    w_row = wls.reshape(1, -1)
    
    # Avoid division by zero (though wls > 0)
    beta = 2.0 * np.pi * n_col * d_col / w_row
    
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    
    # Characteristic matrices for each layer and wavelength
    # M[i, j] is a 2x2 matrix, but we can store elements in arrays
    # M11 = cos_b
    # M12 = 1j * sin_b / n
    # M21 = 1j * n * sin_b
    # M22 = cos_b
    
    # We need to multiply these matrices sequentially.
    # Total matrix T = M_1 @ M_2 @ ... @ M_N
    # We can do this iteratively for each wavelength, but that's a loop.
    # Alternatively, we can use the fact that we just need the final admittance.
    # Let's stick to a loop over layers but vectorized over wavelengths.
    
    # Initialize total matrix elements for all wavelengths
    # T11, T12, T21, T22 shape: (n_points,)
    T11 = np.ones_like(wls, dtype=complex)
    T12 = np.zeros_like(wls, dtype=complex)
    T21 = np.zeros_like(wls, dtype=complex)
    T22 = np.ones_like(wls, dtype=complex)
    
    for i in range(n_layers):
        n_i = n_vals[i]
        cos_i = cos_b[i]
        sin_i = sin_b[i]
        
        m11 = cos_i
        m12 = 1.0j * sin_i / n_i
        m21 = 1.0j * n_i * sin_i
        m22 = cos_i
        
        # New T = T @ M
        # T11' = T11*m11 + T12*m21
        # T12' = T11*m12 + T12*m22
        # T21' = T21*m11 + T22*m21
        # T22' = T21*m12 + T22*m22
        
        new_T11 = T11 * m11 + T12 * m21
        new_T12 = T11 * m12 + T12 * m22
        new_T21 = T21 * m11 + T22 * m21
        new_T22 = T21 * m12 + T22 * m22
        
        T11 = new_T11
        T12 = new_T12
        T21 = new_T21
        T22 = new_T22
        
    # Calculate admittance Y
    # Y = (T21 + T22 * Y_s) / (T11 + T12 * Y_s)
    Y_s = substrate_index
    Y = (T21 + T22 * Y_s) / (T11 + T12 * Y_s)
    
    # Reflectance
    Y_0 = ambient_index
    r = (Y_0 - Y) / (Y_0 + Y)
    refl = np.abs(r) ** 2
    
    return 100.0 * np.mean(refl)

def design(problem):
    """
    Design a stack of at most twenty low- and high-index layers.
    Uses global optimization to find thicknesses for a fixed layer pattern.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    # We will try a few fixed patterns (sequences of L and H) and optimize thicknesses for each.
    # Pattern 1: Alternating L/H, starting with L (standard QWOT-like structure)
    # Pattern 2: Alternating H/L
    # Pattern 3: All L
    # Pattern 4: All H
    # Pattern 5: Mixed
    
    # Let's focus on a good number of layers, e.g., 4 to 8 layers.
    # More layers allow better broadband performance.
    # Let's try N=6 and N=8 layers.
    
    best_score = 100.0  # Initial worst case
    best_materials = []
    best_thicknesses = []
    
    # Define candidate patterns
    patterns = []
    for N in [4, 6, 8, 10]:
        # Alternating starting with L
        patterns.append(['L' if i % 2 == 0 else 'H' for i in range(N)])
        # Alternating starting with H
        patterns.append(['H' if i % 2 == 0 else 'L' for i in range(N)])
        # Symmetric L H ... H L
        if N >= 2:
            sym = ['L'] + ['H'] * (N-2) + ['L']
            patterns.append(sym)
            
    # Remove duplicates
    unique_patterns = []
    seen = set()
    for p in patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
            
    # For each pattern, run dual_annealing
    for pattern in unique_patterns:
        if time.time() - start_time > deadline - 5:
            break
            
        N = len(pattern)
        # Bounds for each thickness: [0, max_layer_nm]
        # Also need to ensure sum <= max_total_nm.
        # Dual annealing handles bounds well. The sum constraint is harder.
        # We can add a penalty to the objective if sum > max_total_nm.
        
        def objective(x):
            # x is array of thicknesses
            if np.any(x < 0):
                return 1000.0
            if np.sum(x) > max_total_nm:
                return 1000.0 + (np.sum(x) - max_total_nm)
            
            # Check individual layer max
            if np.any(x > max_layer_nm):
                return 1000.0
            
            score = _mean_reflectance_vec(x, pattern, wls, n_amb, n_sub, materials_dict)
            return score
        
        # Initial guess: quarter wave at center wavelength
        wl_center = 0.5 * (band_nm[0] + band_nm[1])
        x0 = []
        for m in pattern:
            n = nL if m == 'L' else nH
            x0.append(wl_center / (4.0 * n))
        x0 = np.array(x0)
        
        # Ensure x0 is within bounds
        x0 = np.clip(x0, 0, max_layer_nm)
        
        bounds = [(0, max_layer_nm) for _ in range(N)]
        
        try:
            res = dual_annealing(objective, bounds, maxiter=50, seed=42, 
                                  initial_temp=5230.0, restart_temp_ratio=2e-5, 
                                  visit=2.62, accept=0.01, maxfun=1000)
            
            if res.success or res.fun < best_score:
                # Verify validity
                x_opt = res.x
                if np.all(x_opt >= 0) and np.all(x_opt <= max_layer_nm) and np.sum(x_opt) <= max_total_nm:
                    score = res.fun
                    if score < best_score:
                        best_score = score
                        best_materials = pattern[:]
                        best_thicknesses = list(x_opt)
                        
        except Exception:
            continue
            
    # If no valid solution found, fallback to a simple single layer
    if not best_materials:
        best_materials = ['L']
        best_thicknesses = [wl_center / (4.0 * nL)]
        
    # Final local refinement using Nelder-Mead or similar on the best solution found
    if time.time() - start_time < deadline - 5 and best_materials:
        from scipy.optimize import minimize
        
        def objective_refine(x):
            if np.any(x < 0) or np.any(x > max_layer_nm) or np.sum(x) > max_total_nm:
                return 1000.0
            return _mean_reflectance_vec(x, best_materials, wls, n_amb, n_sub, materials_dict)
            
        x0_refine = np.array(best_thicknesses)
        bounds_refine = [(0, max_layer_nm) for _ in range(len(best_materials))]
        
        try:
            res_refine = minimize(objective_refine, x0_refine, method='Nelder-Mead', 
                                   options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 1000, 'maxfev': 1000})
            if res_refine.fun < best_score:
                x_opt = res_refine.x
                if np.all(x_opt >= 0) and np.all(x_opt <= max_layer_nm) and np.sum(x_opt) <= max_total_nm:
                    best_score = res_refine.fun
                    best_thicknesses = list(x_opt)
        except Exception:
            pass

    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
