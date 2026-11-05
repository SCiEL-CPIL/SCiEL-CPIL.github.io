"""Broadband antireflection coating design via Multi-Start Nelder-Mead with variable structure."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _fast_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance fraction for a single design across all wavelengths.
    Uses vectorized operations over the wavelength axis.
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return float(r * r)
    
    # Pre-compute refractive indices for each layer
    n_layers_arr = np.array([n_H if m == 'H' else n_L for m in materials])
    
    # M = Identity
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for i in range(n_layers):
        n = n_layers_arr[i]
        d = thicknesses[i]
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        # M = M @ L
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Admittance: Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.real(r * r.conjugate())
    
    return float(np.mean(R))

def _evaluate_design(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm):
    """
    Evaluate a design. Returns reflectance percentage if valid, inf otherwise.
    """
    if len(thicknesses) == 0:
        return float('inf')
    
    # Check constraints
    if any(t < 0 or t > max_layer_nm for t in thicknesses):
        return float('inf')
    if sum(thicknesses) > max_total_nm:
        return float('inf')
        
    R_frac = _fast_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls)
    return R_frac * 100.0

def design(problem):
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 401 equally spaced wavelengths
    wls = np.linspace(wl_min, wl_max, 401)
    wl_center = 0.5 * (wl_min + wl_max)
    
    # Fallback solution: Single quarter-wave L layer
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    best_score = float('inf')
    best_thicknesses = [qw_L]
    best_materials = ['L']
    
    # Evaluate fallback
    score = _evaluate_design(best_thicknesses, best_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
    if score < best_score:
        best_score = score

    # Strategy: Multi-Start Nelder-Mead
    # We define a search space for a maximum number of layers (e.g., 10 or 12).
    # The optimizer can set thicknesses to 0 to effectively remove layers.
    # We also encode the material choice in the thickness sign? No, thickness must be positive.
    # Instead, we can use a separate binary vector for materials, but that makes it a mixed-integer problem.
    # Alternative: We try a few fixed patterns with variable length (pruning zeros).
    # Or, we can use a continuous variable x in [-1, 1] for each layer slot.
    # If x > 0, it's a High index layer with thickness x * scale.
    # If x < 0, it's a Low index layer with thickness -x * scale.
    # This allows the optimizer to switch materials and adjust thickness continuously.
    # Scale factor: max_layer_nm.
    
    max_search_layers = 12 # Keep it manageable for NM
    scale = max_layer_nm
    
    def objective(x):
        # x is an array of length max_search_layers, values in [-1, 1]
        thicknesses = []
        materials = []
        for val in x:
            if val > 0:
                d = val * scale
                m = 'H'
            else:
                d = -val * scale
                m = 'L'
            
            # Clamp to max_layer_nm (should be handled by bounds, but safe)
            if d > max_layer_nm:
                d = max_layer_nm
            if d < 0:
                d = 0
                
            # Prune very small thicknesses to reduce effective layers?
            # No, keep them for now, the reflectance function handles 0 thickness fine (identity matrix).
            thicknesses.append(d)
            materials.append(m)
            
        # Check total thickness constraint
        if sum(thicknesses) > max_total_nm:
            return float('inf')
            
        # Check if we have any non-zero layers
        if all(t < 1e-6 for t in thicknesses):
            return float('inf')
            
        return _fast_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls) * 100.0

    # Nelder-Mead settings
    nm_options = {
        'maxiter': 1000,
        'xatol': 1e-4,
        'fatol': 1e-6,
        'adaptive': True
    }
    
    # We will run multiple restarts with different initial guesses
    # Initial guesses:
    # 1. Random
    # 2. Based on quarter-wave approximations
    # 3. Based on known good patterns (e.g., L-H-L-H)
    
    num_restarts = 200 # We can afford many because NM is fast per step
    
    # Pre-compute some good initial structures
    init_structures = []
    
    # 1. Single layer L
    x1 = np.zeros(max_search_layers)
    x1[0] = qw_L / scale
    init_structures.append(x1)
    
    # 2. Single layer H
    x2 = np.zeros(max_search_layers)
    x2[0] = -qw_H / scale # Negative for L? Wait. My mapping: val>0 -> H, val<0 -> L.
    # Let's stick to: val > 0 => H, val < 0 => L.
    # So for L, we need negative values.
    # x2[0] = -qw_H / scale is for H? No.
    # If I want H, val > 0. If I want L, val < 0.
    # So for Single H: x[0] = qw_H / scale.
    # For Single L: x[0] = -qw_L / scale.
    
    x2[0] = qw_H / scale
    init_structures.append(x2)
    
    # 3. Two layers L-H
    x3 = np.zeros(max_search_layers)
    x3[0] = -qw_L / scale
    x3[1] = qw_H / scale
    init_structures.append(x3)
    
    # 4. Two layers H-L
    x4 = np.zeros(max_search_layers)
    x4[0] = qw_H / scale
    x4[1] = -qw_L / scale
    init_structures.append(x4)
    
    # 5. Three layers L-H-L
    x5 = np.zeros(max_search_layers)
    x5[0] = -qw_L / scale
    x5[1] = qw_H / scale
    x5[2] = -qw_L / scale
    init_structures.append(x5)
    
    # 6. Three layers H-L-H
    x6 = np.zeros(max_search_layers)
    x6[0] = qw_H / scale
    x6[1] = -qw_L / scale
    x6[2] = qw_H / scale
    init_structures.append(x6)
    
    # Add random perturbations to these
    for _ in range(5):
        for base in init_structures[:]:
            x_pert = base + np.random.uniform(-0.1, 0.1, size=max_search_layers)
            init_structures.append(x_pert)
    
    # Also add purely random starts
    for _ in range(50):
        x_rand = np.random.uniform(-1, 1, size=max_search_layers)
        init_structures.append(x_rand)
    
    # Shuffle and take up to num_restarts
    np.random.shuffle(init_structures)
    init_structures = init_structures[:num_restarts]
    
    bounds = [(-1.0, 1.0) for _ in range(max_search_layers)]
    
    for i, x0 in enumerate(init_structures):
        if time.time() > deadline - 5.0:
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, options=nm_options)
            
            if res.success or res.fun < best_score:
                x_opt = res.x
                # Reconstruct design
                thicknesses = []
                materials = []
                for val in x_opt:
                    if val > 0:
                        d = val * scale
                        m = 'H'
                    else:
                        d = -val * scale
                        m = 'L'
                    
                    if d > max_layer_nm:
                        d = max_layer_nm
                    if d < 0:
                        d = 0
                    thicknesses.append(d)
                    materials.append(m)
                
                # Filter out zero-thickness layers for cleaner output
                final_t = []
                final_m = []
                for t, m in zip(thicknesses, materials):
                    if t > 1e-6:
                        final_t.append(t)
                        final_m.append(m)
                
                if len(final_t) == 0:
                    continue
                    
                # Check constraints
                if sum(final_t) > max_total_nm:
                    continue
                if any(t > max_layer_nm for t in final_t):
                    continue
                if len(final_t) > max_layers:
                    continue
                    
                score = _evaluate_design(final_t, final_m, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
                
                if score < best_score:
                    best_score = score
                    best_thicknesses = final_t
                    best_materials = final_m
                    
        except Exception:
            continue

    # Final validity check
    if not best_thicknesses or any(t < 0 or t > max_layer_nm for t in best_thicknesses) or sum(best_thicknesses) > max_total_nm:
        return ["L"], [qw_L]
        
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
