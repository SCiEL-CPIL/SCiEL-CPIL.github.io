"""
Broadband antireflection coating design via Deterministic Multi-Start Local Optimization.

Strategy:
1. Define a set of promising structural templates (fixed material sequences).
2. For each template, use Nelder-Mead simplex optimization to find optimal thicknesses.
3. Use a highly vectorized reflectance calculation to speed up evaluations.
4. Return the best valid design found across all templates.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _vectorized_reflectance_pct(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance percentage for a given design.
    
    Args:
        thicknesses: array of physical thicknesses in nm
        materials: list of 'L'/'H' labels corresponding to thicknesses
        n_ambient: ambient refractive index
        n_substrate: substrate refractive index
        n_L: low index
        n_H: high index
        wls: array of wavelengths in nm
        
    Returns:
        Mean reflectance in percent (float)
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return float(r * r) * 100.0
    
    # Initialize total characteristic matrix M = I
    # Shape: (len(wls), 2, 2)
    M = np.zeros((len(wls), 2, 2), dtype=np.complex128)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    # Loop through layers
    for i in range(n_layers):
        n = n_H if materials[i] == 'H' else n_L
        d = thicknesses[i]
        
        if d < 1e-6:
            continue
            
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Layer matrix elements
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        # Matrix multiplication M_new = M @ L
        # M is (N, 2, 2), L is scalar per wavelength but effectively (N, 2, 2) if broadcast
        # We do element-wise ops to avoid explicit (N,2,2) L construction overhead if possible, 
        # but numpy broadcasting handles it well.
        
        new_M11 = M[:, 0, 0] * L11 + M[:, 0, 1] * L21
        new_M12 = M[:, 0, 0] * L12 + M[:, 0, 1] * L22
        new_M21 = M[:, 1, 0] * L11 + M[:, 1, 1] * L21
        new_M22 = M[:, 1, 0] * L12 + M[:, 1, 1] * L22
        
        M[:, 0, 0] = new_M11
        M[:, 0, 1] = new_M12
        M[:, 1, 0] = new_M21
        M[:, 1, 1] = new_M22
        
    # Admittance Y
    Y = (M[:, 0, 0] * n_substrate + M[:, 0, 1]) / (M[:, 1, 0] * n_substrate + M[:, 1, 1])
    
    # Reflection coefficient r
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r)**2
    
    return float(np.mean(R) * 100.0)

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
    
    wls = np.linspace(wl_min, wl_max, 401)
    
    # Pre-compute quarter-wave thicknesses for reference
    wl_center = 0.5 * (wl_min + wl_max)
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    # Define candidate structures (material sequences)
    # We include various lengths and patterns.
    # Common good patterns for broadband AR:
    # 1. Single layer (L)
    # 2. Two layers (L, H) or (H, L)
    # 3. Three layers (L, H, L) or (H, L, H)
    # 4. Five layers (L, H, L, H, L) etc.
    # We'll generate a diverse set.
    
    templates = []
    
    # Simple
    templates.append(['L'])
    templates.append(['H'])
    
    # Two layers
    templates.append(['L', 'H'])
    templates.append(['H', 'L'])
    
    # Three layers
    templates.append(['L', 'H', 'L'])
    templates.append(['H', 'L', 'H'])
    templates.append(['L', 'L', 'H'])
    templates.append(['H', 'H', 'L'])
    templates.append(['L', 'H', 'H'])
    templates.append(['H', 'L', 'L'])
    
    # Four layers
    templates.append(['L', 'H', 'L', 'H'])
    templates.append(['H', 'L', 'H', 'L'])
    templates.append(['L', 'H', 'H', 'L'])
    templates.append(['H', 'L', 'L', 'H'])
    
    # Five layers
    templates.append(['L', 'H', 'L', 'H', 'L'])
    templates.append(['H', 'L', 'H', 'L', 'H'])
    templates.append(['L', 'H', 'H', 'L', 'L'])
    templates.append(['H', 'L', 'L', 'H', 'H'])
    
    # Six layers
    templates.append(['L', 'H', 'L', 'H', 'L', 'H'])
    templates.append(['H', 'L', 'H', 'L', 'H', 'L'])
    
    # Seven layers
    templates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L'])
    templates.append(['H', 'L', 'H', 'L', 'H', 'L', 'H'])
    
    # Filter templates that exceed max_layers
    valid_templates = [t for t in templates if len(t) <= max_layers]
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # For each template, run Nelder-Mead optimization
    for mats in valid_templates:
        if time.time() > deadline - 5.0:
            break
            
        n_layers = len(mats)
        
        # Initial guess: Quarter-wave thicknesses
        x0 = np.array([qw_H if m == 'H' else qw_L for m in mats])
        
        # Bounds for Nelder-Mead are handled by clamping in the objective, 
        # or we can use trust-constr, but Nelder-Mead is robust enough with clamping.
        
        def objective(x):
            # Clamp thicknesses to valid range [0, max_layer_nm]
            x_clamped = np.clip(x, 0, max_layer_nm)
            
            # Check total thickness constraint
            total_t = np.sum(x_clamped)
            if total_t > max_total_nm:
                # Apply penalty to push solution back
                penalty = 10.0 * (total_t - max_total_nm)
            else:
                penalty = 0.0
            
            # Calculate reflectance
            r_pct = _vectorized_reflectance_pct(x_clamped, mats, n_ambient, n_substrate, n_L, n_H, wls)
            
            return r_pct + penalty
        
        # Run optimization
        try:
            # Use Nelder-Mead
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            x_opt = res.x
            x_opt = np.clip(x_opt, 0, max_layer_nm)
            
            # Final check on constraints
            if np.sum(x_opt) <= max_total_nm + 1e-6:
                score = _vectorized_reflectance_pct(x_opt, mats, n_ambient, n_substrate, n_L, n_H, wls)
                
                if score < best_score:
                    best_score = score
                    best_materials = mats[:]
                    best_thicknesses = x_opt.tolist()
                    
        except Exception:
            pass
    
    # If no valid solution found or all failed, fallback to a simple known good design
    if best_thicknesses is None:
        # Simple quarter-wave L layer
        best_materials = ['L']
        best_thicknesses = [qw_L]
        # Verify it's valid
        if best_thicknesses[0] > max_layer_nm:
            best_thicknesses[0] = max_layer_nm
        if sum(best_thicknesses) > max_total_nm:
            scale = max_total_nm / sum(best_thicknesses)
            best_thicknesses = [t * scale for t in best_thicknesses]
            
    # Ensure return types are correct
    return best_materials, [float(t) for t in best_thicknesses]

# EVOLVE-BLOCK-END
