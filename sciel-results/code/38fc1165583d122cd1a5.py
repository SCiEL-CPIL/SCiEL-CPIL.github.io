"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _reflectance_vectorized(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """
    Compute reflectance at all wavelengths in wls using the characteristic matrix method.
    Vectorized over wavelengths for speed.
    """
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    n_layers = len(materials)
    if n_layers == 0:
        # Bare substrate
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wls, np.abs(r)**2)

    # Precompute optical properties
    n_vals = np.array([materials_dict[m] for m in materials])
    d_vals = np.array(thicknesses)
    
    # beta = 2*pi*n*d/wl
    # Shape: (n_wls, n_layers)
    # wls: (n_wls,) -> (n_wls, 1)
    # n_vals: (n_layers,) -> (1, n_layers)
    # d_vals: (n_layers,) -> (1, n_layers)
    beta = 2.0 * np.pi * n_vals[None, :] * d_vals[None, :] / wls[:, None]
    
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    
    # Characteristic matrices M_i for each layer and wavelength
    # M = [[cos, j*sin/n], [n*j*sin, cos]]
    # We need the product of these matrices.
    # Since n is constant for a given layer index, we can precompute the '1/n' and 'n' terms.
    
    # Initialize total matrix T for each wavelength
    # T starts as Identity
    # T: (n_wls, 2, 2) complex
    T = np.zeros((len(wls), 2, 2), dtype=complex)
    T[:, 0, 0] = 1.0
    T[:, 1, 1] = 1.0
    
    for i in range(n_layers):
        n_i = n_vals[i]
        # cos_i = cos_beta[:, i]
        # sin_i = sin_beta[:, i]
        
        # M_i = [[cos_i, j*sin_i/n_i], [n_i*j*sin_i, cos_i]]
        M = np.zeros((len(wls), 2, 2), dtype=complex)
        M[:, 0, 0] = cos_beta[:, i]
        M[:, 0, 1] = 1.0j * sin_beta[:, i] / n_i
        M[:, 1, 0] = n_i * 1.0j * sin_beta[:, i]
        M[:, 1, 1] = cos_beta[:, i]
        
        # T = T @ M
        # Batch matrix multiplication
        T = np.matmul(T, M)
        
    # Calculate reflectance
    # Y_sub = substrate_index
    # Y_in = (T[1,0] + T[1,1]*Y_sub) / (T[0,0] + T[0,1]*Y_sub)
    # R = |(Y_0 - Y_in) / (Y_0 + Y_in)|^2
    
    Y_sub = substrate_index
    Y_0 = ambient_index
    
    numerator = T[:, 1, 0] + T[:, 1, 1] * Y_sub
    denominator = T[:, 0, 0] + T[:, 0, 1] * Y_sub
    
    Y_in = numerator / denominator
    
    r = (Y_0 - Y_in) / (Y_0 + Y_in)
    R = np.abs(r)**2
    
    return R

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

def design(problem):
    """
    Design a broadband antireflection coating using multi-start Nelder-Mead optimization.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit (evaluator kills at 120s, task says 60s CPU but we have 120s wall clock limit mentioned in prompt "Each evaluation must finish within 120 s")
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layers = problem["max_layers"]
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_score = float('inf')
    best_materials = ["L"]
    best_thicknesses = [100.0]
    
    # Objective function for Nelder-Mead
    # x: array of thicknesses
    # materials: list of 'L'/'H'
    def objective(x, mats):
        if time.time() - start_time > deadline:
            return 1e9
        # Clip to valid range to handle numerical overshoots
        x_clipped = np.clip(x, 0.0, max_layer_nm)
        if np.sum(x_clipped) > max_total_nm:
            return 1e9
        try:
            score = _mean_reflectance(mats, x_clipped, band_nm, 401, n_amb, n_sub, materials_dict)
            return score
        except Exception:
            return 1e9

    # Define a set of promising topologies (material sequences)
    # We focus on alternating layers which are standard for AR coatings.
    # We will test various numbers of layers and starting materials.
    
    topologies = []
    
    # Single layer
    topologies.append(['L'])
    topologies.append(['H'])
    
    # Two layers
    topologies.append(['L', 'H'])
    topologies.append(['H', 'L'])
    
    # Three layers
    topologies.append(['L', 'H', 'L'])
    topologies.append(['H', 'L', 'H'])
    
    # Four layers
    topologies.append(['L', 'H', 'L', 'H'])
    topologies.append(['H', 'L', 'H', 'L'])
    
    # Five layers
    topologies.append(['L', 'H', 'L', 'H', 'L'])
    topologies.append(['H', 'L', 'H', 'L', 'H'])
    
    # Six layers
    topologies.append(['L', 'H', 'L', 'H', 'L', 'H'])
    topologies.append(['H', 'L', 'H', 'L', 'H', 'L'])
    
    # Seven layers
    topologies.append(['L', 'H', 'L', 'H', 'L', 'H', 'L'])
    topologies.append(['H', 'L', 'H', 'L', 'H', 'L', 'H'])
    
    # Eight layers
    topologies.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    topologies.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Ten layers (often good for broadband)
    topologies.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    topologies.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Twelve layers
    topologies.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    topologies.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Filter topologies that respect max_layers
    topologies = [t for t in topologies if len(t) <= max_layers]
    
    # For each topology, run Nelder-Mead from a few starting points
    # Starting points: Quarter-wave at center wavelength, and some random variations
    
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    
    for mats in topologies:
        if time.time() - start_time > deadline - 5:
            break
            
        n_layers = len(mats)
        
        # Generate starting points
        starts = []
        
        # Start 1: Quarter wave at center wavelength
        x0 = np.zeros(n_layers)
        for i, m in enumerate(mats):
            n = nL if m == 'L' else nH
            x0[i] = wl_center / (4.0 * n)
        starts.append(x0)
        
        # Start 2: Quarter wave at short wavelength
        x0_short = np.zeros(n_layers)
        for i, m in enumerate(mats):
            n = nL if m == 'L' else nH
            x0_short[i] = band_nm[0] / (4.0 * n)
        starts.append(x0_short)
        
        # Start 3: Quarter wave at long wavelength
        x0_long = np.zeros(n_layers)
        for i, m in enumerate(mats):
            n = nL if m == 'L' else nH
            x0_long[i] = band_nm[1] / (4.0 * n)
        starts.append(x0_long)
        
        # Start 4: Random within bounds
        np.random.seed(123 + n_layers)
        x0_rand = np.random.uniform(0.1 * max_layer_nm, 0.9 * max_layer_nm, n_layers)
        # Scale to fit total budget if necessary
        total = np.sum(x0_rand)
        if total > max_total_nm:
            x0_rand *= (max_total_nm / total) * 0.9
        starts.append(x0_rand)
        
        # Run optimization for each start
        for x_init in starts:
            if time.time() - start_time > deadline - 2:
                break
            
            # Check if start is valid
            if np.any(x_init > max_layer_nm) or np.sum(x_init) > max_total_nm:
                continue
                
            try:
                res = minimize(
                    objective, 
                    x_init, 
                    args=(mats,),
                    method='Nelder-Mead',
                    options={'xatol': 1e-3, 'fatol': 1e-5, 'maxiter': 200, 'adaptive': True}
                )
                
                if time.time() - start_time > deadline:
                    break
                    
                x_opt = np.clip(res.x, 0.0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm:
                    score = _mean_reflectance(mats, x_opt, band_nm, 401, n_amb, n_sub, materials_dict)
                    if score < best_score:
                        best_score = score
                        best_materials = mats[:]
                        best_thicknesses = x_opt.tolist()
                        
            except Exception:
                pass

    # If no valid solution was found (shouldn't happen), fallback to a simple quarter-wave
    if best_score == float('inf'):
        best_materials = ['L']
        best_thicknesses = [wl_center / (4.0 * nL)]
        best_score = _mean_reflectance(best_materials, best_thicknesses, band_nm, 401, n_amb, n_sub, materials_dict)

    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
