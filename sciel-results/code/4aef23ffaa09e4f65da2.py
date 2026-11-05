"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_index, substrate_index):
    """
    Compute normal-incidence reflectance using the characteristic matrix method.
    
    Parameters:
    - wavelengths_nm: array of wavelengths in nm
    - indices: array of refractive indices for each layer
    - thicknesses_nm: array of physical thicknesses in nm
    - ambient_index: refractive index of ambient medium
    - substrate_index: refractive index of substrate
    
    Returns:
    - reflectance: array of reflectance values (0 to 1)
    """
    # Convert wavelengths to radians of phase thickness
    # Phase thickness = 2 * pi * n * d / lambda
    # At normal incidence, characteristic matrix for a layer is:
    # [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # where delta = 2*pi*n*d/lambda
    
    # We'll compute the total characteristic matrix by multiplying layer matrices
    # Starting from substrate side to air side (or vice versa, need to be consistent)
    
    # Let's use the standard formulation:
    # The total matrix M = M_1 * M_2 * ... * M_N
    # where M_i is the characteristic matrix of layer i
    # Then the admittance Y = (m11 + m12*ns) / (m21 + m22*ns)
    # where ns is substrate index
    # Reflectance R = |(Y - na)/(Y + na)|^2 where na is ambient index
    
    # Actually, let me be more careful. The characteristic matrix method:
    # For a stack of N layers on a substrate with index ns, in ambient with index na
    # The total characteristic matrix is M = M_1 M_2 ... M_N
    # where M_j = [[cos(delta_j), i sin(delta_j)/n_j], [i n_j sin(delta_j), cos(delta_j)]]
    # delta_j = 2*pi*n_j*d_j/lambda
    
    # Then the optical admittance at the air-side interface is:
    # Y = (m11 + m12 * ns) / (m21 + m22 * ns)
    # where m11, m12, m21, m22 are elements of M
    
    # Reflectance R = |1 - Y/na|^2 / |1 + Y/na|^2 = |(na - Y)/(na + Y)|^2
    
    # Let's implement this
    n_layers = len(indices)
    
    if n_layers == 0:
        # No coating, just bare substrate
        R = ((ambient_index - substrate_index) / (ambient_index + substrate_index))**2
        return np.full_like(wavelengths_nm, R)
    
    reflectance = np.zeros_like(wavelengths_nm)
    
    for idx, wl in enumerate(wavelengths_nm):
        # Compute phase thicknesses
        deltas = 2.0 * np.pi * indices * thicknesses_nm / wl
        
        # Build the total characteristic matrix
        # Start with identity matrix
        m11 = 1.0 + 0j
        m12 = 0.0 + 0j
        m21 = 0.0 + 0j
        m22 = 1.0 + 0j
        
        for j in range(n_layers):
            n_j = indices[j]
            delta_j = deltas[j]
            
            c = np.cos(delta_j)
            s = np.sin(delta_j)
            
            # Layer characteristic matrix
            m11_j = c
            m12_j = 1j * s / n_j
            m21_j = 1j * n_j * s
            m22_j = c
            
            # Multiply: M = M * M_j
            # New m11 = m11*m11_j + m12*m21_j
            # New m12 = m11*m12_j + m12*m22_j
            # New m21 = m21*m11_j + m22*m21_j
            # New m22 = m21*m12_j + m22*m22_j
            
            new_m11 = m11 * m11_j + m12 * m21_j
            new_m12 = m11 * m12_j + m12 * m22_j
            new_m21 = m21 * m11_j + m22 * m21_j
            new_m22 = m21 * m12_j + m22 * m22_j
            
            m11 = new_m11
            m12 = new_m12
            m21 = new_m21
            m22 = new_m22
        
        # Compute admittance
        Y = (m11 + m12 * substrate_index) / (m21 + m22 * substrate_index)
        
        # Compute reflectance
        R = np.abs((ambient_index - Y) / (ambient_index + Y))**2
        reflectance[idx] = R
    
    return reflectance

def _objective(params, problem):
    """
    Objective function for optimization.
    params: flattened array of [n_layers, thickness_1, ..., thickness_n, type_1, ..., type_n]
    Actually, let's use a different parameterization.
    
    Let's use: first parameter is number of layers (integer, 1 to max_layers)
    Then for each layer, we have thickness and type (0 for L, 1 for H)
    
    Actually, for continuous optimization, let's fix the number of layers and optimize
    thicknesses and types. But types are discrete. Let's use a hybrid approach.
    
    Alternative: use differential evolution with integer encoding for types.
    """
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    
    # Extract parameters
    # params[0] = number of layers (1 to max_layers)
    n_layers = int(round(params[0]))
    n_layers = max(1, min(max_layers, n_layers))
    
    # Next n_layers*2 parameters: thickness and type for each layer
    # Type: 0 for L, 1 for H
    indices = []
    thicknesses = []
    for i in range(n_layers):
        thickness = params[1 + 2*i]
        thickness = max(0.0, min(max_layer_nm, thickness))
        type_val = params[1 + 2*i + 1]
        # Map type_val to 0 or 1
        if type_val < 0.5:
            indices.append(materials["L"])
        else:
            indices.append(materials["H"])
        thicknesses.append(thickness)
    
    # Check total thickness constraint
    total_thickness = sum(thicknesses)
    if total_thickness > max_total_nm:
        # Penalize
        return 1.0 + (total_thickness - max_total_nm) / max_total_nm
    
    # Compute wavelengths
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Compute reflectance
    R = _compute_reflectance(wavelengths, indices, thicknesses, ambient_index, substrate_index)
    
    # Mean reflectance in percent
    mean_R = np.mean(R) * 100.0
    
    return mean_R

def _differential_evolution_design(problem):
    """
    Use differential evolution to find optimal coating design.
    """
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    
    # Define bounds for differential evolution
    # params[0]: number of layers, 1 to max_layers
    # For each layer i: thickness (0 to max_layer_nm), type (0 to 1)
    
    # We'll try a fixed number of layers first, then increase
    best_result = None
    best_score = float('inf')
    
    # Try different numbers of layers
    for n_layers in range(1, min(max_layers, 10) + 1):
        # Bounds: [n_layers, t1, type1, t2, type2, ..., tn, typen]
        bounds = [(1, max_layers)]  # n_layers
        for i in range(max_layers):
            bounds.append((0, max_layer_nm))  # thickness
            bounds.append((0, 1))  # type
        
        # Objective function for fixed n_layers
        def obj_fixed(params):
            # Force n_layers to be the current value
            params[0] = n_layers
            return _objective(params, problem)
        
        # Run differential evolution
        try:
            result = differential_evolution(
                obj_fixed,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-4,
                polish=True
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
    
    if best_result is None:
        # Fallback to simple design
        centre = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [centre / (4.0 * materials["L"])]
    
    # Extract design from best_result
    n_layers = int(round(best_result[0]))
    n_layers = max(1, min(max_layers, n_layers))
    
    materials_list = []
    thicknesses_list = []
    for i in range(n_layers):
        thickness = best_result[1 + 2*i]
        thickness = max(0.0, min(max_layer_nm, thickness))
        type_val = best_result[1 + 2*i + 1]
        if type_val < 0.5:
            materials_list.append("L")
            idx = materials["L"]
        else:
            materials_list.append("H")
            idx = materials["H"]
        thicknesses_list.append(thickness)
    
    # Check constraints
    total_thickness = sum(thicknesses_list)
    if total_thickness > max_total_nm:
        # Scale down
        scale = max_total_nm / total_thickness
        thicknesses_list = [t * scale for t in thicknesses_list]
    
    return materials_list, thicknesses_list

def design(problem):
    """
    Design an antireflection coating using differential evolution.
    """
    return _differential_evolution_design(problem)
# EVOLVE-BLOCK-END
