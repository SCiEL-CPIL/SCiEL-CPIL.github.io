"""Broadband AR coating design using characteristic-matrix method and SciPy optimization."""
import numpy as np
import time
import scipy.optimize

def _characteristic_matrix(n0, ns, materials, thicknesses, lambdas):
    """
    Compute reflectance using the characteristic matrix method.
    n0: ambient index, ns: substrate index
    materials: list of layer indices (same order as thicknesses)
    thicknesses: list of physical thicknesses in nm
    lambdas: array of wavelengths in nm
    Returns: array of reflectance values (0 to 1)
    """
    N = len(thicknesses)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(lambdas, r * r, dtype=float)
    
    # Build total characteristic matrix M = M1 * M2 * ... * MN
    # Mj = [[cos(delta_j), i*sin(delta_j)/nj], [i*nj*sin(delta_j), cos(delta_j)]]
    # delta_j = 2*pi*nj*d_j/lambda
    
    M11 = np.ones_like(lambdas, dtype=complex)
    M12 = np.zeros_like(lambdas, dtype=complex)
    M21 = np.zeros_like(lambdas, dtype=complex)
    M22 = np.ones_like(lambdas, dtype=complex)
    
    for idx in range(N):
        nj = materials[idx]
        dj = thicknesses[idx]
        delta = 2.0 * np.pi * nj * dj / lambdas
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Layer matrix
        m11 = cos_d
        m12 = 1j * sin_d / nj
        m21 = 1j * nj * sin_d
        m22 = cos_d
        
        # Multiply M = M * Mj
        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = S21 * m11 + M22 * m21
        new_M22 = S21 * m12 + M22 * m22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    # Reflectance calculation
    Ys = np.full(len(lambdas), ns, dtype=complex)
    numerator = M11 + Ys * M12
    denominator = M21 + Ys * M22
    
    Y = numerator / denominator
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r) ** 2
    
    return R

def _mean_reflectance_percent(n0, ns, material_indices, thicknesses, lambdas):
    """Compute mean reflectance in percent."""
    R = _characteristic_matrix(n0, ns, material_indices, thicknesses, lambdas)
    return np.mean(R) * 100.0

def design(problem):
    """
    Design a broadband AR coating minimizing mean reflectance over 400-800 nm.
    Returns (materials_list, thicknesses_list).
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation: 401 equally spaced points
    lambdas = np.linspace(band[0], band[1], 401)
    
    # Analytical initialization: quarter-wave stack
    lam_center_geo = np.sqrt(band[0] * band[1])
    lam_center_arith = 0.5 * (band[0] + band[1])
    
    best_score = np.inf
    best_materials = None
    best_thicknesses = None
    
    # Try different stack configurations
    # Symmetric stacks (L H L ... L) are standard for broadband AR
    # We try odd numbers of layers to maintain symmetry (start/end with L)
    # We also try even numbers if max_layers allows, but odd is usually better for L-H-L pattern
    
    # List of candidate layer counts
    candidate_layers = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    
    for n_layers in candidate_layers:
        if n_layers > max_layers:
            break
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Create alternating L/H pattern starting and ending with L
        materials_labels = []
        material_indices = []
        for i in range(n_layers):
            if i % 2 == 0:
                materials_labels.append("L")
                material_indices.append(nL)
            else:
                materials_labels.append("H")
                material_indices.append(nH)
        
        # Initial guess: Quarter-wave at geometric mean
        init_thicknesses = []
        for m in material_indices:
            d = lam_center_geo / (4.0 * m)
            d = min(d, max_layer_nm)
            init_thicknesses.append(d)
            
        # Check initial total thickness
        if sum(init_thicknesses) > max_total_nm:
            scale = max_total_nm / sum(init_thicknesses)
            init_thicknesses = [d * scale for d in init_thicknesses]
        
        # Define objective function for SciPy
        def objective(x):
            # x is a numpy array of thicknesses
            # Ensure x is within bounds (SciPy should handle this, but just in case)
            # Calculate reflectance
            return _mean_reflectance_percent(n0, ns, material_indices, x, lambdas)
        
        # Bounds for each layer
        bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
        
        # Constraints: sum(x) <= max_total_nm
        # SLSQP constraint format: fun(x) >= 0
        constraints = [{'type': 'ineq', 'fun': lambda x: max_total_nm - np.sum(x)}]
        
        # Use SLSQP for robustness with constraints
        # Limit iterations to save time
        try:
            res = scipy.optimize.minimize(
                objective, 
                init_thicknesses, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if res.success:
                score = res.fun
                if score < best_score:
                    best_score = score
                    best_materials = materials_labels[:]
                    best_thicknesses = res.x.tolist()
                    
        except Exception:
            # If optimization fails, try a simpler gradient-free method or skip
            pass

    # If no valid solution found via SciPy, fall back to a simple analytical solution
    if best_materials is None:
        # Try a single layer L
        d = lam_center_geo / (4.0 * nL)
        d = min(d, max_layer_nm)
        return ["L"], [d]
        
    # Final validation and cleanup
    # Clip thicknesses to be strictly within bounds
    best_thicknesses = [max(0.0, min(d, max_layer_nm)) for d in best_thicknesses]
    
    # Check total thickness again after clipping
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [d * scale for d in best_thicknesses]
        
    return best_materials, best_thicknesses
