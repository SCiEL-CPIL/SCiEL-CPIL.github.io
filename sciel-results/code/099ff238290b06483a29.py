"""Broadband AR coating design using characteristic-matrix method."""
import numpy as np
import time

def _characteristic_matrix(n0, ns, materials, thicknesses, lambdas):
    """
    Compute reflectance using the characteristic matrix method.
    n0: ambient index, ns: substrate index
    materials: dict of layer indices (same order as thicknesses)
    thicknesses: list of physical thicknesses in nm
    lambdas: array of wavelengths in nm
    Returns: array of reflectance values (0 to 1)
    """
    N = len(thicknesses)
    # Build the total characteristic matrix M = M1 * M2 * ... * MN
    # For each layer j: Mj = [[cos(delta_j), i*sin(delta_j)/nj], [i*nj*sin(delta_j), cos(delta_j)]]
    # where delta_j = 2*pi*nj*d_j/lambda
    
    # Pre-compute for all wavelengths at once
    # M_total starts as identity
    M11 = np.ones(len(lambdas), dtype=complex)
    M12 = np.zeros(len(lambdas), dtype=complex)
    M21 = np.zeros(len(lambdas), dtype=complex)
    M22 = np.ones(len(lambdas), dtype=complex)
    
    for j in range(N):
        nj = materials[j]
        dj = thicknesses[j]
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
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    
    # Reflectance: R = |(n0*M11 + n0*ns*M12 - M21 - ns*M22) / (n0*M11 + n0*ns*M12 + M21 + ns*M22)|^2
    # Wait, the standard formula is:
    # Y = (n0*M11 + ns*M12) / (M21 + ns*M22)  -- admittance
    # Actually, let me use the standard characteristic matrix formula:
    # The input admittance is Y = (M11 + Ys*M12) / (M21 + Ys*M22) where Ys = ns (for normal incidence)
    # Then r = (n0 - Y) / (n0 + Y), R = |r|^2
    
    # For normal incidence, the admittance of substrate is just ns
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
    DEADLINE = 140.0  # safety margin before 150s kill
    
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
    # For a quarter-wave AR coating, optimal design uses alternating L and H layers
    # centered at the geometric mean wavelength
    lam_center = np.sqrt(band[0] * band[1])
    
    # Start with a reasonable number of layers
    # A common approach: use an odd number of layers starting and ending with L
    # Let's try different numbers of layers and pick the best
    
    best_score = np.inf
    best_materials = None
    best_thicknesses = None
    
    # Try different stack configurations
    # For broadband AR, a good starting point is a quarter-wave stack
    # with alternating L and H layers
    
    # Try various numbers of layers (odd numbers work well for symmetric stacks)
    for n_layers in [3, 5, 7, 9, 11, 13, 15]:
        if n_layers > max_layers:
            break
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Create alternating L/H pattern starting and ending with L
        # Pattern: L H L H L ... L
        materials = []
        for i in range(n_layers):
            if i % 2 == 0:
                materials.append("L")
            else:
                materials.append("H")
        
        # Initialize thicknesses to quarter-wave at center wavelength
        thicknesses = []
        for m in materials:
            n = nL if m == "L" else nH
            d = lam_center / (4.0 * n)
            d = min(d, max_layer_nm)
            thicknesses.append(d)
        
        # Check total thickness constraint
        total = sum(thicknesses)
        if total > max_total_nm:
            # Scale down
            scale = max_total_nm / total
            thicknesses = [d * scale for d in thicknesses]
        
        material_indices = [nL if m == "L" else nH for m in materials]
        
        # Compute initial score
        score = _mean_reflectance_percent(n0, ns, material_indices, thicknesses, lambdas)
        
        # Local optimization using coordinate descent
        # Optimize each thickness in turn
        improved = True
        max_iter = 50
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            if time.time() - start_time > DEADLINE - 5:
                break
            
            for i in range(len(thicknesses)):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                # Try golden section search on thickness i
                # Bounds: [0, max_layer_nm], but also need total <= max_total_nm
                lo = 0.0
                hi = max_layer_nm
                
                # Also respect total thickness constraint
                other_sum = sum(thicknesses) - thicknesses[i]
                hi = min(hi, max_total_nm - other_sum)
                if hi < lo:
                    continue
                
                # Golden section search
                gr = (np.sqrt(5) - 1) / 2
                x1 = hi - gr * (hi - lo)
                x2 = lo + gr * (hi - lo)
                
                # Evaluate at x1 and x2
                thicknesses[i] = x1
                score_x1 = _mean_reflectance_percent(n0, ns, material_indices, thicknesses, lambdas)
                
                thicknesses[i] = x2
                score_x2 = _mean_reflectance_percent(n0, ns, material_indices, thicknesses, lambdas)
                
                # Iterate golden section
                for _ in range(40):
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    if score_x1 < score_x2:
                        hi = x2
                        x2 = x1
                        score_x2 = score_x1
                        x1 = hi - gr * (hi - lo)
                        thicknesses[i] = x1
                        score_x1 = _mean_reflectance_percent(n0, ns, material_indices, thicknesses, lambdas)
                    else:
                        lo = x1
                        x1 = x2
                        score_x1 = score_x2
                        x2 = lo + gr * (hi - lo)
                        thicknesses[i] = x2
                        score_x2 = _mean_reflectance_percent(n0, ns, material_indices, thicknesses, lambdas)
                
                # Set to best point found
                if score_x1 < score_x2:
                    thicknesses[i] = x1
                    new_score = score_x1
                else:
                    thicknesses[i] = x2
                    new_score = score_x2
                
                if new_score < score - 1e-6:
                    score = new_score
                    improved = True
        
        # Update best
        if score < best_score:
            best_score = score
            best_materials = materials[:]
            best_thicknesses = thicknesses[:]
    
    # If no valid solution found, return a simple quarter-wave layer
    if best_materials is None:
        lam_center = 0.5 * (band[0] + band[1])
        d = lam_center / (4.0 * nL)
        d = min(d, max_layer_nm)
        return ["L"], [d]
    
    # Final validation
    total = sum(best_thicknesses)
    if total > max_total_nm + 1e-6:
        # Scale down
        scale = max_total_nm / total
        best_thicknesses = [d * scale for d in best_thicknesses]
    
    # Ensure all thicknesses are within bounds
    best_thicknesses = [max(0.0, min(d, max_layer_nm)) for d in best_thicknesses]
    
    return best_materials, best_thicknesses
