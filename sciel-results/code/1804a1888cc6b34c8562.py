"""Antireflection coating design via multi-start local optimization."""
import time
import numpy as np

def _characteristic_matrix(n, d, lam, n0):
    """Compute 2x2 characteristic matrix for a single layer.
    
    Parameters:
        n: refractive index of the layer
        d: physical thickness in nm
        lam: wavelength in nm
        n0: ambient index (for phase calculation)
    
    Returns:
        2x2 numpy complex matrix
    """
    beta = 2.0 * np.pi * n * d / lam
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    # For normal incidence, the characteristic matrix is:
    # [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
    return np.array([[cos_b, 1j * sin_b / n],
                     [1j * n * sin_b, cos_b]], dtype=complex)

def _reflectance_stack(materials, thicknesses, wavelengths, n0, ns):
    """Compute reflectance spectrum using characteristic matrix method.
    
    Parameters:
        materials: list of 'L' or 'H'
        thicknesses: list of thicknesses in nm
        wavelengths: array of wavelengths in nm
        n0: ambient index
        ns: substrate index
    
    Returns:
        Array of reflectance values (fraction, not percent)
    """
    n_vals = np.array([1.38 if m == 'L' else 2.35 for m in materials])
    d_vals = np.array(thicknesses, dtype=float)
    
    reflectances = np.zeros(len(wavelengths))
    
    for i, lam in enumerate(wavelengths):
        # Build the total matrix product
        # Start with the substrate
        # The total matrix M = M1 * M2 * ... * MN
        # Then Y = M[0,1] / M[0,0] (admittance)
        # R = |(Y - n0)/(Y + n0)|^2
        
        # For a stack, we multiply matrices from air side to substrate
        # M_total = M_layer1 * M_layer2 * ... * M_layerN
        # Then we need to account for the substrate boundary
        
        # Actually, the standard approach:
        # M_total = M1 * M2 * ... * MN
        # Then the input admittance Y = (M00 + M01 * ns) / (M10 + M11 * ns)
        # Wait, let me reconsider.
        
        # The characteristic matrix for each layer j is:
        # Mj = [[cos(beta_j), i*sin(beta_j)/n_j], [i*n_j*sin(beta_j), cos(beta_j)]]
        
        # The total matrix is M = M1 * M2 * ... * MN
        # The input admittance (referred to the first medium) is:
        # Y = (M[0,0] + M[0,1] * ns) / (M[1,0] + M[1,1] * ns)
        # Hmm, this isn't quite right either. Let me think more carefully.
        
        # Standard thin-film theory:
        # For a stack of N layers on a substrate with index ns, illuminated from medium n0:
        # The total characteristic matrix is M = M1 * M2 * ... * MN
        # The input admittance Y_in is given by:
        # Y_in = (M[0,0] + M[0,1] * ns) / (M[1,0] + M[1,1] * ns)
        # Wait, I need to be more careful.
        
        # Let me use the standard formulation:
        # For each layer j, the characteristic matrix is:
        # Mj = [[cos(delta_j), i*sin(delta_j)/nj], [i*nj*sin(delta_j), cos(delta_j)]]
        # where delta_j = 2*pi*nj*dj/lam
        # 
        # The total matrix for N layers is M = M1 * M2 * ... * MN
        # 
        # The input admittance Y is:
        # Y = (M[0,0] + M[0,1]*ns) / (M[1,0] + M[1,1]*ns)
        # 
        # Hmm, actually I think the correct formula involves the substrate.
        # Let me look at this differently.
        
        # The transfer matrix approach:
        # At the interface between the last layer and the substrate, the admittance is ns.
        # Working backwards through the layers:
        # Y_N = ns
        # Y_{j-1} = (Y_j * cos(delta_j) + i*nj*sin(delta_j)) / (nj*cos(delta_j) + i*Y_j*sin(delta_j))
        # 
        # Or equivalently, using the matrix product:
        # M = M1 * M2 * ... * MN
        # Y_in = (M[0,0] + M[0,1]*ns) / (M[1,0] + M[1,1]*ns)
        
        # Let me just compute the matrix product and use the formula.
        
        M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]], dtype=complex)
        for j in range(len(materials)):
            Mj = _characteristic_matrix(n_vals[j], d_vals[j], lam, n0)
            M = M @ Mj
        
        # The input admittance
        # Y = (M[0,0] + M[0,1] * ns) / (M[1,0] + M[1,1] * ns)
        Y = (M[0, 0] + M[0, 1] * ns) / (M[1, 0] + M[1, 1] * ns)
        
        # Reflectance
        r = (Y - n0) / (Y + n0)
        reflectances[i] = np.abs(r) ** 2
    
    return reflectances

def _objective(materials, thicknesses, problem, wavelengths):
    """Compute mean reflectance in percent."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    refl = _reflectance_stack(materials, thicknesses, wavelengths, n0, ns)
    return 100.0 * np.mean(refl)

def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelength grid
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Material indices
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    
    # Best solution found
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    def update_best(materials, thicknesses, score):
        nonlocal best_score, best_materials, best_thicknesses
        if score < best_score:
            best_score = score
            best_materials = list(materials)
            best_thicknesses = list(thicknesses)
    
    # Initial guess: a few quarter-wave layers
    # Start with a reasonable initial design
    # Try a 4-layer stack: L H L H or similar
    
    # Multi-start optimization using Nelder-Mead
    from scipy.optimize import minimize
    
    def _encode(materials, thicknesses):
        """Encode a design as a vector for optimization."""
        # We'll optimize over both material choice and thickness
        # For simplicity, fix the number of layers and material pattern,
        # and optimize thicknesses only
        return thicknesses
    
    def _decode(x, n_layers, material_pattern):
        """Decode optimization vector back to design."""
        thicknesses = x.tolist()
        return material_pattern, thicknesses
    
    # Try different numbers of layers and material patterns
    # We'll use a set of predefined patterns and optimize thicknesses
    
    patterns = []
    
    # Single layer
    patterns.append(['L'])
    patterns.append(['H'])
    
    # Two layers
    patterns.append(['L', 'H'])
    patterns.append(['H', 'L'])
    
    # Three layers
    patterns.append(['L', 'H', 'L'])
    patterns.append(['H', 'L', 'H'])
    patterns.append(['L', 'L', 'H'])
    patterns.append(['H', 'H', 'L'])
    
    # Four layers
    patterns.append(['L', 'H', 'L', 'H'])
    patterns.append(['H', 'L', 'H', 'L'])
    patterns.append(['L', 'L', 'H', 'H'])
    patterns.append(['H', 'H', 'L', 'L'])
    
    # Five layers
    patterns.append(['L', 'H', 'L', 'H', 'L'])
    patterns.append(['H', 'L', 'H', 'L', 'H'])
    
    # Six layers
    patterns.append(['L', 'H', 'L', 'H', 'L', 'H'])
    patterns.append(['H', 'L', 'H', 'L', 'H', 'L'])
    
    # For each pattern, run multi-start Nelder-Mead
    for pattern in patterns:
        if time.time() > deadline:
            break
        
        n_layers = len(pattern)
        
        # Initial thickness guesses: quarter-wave at center wavelength
        center_lam = 0.5 * (band[0] + band[1])
        
        # Multiple random starts
        n_starts = 5
        for start_i in range(n_starts):
            if time.time() > deadline:
                break
            
            # Generate initial thicknesses
            if start_i == 0:
                # Quarter-wave initial guess
                init = []
                for m in pattern:
                    n = nL if m == 'L' else nH
                    init.append(center_lam / (4.0 * n))
            else:
                # Random initial guess
                init = []
                for m in pattern:
                    n = nL if m == 'L' else nH
                    # Random thickness between 0 and max_layer_nm
                    init.append(np.random.uniform(0.1, max_layer_nm))
            
            # Ensure total thickness constraint
            total = sum(init)
            if total > max_total_nm:
                scale = max_total_nm / total
                init = [t * scale for t in init]
            
            # Define the objective function for optimization
            def obj(x):
                # Check constraints
                if any(t < 0 or t > max_layer_nm for t in x):
                    return 1e6
                if sum(x) > max_total_nm:
                    return 1e6
                
                score = _objective(pattern, x, problem, wavelengths)
                return score
            
            # Run Nelder-Mead
            try:
                result = minimize(obj, init, method='Nelder-Mead',
                                  options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if result.fun < best_score:
                    # Check constraints
                    x = result.x
                    if all(0 <= t <= max_layer_nm for t in x) and sum(x) <= max_total_nm:
                        update_best(pattern, x, result.fun)
            except Exception:
                pass
    
    # If no valid solution found, return a simple quarter-wave layer
    if best_materials is None:
        center_lam = 0.5 * (band[0] + band[1])
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * nL)]
    
    # Final constraint check and cleanup
    # Ensure all thicknesses are within bounds
    cleaned_thicknesses = []
    for t in best_thicknesses:
        t = max(0.0, min(max_layer_nm, t))
        cleaned_thicknesses.append(t)
    
    # Check total thickness
    total = sum(cleaned_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        cleaned_thicknesses = [t * scale for t in cleaned_thicknesses]
    
    return best_materials, cleaned_thicknesses
