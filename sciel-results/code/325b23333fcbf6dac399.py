"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _reflectance_stack(labels, thicknesses, n_ambient, n_substrate, materials, wavelengths):
    """
    Compute normal-incidence reflectance spectrum using the characteristic-matrix method.
    Returns array of reflectance (0 to 1) for each wavelength.
    """
    N = len(wavelengths)
    n_L = materials['L']
    n_H = materials['H']
    
    # Precompute indices for each layer
    n_layers = []
    for lab in labels:
        n_layers.append(n_L if lab == 'L' else n_H)
    
    reflectances = np.zeros(N)
    for i, lam in enumerate(wavelengths):
        # Build characteristic matrices
        M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for j in range(len(labels)):
            n_j = n_layers[j]
            d_j = thicknesses[j]
            beta_j = 2.0 * math.pi * n_j * d_j / lam
            # For normal incidence, the characteristic matrix for layer j is:
            # [[cos(beta_j), i*sin(beta_j)/n_j], [i*n_j*sin(beta_j), cos(beta_j)]]
            c = math.cos(beta_j)
            s = math.sin(beta_j)
            M_j = np.array([[c, 1j * s / n_j],
                            [1j * n_j * s, c]], dtype=complex)
            M_total = M_total @ M_j
        
        # Substrate and ambient
        n_s = n_substrate
        n_0 = n_ambient
        
        # The admittance at the interface
        B, C = M_total[0, 0], M_total[1, 0]
        A, D = M_total[0, 1], M_total[1, 1]
        
        # For normal incidence:
        # Y = (n_s * B + C) / (n_s * A + D)  -- this is the input admittance
        # Actually, let me use the standard formulation:
        # The characteristic matrix for the stack is M = M_1 @ M_2 @ ... @ M_k
        # Then the input admittance Y_in = (B + n_s * C) / (A + n_s * D)
        # where A, B, C, D are elements of M
        # Wait, let me be more careful.
        
        # Standard thin-film theory:
        # For a stack of layers on a substrate, the characteristic matrix is:
        # M = M_1 * M_2 * ... * M_k
        # where each M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
        # delta_j = (2*pi*n_j*d_j/lambda)*cos(theta_j)
        
        # The input admittance is:
        # Y_in = (B + n_s*C) / (A + n_s*D)
        # where M = [[A, B], [C, D]]
        
        # Wait, I need to be consistent with notation. Let me use:
        # M_total = [[M11, M12], [M21, M22]]
        # Y_in = (M21 + n_s * M22) / (M11 + n_s * M12)
        
        M11, M12 = M_total[0, 0], M_total[0, 1]
        M21, M22 = M_total[1, 0], M_total[1, 1]
        
        Y_in = (M21 + n_s * M22) / (M11 + n_s * M12)
        
        # Reflectance at the air-coating interface:
        # r = (n_0 - Y_in) / (n_0 + Y_in)
        # R = |r|^2
        r = (n_0 - Y_in) / (n_0 + Y_in)
        R = np.abs(r) ** 2
        reflectances[i] = R
    
    return reflectances

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    band = problem["band_nm"]
    lam_min, lam_max = band[0], band[1]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials['L']
    n_H = materials['H']
    
    # Generate test wavelengths
    wavelengths = np.linspace(lam_min, lam_max, 401)
    
    # Baseline: single quarter-wave L layer
    lam_center = 0.5 * (lam_min + lam_max)
    best_labels = ["L"]
    best_thicknesses = [lam_center / (4.0 * n_L)]
    best_R = _reflectance_stack(best_labels, best_thicknesses, n_ambient, n_substrate, materials, wavelengths).mean()
    
    # Try a few more initial designs
    # Two quarter-wave layers: L then H
    try:
        labels = ["L", "H"]
        thicknesses = [lam_center / (4.0 * n_L), lam_center / (4.0 * n_H)]
        R = _reflectance_stack(labels, thicknesses, n_ambient, n_substrate, materials, wavelengths).mean()
        if R < best_R:
            best_labels = labels
            best_thicknesses = thicknesses
            best_R = R
    except:
        pass
    
    # Three quarter-wave layers: L, H, L
    try:
        labels = ["L", "H", "L"]
        thicknesses = [lam_center / (4.0 * n_L), lam_center / (4.0 * n_H), lam_center / (4.0 * n_L)]
        R = _reflectance_stack(labels, thicknesses, n_ambient, n_substrate, materials, wavelengths).mean()
        if R < best_R:
            best_labels = labels
            best_thicknesses = thicknesses
            best_R = R
    except:
        pass
    
    # Use differential evolution or similar optimization
    # We'll optimize both the number of layers and their thicknesses
    
    def _objective(params, n_layers):
        """Objective function for optimization."""
        # params: array of thicknesses
        if len(params) != n_layers:
            return 1.0
        
        # Check constraints
        if np.any(params < 0) or np.any(params > max_layer_nm):
            return 1.0
        if np.sum(params) > max_total_nm:
            return 1.0
        
        # Determine layer types: alternate L, H, L, H, ...
        labels = []
        for i in range(n_layers):
            labels.append('L' if i % 2 == 0 else 'H')
        
        try:
            R = _reflectance_stack(labels, list(params), n_ambient, n_substrate, materials, wavelengths).mean()
            return R
        except:
            return 1.0
    
    # Try different numbers of layers with alternating L/H pattern
    # Start with a reasonable number of layers
    for n_layers in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
        if time.time() > deadline:
            break
        
        # Initial guess: quarter-wave thicknesses
        init_thicknesses = np.array([lam_center / (4.0 * (n_L if i % 2 == 0 else n_H)) for i in range(n_layers)])
        
        # Scale to fit within total thickness constraint
        total = np.sum(init_thicknesses)
        if total > max_total_nm:
            init_thicknesses = init_thicknesses * (max_total_nm / total)
        
        # Use scipy's differential evolution
        try:
            from scipy.optimize import differential_evolution
            bounds = [(0, max_layer_nm)] * n_layers
            result = differential_evolution(
                _objective,
                bounds,
                args=(n_layers,),
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=True
            )
            
            if result.success and result.fun < best_R:
                # Reconstruct labels
                labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
                best_labels = labels
                best_thicknesses = list(result.x)
                best_R = result.fun
        except:
            pass
    
    # Also try non-alternating patterns for small numbers of layers
    # For 2-4 layers, try all possible combinations
    if time.time() < deadline:
        from itertools import product
        
        for n_layers in [2, 3, 4]:
            if time.time() > deadline:
                break
            
            for pattern in product(['L', 'H'], repeat=n_layers):
                if time.time() > deadline:
                    break
                
                # Initial guess: quarter-wave
                init_thicknesses = np.array([lam_center / (4.0 * (n_L if lab == 'L' else n_H)) for lab in pattern])
                total = np.sum(init_thicknesses)
                if total > max_total_nm:
                    init_thicknesses = init_thicknesses * (max_total_nm / total)
                
                def _obj_pattern(params):
                    if np.any(params < 0) or np.any(params > max_layer_nm):
                        return 1.0
                    if np.sum(params) > max_total_nm:
                        return 1.0
                    try:
                        R = _reflectance_stack(list(pattern), list(params), n_ambient, n_substrate, materials, wavelengths).mean()
                        return R
                    except:
                        return 1.0
                
                try:
                    from scipy.optimize import differential_evolution
                    bounds = [(0, max_layer_nm)] * n_layers
                    result = differential_evolution(
                        _obj_pattern,
                        bounds,
                        maxiter=50,
                        popsize=15,
                        tol=1e-6,
                        seed=42,
                        polish=True
                    )
                    
                    if result.success and result.fun < best_R:
                        best_labels = list(pattern)
                        best_thicknesses = list(result.x)
                        best_R = result.fun
                except:
                    pass
    
    # Ensure constraints are met
    for i in range(len(best_thicknesses)):
        best_thicknesses[i] = max(0.0, min(max_layer_nm, best_thicknesses[i]))
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    # Remove zero-thickness layers
    final_labels = []
    final_thicknesses = []
    for lab, t in zip(best_labels, best_thicknesses):
        if t > 1e-6:
            final_labels.append(lab)
            final_thicknesses.append(t)
    
    # If all layers were removed, use the baseline
    if len(final_labels) == 0:
        final_labels = ["L"]
        final_thicknesses = [lam_center / (4.0 * n_L)]
    
    return final_labels, final_thicknesses
# EVOLVE-BLOCK-END
