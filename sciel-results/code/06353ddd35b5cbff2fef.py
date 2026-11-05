"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a broadband antireflection coating using a multi-layer quarter-wave stack
    optimized for the center of the band, with additional optimization via gradient descent
    on layer thicknesses to minimize mean reflectance over the band.
    
    We use a symmetric or near-symmetric stack of L and H layers.
    Start with a reasonable initial guess (e.g., 3-5 layers) and refine with scipy minimize.
    """
    from scipy.optimize import minimize
    
    band = problem["band_nm"]
    lam_min, lam_max = band[0], band[1]
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Number of wavelengths for evaluation
    num_wl = 401
    wavelengths = np.linspace(lam_min, lam_max, num_wl)
    
    def compute_reflectance(thicknesses, layers):
        """
        Compute reflectance spectrum using characteristic matrix method.
        thicknesses: array of physical thicknesses in nm
        layers: list of 'L' or 'H'
        Returns: array of reflectance (0-1) at each wavelength.
        """
        n_layers = len(layers)
        if n_layers == 0:
            # Bare substrate
            r = (n_amb - n_sub) / (n_amb + n_sub)
            return np.full(num_wl, abs(r)**2)
        
        reflectances = np.zeros(num_wl)
        for i, lam in enumerate(wavelengths):
            # Build characteristic matrix
            # M = M1 * M2 * ... * Mn
            # Each layer matrix: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
            # where delta = 2*pi*n*d/lam
            total_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
            
            for j in range(n_layers):
                n_j = n_L if layers[j] == 'L' else n_H
                d_j = thicknesses[j]
                delta_j = 2.0 * np.pi * n_j * d_j / lam
                cos_d = np.cos(delta_j)
                sin_d = np.sin(delta_j)
                
                layer_matrix = np.array([
                    [cos_d, 1j * sin_d / n_j],
                    [1j * n_j * sin_d, cos_d]
                ], dtype=complex)
                
                total_matrix = total_matrix @ layer_matrix
            
            # The admittance seen from air side
            # B = M11 + M12, C = M21 + M22
            # Y = n_amb * C / B (for normal incidence, with appropriate scaling)
            # Actually, the characteristic matrix method:
            # [B, C] = [M11, M12; M21, M22] @ [1, 0] ... no
            # Standard: if the substrate has index n_s, and we compute the matrix for all layers,
            # then the admittance at the air interface is:
            # Y = (M21 + M22 * n_s) / (M11 + M12 * n_s)
            # Wait, let me be careful. The standard formula:
            # If we have a stack of N layers on a substrate of index n_{N+1},
            # and the ambient is n_0, then:
            # [B, C] = M_total @ [1, n_{N+1}]^T ... no
            # Actually: 
            # M_total = [[M11, M12], [M21, M22]]
            # The electric and magnetic field at the first interface:
            # E_0 = M11 * E_N + M12 * H_N
            # H_0 = M21 * E_N + M22 * H_N
            # At the substrate, for a plane wave: H_N = n_{N+1} * E_N (normalized)
            # So: E_0 = (M11 + M12 * n_{N+1}) * E_N
            #       H_0 = (M21 + M22 * n_{N+1}) * E_N
            # Admittance Y = H_0 / E_0 = (M21 + M22 * n_{N+1}) / (M11 + M12 * n_{N+1})
            # Reflectance: r = (n_amb - Y) / (n_amb + Y)
            # R = |r|^2
            
            M11, M12 = total_matrix[0, 0], total_matrix[0, 1]
            M21, M22 = total_matrix[1, 0], total_matrix[1, 1]
            
            B = M11 + M12 * n_sub
            C = M21 + M22 * n_sub
            Y = C / B
            r = (n_amb - Y) / (n_amb + Y)
            reflectances[i] = np.abs(r)**2
        
        return reflectances
    
    def objective(x):
        """Objective function: mean reflectance in percent."""
        # x contains thicknesses, and we need to encode the layer structure
        # For simplicity, let's fix the layer structure and only optimize thicknesses
        # We'll try a few different structures and pick the best
        
        # Actually, let's make this more flexible: we'll try several fixed structures
        # and optimize thicknesses for each, then return the best
        
        # This is a bit awkward for minimize. Let's restructure.
        # For now, let's just do a simple approach: try a few candidate structures
        # with optimized thicknesses using a simple local search.
        
        pass
    
    # Let's try a different approach: define a few candidate layer structures
    # and for each, optimize the thicknesses using scipy minimize
    
    candidate_structures = [
        # Single layer
        ['L'],
        # Two layers
        ['L', 'H'],
        ['H', 'L'],
        # Three layers
        ['L', 'H', 'L'],
        ['H', 'L', 'H'],
        ['L', 'L', 'H'],
        ['H', 'H', 'L'],
        # Four layers
        ['L', 'H', 'L', 'H'],
        ['H', 'L', 'H', 'L'],
        ['L', 'H', 'H', 'L'],
        ['H', 'L', 'L', 'H'],
        # Five layers
        ['L', 'H', 'L', 'H', 'L'],
        ['H', 'L', 'H', 'L', 'H'],
        # Six layers
        ['L', 'H', 'L', 'H', 'L', 'H'],
        ['H', 'L', 'H', 'L', 'H', 'L'],
    ]
    
    best_reflectance = np.inf
    best_solution = None
    
    for structure in candidate_structures:
        n_layers = len(structure)
        
        # Initial guess: quarter-wave at center wavelength
        lam_center = 0.5 * (lam_min + lam_max)
        initial_thicknesses = np.array([
            lam_center / (4.0 * (n_L if structure[i] == 'L' else n_H))
            for i in range(n_layers)
        ])
        
        # Ensure initial guess is within bounds
        initial_thicknesses = np.clip(initial_thicknesses, 0, max_layer_nm)
        
        # Check total thickness constraint
        if np.sum(initial_thicknesses) > max_total_nm:
            # Scale down
            initial_thicknesses = initial_thicknesses * (max_total_nm / np.sum(initial_thicknesses))
        
        def obj(x):
            # x is the array of thicknesses
            # Check constraints
            if np.any(x < 0) or np.any(x > max_layer_nm):
                return 1.0  # Large penalty
            if np.sum(x) > max_total_nm:
                return 1.0  # Large penalty
            
            refl = compute_reflectance(x, structure)
            return np.mean(refl) * 100.0  # In percent
        
        # Use L-BFGS-B with bounds
        bounds = [(0, max_layer_nm)] * n_layers
        
        try:
            result = minimize(obj, initial_thicknesses, method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
            
            if result.success:
                final_thicknesses = result.x
                # Final check
                if np.all(final_thicknesses >= 0) and np.all(final_thicknesses <= max_layer_nm) and np.sum(final_thicknesses) <= max_total_nm:
                    refl = compute_reflectance(final_thicknesses, structure)
                    mean_refl = np.mean(refl) * 100.0
                    if mean_refl < best_reflectance:
                        best_reflectance = mean_refl
                        best_solution = (structure, final_thicknesses)
        except Exception:
            pass
    
    # If no solution found, fall back to a simple quarter-wave layer
    if best_solution is None:
        lam_center = 0.5 * (lam_min + lam_max)
        d = lam_center / (4.0 * n_L)
        return ['L'], [d]
    
    structure, thicknesses = best_solution
    return structure, list(thicknesses)

# EVOLVE-BLOCK-END
