"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _characteristic_matrix(n, d, lam, theta=0.0):
    """
    Compute the 2x2 characteristic matrix for a single layer.
    n: refractive index
    d: physical thickness in nm
    lam: wavelength in nm
    theta: angle of incidence (0 for normal)
    """
    # Optical thickness in radians
    beta = 2.0 * np.pi * n * d * np.cos(theta) / lam
    # For normal incidence, cos(theta) = 1
    if theta == 0.0:
        cos_theta = 1.0
    else:
        cos_theta = np.cos(theta)
    
    # For normal incidence, the characteristic matrix is:
    # [cos(beta), i*sin(beta)/n]
    # [i*n*sin(beta), cos(beta)]
    # But we work with the transfer matrix in a way that's convenient for reflectance.
    
    # Actually, let's use the standard form:
    # M = [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # where delta = 2*pi*n*d*cos(theta)/lambda
    
    delta = 2.0 * np.pi * n * d * cos_theta / lam
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    
    M = np.array([
        [cos_d, 1j * sin_d / n],
        [1j * n * sin_d, cos_d]
    ], dtype=complex)
    
    return M

def _reflectance_stack(materials, thicknesses, n_air, n_sub, lam):
    """
    Compute reflectance for a stack of layers at a given wavelength.
    materials: list of 'L' or 'H'
    thicknesses: list of physical thicknesses in nm
    n_air: ambient index
    n_sub: substrate index
    lam: wavelength in nm
    """
    if len(materials) == 0:
        # Bare substrate
        r = (n_air - n_sub) / (n_air + n_sub)
        return abs(r) ** 2
    
    # Build the total characteristic matrix
    # Start with the substrate side
    # The total matrix M_total = M_1 * M_2 * ... * M_N
    # where M_i is the characteristic matrix of layer i
    
    M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    for i, (mat, d) in enumerate(zip(materials, thicknesses)):
        n = problem_global["materials"][mat]
        M_i = _characteristic_matrix(n, d, lam)
        M_total = M_total @ M_i
    
    # The reflectance is computed from the total matrix
    # r = (n_air * M_total[0,0] + n_air * M_total[0,1] * n_sub) / (n_air * M_total[0,0] - n_air * M_total[0,1] * n_sub)
    # Wait, let me recall the formula.
    
    # For a stack between medium 0 (air) and medium s (substrate):
    # The characteristic matrix of the whole stack is M = [[m11, m12], [m21, m22]]
    # Then the effective admittance Y is:
    # Y = (m21 + m22 * n_sub) / (m11 + m12 * n_sub)
    # Wait, that's not quite right. Let me think again.
    
    # The standard formula: 
    # The total characteristic matrix M_total relates the fields at the first interface to the last.
    # For normal incidence:
    # [E_0, H_0] = M_total @ [E_s, H_s]
    # where E and H are the electric and magnetic field amplitudes.
    
    # Actually, the reflectance is:
    # r = (n_air * (M_total[0,0] + M_total[0,1] * n_sub) - (M_total[1,0] + M_total[1,1] * n_sub)) / 
    #     (n_air * (M_total[0,0] + M_total[0,1] * n_sub) + (M_total[1,0] + M_total[1,1] * n_sub))
    
    # Let me use a cleaner formulation. The characteristic matrix M for the stack satisfies:
    # [E]   = M [E]
    # [H]       [H]
    # where the left side is at the air-side interface and the right side is at the substrate-side interface.
    
    # For a single layer with index n and optical thickness delta:
    # M = [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    
    # The reflection coefficient for the stack is:
    # r = (n_air * M[0,0] - M[1,0] - n_air * M[0,1] * n_sub + M[1,1] * n_sub) / 
    #     (n_air * M[0,0] + M[1,0] + n_air * M[0,1] * n_sub + M[1,1] * n_sub)
    
    # Hmm, let me just use a known correct formula.
    
    # Actually, the standard approach:
    # Let M = M_1 M_2 ... M_N be the total characteristic matrix.
    # Then the reflection coefficient is:
    # r = (n_air * (M[0,0] - M[0,1] * n_sub) - (M[1,0] - M[1,1] * n_sub)) / 
    #     (n_air * (M[0,0] - M[0,1] * n_sub) + (M[1,0] - M[1,1] * n_sub))
    
    # Wait, I think I'm overcomplicating this. Let me use a simpler method.
    
    # The characteristic matrix for the stack is M_total.
    # The effective input admittance Y_in is:
    # Y_in = (M_total[1,0] + M_total[1,1] * n_sub) / (M_total[0,0] + M_total[0,1] * n_sub)
    
    # Then the reflection coefficient is:
    # r = (n_air - Y_in) / (n_air + Y_in)
    
    # And reflectance R = |r|^2
    
    Y_in = (M_total[1,0] + M_total[1,1] * n_sub) / (M_total[0,0] + M_total[0,1] * n_sub)
    r = (n_air - Y_in) / (n_air + Y_in)
    return abs(r) ** 2

# We need to store the problem globally for the helper functions
problem_global = None

def design(problem):
    """
    Design a broadband antireflection coating using a stack of low and high index layers.
    """
    global problem_global
    problem_global = problem
    
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Strategy: Use a multilayer quarter-wave stack.
    # A common approach for broadband AR coatings is to use a stack of quarter-wave layers
    # with alternating low and high index materials.
    
    # Let's try a symmetric stack design.
    # For a broadband AR coating, we can use a stack like:
    # L H L H L H L H L H ... (or similar)
    
    # The key insight is that quarter-wave layers at the center wavelength will have
    # anti-reflection properties. For broadband performance, we need multiple layers.
    
    # Let's try a design with 10 layers (5 L-H pairs) or similar.
    
    # Center wavelength
    lam_c = 0.5 * (band_nm[0] + band_nm[1])
    
    # Quarter-wave thickness at center wavelength
    d_L_q = lam_c / (4.0 * n_L)
    d_H_q = lam_c / (4.0 * n_H)
    
    # Try different numbers of layers and optimize
    best_materials = None
    best_thicknesses = None
    best_mean_R = float('inf')
    
    # Try stacks with 2, 4, 6, 8, 10 layers
    for n_layers in [2, 4, 6, 8, 10, 12, 14, 16]:
        if n_layers > max_layers:
            break
        
        # For a symmetric stack, we can try patterns like:
        # L H L H ... or H L H L ...
        # Let's try both starting with L and starting with H
        
        for start_mat in ['L', 'H']:
            # Create a symmetric pattern
            # For n_layers layers, the pattern is determined by the first half
            half = n_layers // 2
            remainder = n_layers % 2
            
            # Build the pattern
            pattern = []
            current_mat = start_mat
            for i in range(n_layers):
                if i < half:
                    pattern.append(current_mat)
                    current_mat = 'H' if current_mat == 'L' else 'L'
                elif i == half and remainder == 0:
                    # Even number of layers, mirror
                    for j in range(half - 1, -1, -1):
                        pattern.append(pattern[j])
                    break
                else:
                    # Odd number of layers, add center layer then mirror
                    pattern.append(current_mat)
                    current_mat = 'H' if current_mat == 'L' else 'L'
                    for j in range(half - 1, -1, -1):
                        pattern.append(pattern[j])
                    break
            
            # Ensure pattern has correct length
            if len(pattern) != n_layers:
                continue
            
            # Now optimize the thicknesses
            # Use a simple optimization: start with quarter-wave values and adjust
            
            # Initial guess: all quarter-wave
            if start_mat == 'L':
                init_thicknesses = [d_L_q if p == 'L' else d_H_q for p in pattern]
            else:
                init_thicknesses = [d_H_q if p == 'H' else d_L_q for p in pattern]
            
            # Clamp to valid range
            init_thicknesses = [max(0, min(t, max_layer_nm)) for t in init_thicknesses]
            
            # Check total thickness
            total_t = sum(init_thicknesses)
            if total_t > max_total_nm:
                # Scale down
                scale = max_total_nm / total_t
                init_thicknesses = [t * scale for t in init_thicknesses]
            
            # Evaluate
            mean_R = _evaluate_stack(pattern, init_thicknesses, n_air, n_sub, wavelengths)
            
            if mean_R < best_mean_R:
                best_mean_R = mean_R
                best_materials = pattern
                best_thicknesses = init_thicknesses
    
    # If we didn't find a good solution, fall back to a single layer
    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [d_L_q]
    
    return best_materials, best_thicknesses

def _evaluate_stack(materials, thicknesses, n_air, n_sub, wavelengths):
    """Evaluate mean reflectance for a given stack."""
    total_R = 0.0
    for lam in wavelengths:
        R = _reflectance_stack(materials, thicknesses, n_air, n_sub, lam)
        total_R += R
    return total_R / len(wavelengths) * 100.0  # in percent
# EVOLVE-BLOCK-END
