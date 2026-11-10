import numpy as np

def design(problem):
    """
    Designs a broadband antireflection coating using a search over
    geometric progression thickness profiles.
    """
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    nL = materials["L"]
    nH = materials["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    # 401 equally spaced points
    wavelengths = np.linspace(band[0], band[1], 401)
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Search parameters
    # Number of layers
    N_range = [3, 5, 7, 9, 11, 13, 15]
    # Ratios for geometric progression of optical thickness
    r_range = np.arange(0.3, 0.95, 0.05)
    # Start types: 'L' or 'H'
    start_types = ['L', 'H']
    
    for N in N_range:
        if N > max_layers:
            continue
        for start in start_types:
            # Determine material sequence
            mats = []
            for i in range(N):
                if i % 2 == 0:
                    mats.append(start)
                else:
                    mats.append('H' if start == 'L' else 'L')
            
            # Convert to indices
            n_seq = [nL if m == 'L' else nH for m in mats]
            
            for r in r_range:
                # Optical thicknesses t_i = C * r^i
                # We want to scale C such that total physical thickness is close to max_total_nm
                # d_i = t_i / n_i = C * r^i / n_i
                # sum(d_i) = C * sum(r^i / n_i) <= max_total_nm
                # Also d_i <= max_layer_nm
                
                # Calculate sum of 1/n_i * r^i
                sum_inv_n = 0.0
                for i in range(N):
                    sum_inv_n += (r ** i) / n_seq[i]
                
                # Max C allowed by total thickness
                C_max_total = max_total_nm / sum_inv_n
                
                # Max C allowed by single layer thickness
                # d_i = C * r^i / n_i <= max_layer_nm
                # C <= max_layer_nm * n_i / r^i
                C_max_layer = float('inf')
                for i in range(N):
                    limit = max_layer_nm * n_seq[i] / (r ** i)
                    if limit < C_max_layer:
                        C_max_layer = limit
                
                C = min(C_max_total, C_max_layer)
                
                # Generate thicknesses
                thicknesses = []
                for i in range(N):
                    t_opt = C * (r ** i)
                    d_phys = t_opt / n_seq[i]
                    thicknesses.append(d_phys)
                
                # Verify constraints just in case
                if sum(thicknesses) > max_total_nm + 1e-6:
                    continue
                if any(d > max_layer_nm + 1e-6 for d in thicknesses):
                    continue
                
                # Calculate Reflectance
                # M = product of layer matrices
                # M_i = [[cos(d_i), i*sin(d_i)/n_i], [i*n_i*sin(d_i), cos(d_i)]]
                # d_i = 2*pi*n_i*thickness_i / lambda
                
                total_R = 0.0
                for lam in wavelengths:
                    M11, M12, M21, M22 = 1.0, 0.0, 0.0, 1.0
                    for i in range(N):
                        n_i = n_seq[i]
                        d_i = thicknesses[i]
                        delta = 2.0 * np.pi * n_i * d_i / lam
                        c = np.cos(delta)
                        s = np.sin(delta)
                        
                        m11 = c
                        m12 = 1j * s / n_i
                        m21 = 1j * n_i * s
                        m22 = c
                        
                        # Matrix multiplication: M = M * m_i
                        # New M11 = M11*m11 + M12*m21
                        # New M12 = M11*m12 + M12*m22
                        # New M21 = M21*m11 + M22*m21
                        # New M22 = M21*m12 + M22*m22
                        
                        new_M11 = M11 * m11 + M12 * m21
                        new_M12 = M11 * m12 + M12 * m22
                        new_M21 = M21 * m11 + M22 * m21
                        new_M22 = M21 * m12 + M22 * m22
                        
                        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
                    
                    # Admittance Y
                    # Y = (M21 + M22*ns) / (M11 + M12*ns)
                    denom = M11 + M12 * ns
                    if abs(denom) < 1e-12:
                        R = 1.0 # Should not happen
                    else:
                        Y = (M21 + M22 * ns) / denom
                        r_coeff = (n0 - Y) / (n0 + Y)
                        R = np.abs(r_coeff) ** 2
                    
                    total_R += R
                
                mean_R = total_R / len(wavelengths)
                
                if mean_R < best_score:
                    best_score = mean_R
                    best_materials = mats[:]
                    best_thicknesses = thicknesses[:]
    
    # Fallback to seed if something went wrong (should not happen)
    if best_materials is None:
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        return ["L"], [centre / (4.0 * problem["materials"]["L"])]
        
    return best_materials, best_thicknesses
