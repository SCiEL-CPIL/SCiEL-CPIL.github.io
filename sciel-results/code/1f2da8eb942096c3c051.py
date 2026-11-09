"""Design broadband antireflection coating using quarter-wave stack optimization."""
# EVOLVE-BLOCK-START
import numpy as np

def _reflectance_spectrum(thicknesses, indices, n_layers, wavelengths, ambient, substrate):
    """
    Compute normal-incidence reflectance using characteristic matrix method.
    Returns reflectance as a fraction (0 to 1).
    """
    n = n_layers
    # Build the total characteristic matrix M = M_1 * M_2 * ... * M_n
    # Each layer i has matrix:
    #   [[cos(delta_i), i*sin(delta_i)/n_i],
    #    [i*n_i*sin(delta_i), cos(delta_i)]]
    # where delta_i = 2*pi*n_i*d_i / wavelength
    
    # M_total = product of all layer matrices
    # We can compute this iteratively
    
    # Initialize M as identity
    m11 = 1.0 + 0.0j
    m12 = 0.0 + 0.0j
    m21 = 0.0 + 0.0j
    m22 = 1.0 + 0.0j
    
    for i in range(n):
        ni = indices[i]
        di = thicknesses[i]
        delta = 2.0 * np.pi * ni * di / wavelengths
        cd = np.cos(delta)
        sd = np.sin(delta)
        
        # Layer matrix
        li11 = cd
        li12 = 1j * sd / ni
        li21 = 1j * ni * sd
        li22 = cd
        
        # M_new = M_old * M_layer
        nm11 = m11 * li11 + m12 * li21
        nm12 = m11 * li12 + m12 * li22
        nm21 = m21 * li11 + m22 * li21
        nm22 = m21 * li12 + m22 * li22
        
        m11, m12, m21, m22 = nm11, nm12, nm21, nm22
    
    # The optical admittance Y = n_s * (m11 + m12) / (m21 + m22) ... wait
    # Actually, the characteristic matrix relates:
    # [E, H]_0 = M * [E, H]_N
    # where N is the substrate side.
    # For a stack on substrate with index n_s, the boundary condition at the last interface gives:
    # Y = n_s * (m22 + m21) / (m12 + m11) ... let me recall the exact formula.
    
    # The standard formula: 
    # The effective admittance of the stack as seen from air is:
    # Y = n_s * (m11 + m12) / (m21 + m22)
    # Wait, I need to be careful. Let me use the standard result.
    
    # Actually, the characteristic matrix M satisfies:
    # [E_0]   = M * [E_N]
    # [H_0]       [H_N]
    # where E_N and H_N are at the substrate interface.
    
    # At the substrate, the admittance is n_s (for normal incidence, lossless):
    # H_N = (n_s / n_0) * E_N ... no, H = (n/n_0) * E for the forward wave only.
    
    # More precisely, if we define H such that H = n * E for the forward wave in a medium of index n,
    # then at the substrate interface (last interface), we have a single forward wave (no back reflection from infinity),
    # so H_N = n_s * E_N.
    
    # Thus: [E_0; H_0] = M * [E_N; n_s * E_N] = E_N * [m11 + m12*n_s; m21 + m22*n_s]
    
    # The admittance at the air side is Y_0 = H_0 / E_0 = (m21 + m22*n_s) / (m11 + m12*n_s)
    
    # The reflectance is then:
    # r = (Y_0 - n_air) / (Y_0 + n_air)
    # R = |r|^2
    
    y0 = (m21 + m22 * substrate) / (m11 + m12 * substrate)
    r = (y0 - ambient) / (y0 + ambient)
    R = np.abs(r) ** 2
    return R

def design(problem):
    """
    Design a broadband antireflection coating.
    Uses a multi-layer quarter-wave stack optimized for the band.
    """
    ambient = problem["ambient_index"]
    substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wl_min, wl_max = band[0], band[1]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    wavelengths = np.linspace(wl_min, wl_max, 401)
    
    # Strategy: Use a symmetric stack of alternating L and H layers.
    # A common effective design is a "quarter-wave" stack where each layer is ~lambda/4 at the center wavelength.
    # For broadband AR, we can use a stack like: H L H L ... H or L H L H ... L
    # The number of layers should be odd or even depending on the desired impedance transformation.
    
    # Let's try a few candidate designs and pick the best one.
    # Candidate 1: Single L layer (baseline, but we can do better)
    # Candidate 2: H-L-H or L-H-L triple stack
    # Candidate 3: 5-layer stack
    # Candidate 4: 7-layer stack
    # etc.
    
    # For a quarter-wave stack, the thickness of each layer is lambda_center / (4 * n_i)
    # But for broadband performance, we may want to adjust thicknesses.
    
    # Let's try a simple approach: use a symmetric stack of N layers, alternating H and L,
    # with each layer thickness set to lambda_center / (4 * n_i).
    # Then evaluate and pick the best N.
    
    wl_center = 0.5 * (wl_min + wl_max)
    
    best_R = np.inf
    best_design = None
    
    # Try stacks with 1 to max_layers layers
    for n_layers in range(1, max_layers + 1):
        # Alternate starting with H or L
        for start_char in ['H', 'L']:
            # Build the layer sequence
            labels = []
            for i in range(n_layers):
                if i % 2 == 0:
                    labels.append(start_char)
                else:
                    labels.append('L' if start_char == 'H' else 'H')
            
            # Compute thicknesses: quarter-wave at center wavelength
            thicknesses = []
            for label in labels:
                n_i = n_H if label == 'H' else n_L
                d_i = wl_center / (4.0 * n_i)
                # Check constraint
                if d_i > max_layer_nm:
                    break
                thicknesses.append(d_i)
            
            if len(thicknesses) != n_layers:
                continue
            
            total = sum(thicknesses)
            if total > max_total_nm:
                continue
            
            # Compute reflectance
            indices = [n_H if l == 'H' else n_L for l in labels]
            R = _reflectance_spectrum(thicknesses, indices, n_layers, wavelengths, ambient, substrate)
            mean_R = np.mean(R)
            
            if mean_R < best_R:
                best_R = mean_R
                best_design = (labels, thicknesses)
    
    # If no valid design found, fall back to single L layer
    if best_design is None:
        d = wl_center / (4.0 * n_L)
        return ["L"], [d]
    
    return best_design

# EVOLVE-BLOCK-END
