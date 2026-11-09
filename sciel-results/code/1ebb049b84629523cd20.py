"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _reflectance_spectrum(materials, thicknesses, band_nm, ambient_index, substrate_index):
    """
    Compute the normal-incidence reflectance spectrum using the characteristic-matrix method.
    Returns reflectance in percent for each wavelength in band_nm (401 points).
    """
    n_l = materials['L']
    n_h = materials['H']
    
    # Generate 401 equally spaced wavelengths
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    n_waves = len(wavelengths)
    
    # Number of layers
    n_layers = len(materials_list) if False else len(thicknesses)
    
    # We need to know the material labels. The function signature only gets thicknesses,
    # but in the design function we know the labels. Let's pass labels too or restructure.
    # Actually, the evaluator calls design(problem) which returns (materials, thicknesses).
    # The internal helper needs to know which layer is which.
    # Let's make the helper accept labels as well.
    
    # For now, this is a placeholder. The actual computation will be done in a nested function
    # inside design where we have access to the labels.
    pass

def _compute_reflectance(labels, thicknesses, band_nm, ambient_index, substrate_index, materials_dict):
    """
    Compute mean reflectance in percent for a given stack.
    """
    n_l = materials_dict['L']
    n_h = materials_dict['H']
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Precompute indices for each layer
    n_layers = len(labels)
    layer_indices = np.zeros(n_layers)
    for i, lab in enumerate(labels):
        layer_indices[i] = materials_dict[lab]
    
    # Matrix M = product of layer matrices
    # For each wavelength, compute the characteristic matrix
    # M_total = M_1 * M_2 * ... * M_N
    # Each layer matrix M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
    # where delta_i = 2*pi*n_i*d_i / lambda
    
    # We can vectorize over wavelengths
    # But since n_layers is small (<=20) and n_waves=401, we can loop over wavelengths or vectorize.
    # Let's vectorize over wavelengths for speed.
    
    # Initialize M_total as identity matrix for each wavelength
    # Shape: (n_waves, 2, 2)
    M_total = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M_total[:, 0, 0] = 1.0
    M_total[:, 0, 1] = 0.0
    M_total[:, 1, 0] = 0.0
    M_total[:, 1, 1] = 1.0
    
    for i in range(n_layers):
        n_i = layer_indices[i]
        d_i = thicknesses[i]
        
        # delta_i = 2*pi*n_i*d_i / lambda
        delta = 2.0 * np.pi * n_i * d_i / wavelengths  # shape (n_waves,)
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Layer matrix M_i
        # [[cos_d, 1j*sin_d/n_i], [1j*n_i*sin_d, cos_d]]
        
        # M_total = M_total @ M_i
        # M_total[:, 0, 0] = M_total[:, 0, 0]*cos_d + M_total[:, 0, 1]*(1j*sin_d/n_i)
        # M_total[:, 0, 1] = M_total[:, 0, 0]*... wait, need to be careful with in-place updates.
        
        # Let's compute the new M_total
        M00_new = M_total[:, 0, 0] * cos_d + M_total[:, 0, 1] * (1j * sin_d / n_i)
        M01_new = M_total[:, 0, 0] * (1j * n_i * sin_d) + M_total[:, 0, 1] * cos_d
        M10_new = M_total[:, 1, 0] * cos_d + M_total[:, 1, 1] * (1j * sin_d / n_i)
        M11_new = M_total[:, 1, 0] * (1j * n_i * sin_d) + M_total[:, 1, 1] * cos_d
        
        M_total[:, 0, 0] = M00_new
        M_total[:, 0, 1] = M01_new
        M_total[:, 1, 0] = M10_new
        M_total[:, 1, 1] = M11_new
    
    # Now compute reflectance
    # n_s = substrate_index
    # n_0 = ambient_index
    # R = |(M11 - n_s*M12)/(M11 + n_s*M12) * ...| 
    # Actually, the formula is:
    # Y = (M11 + M12*n_s) / (M21 + M22*n_s)  [admittance]
    # R = |(Y - n_0)/(Y + n_0)|^2
    
    n_s = substrate_index
    n_0 = ambient_index
    
    Y = (M_total[:, 0, 0] + M_total[:, 0, 1] * n_s) / (M_total[:, 1, 0] + M_total[:, 1, 1] * n_s)
    
    R = np.abs((Y - n_0) / (Y + n_0)) ** 2
    
    # Convert to percent
    R_percent = R * 100.0
    
    return np.mean(R_percent)

def design(problem):
    """
    Design a stack of at most twenty low- and high-index layers (1.38 and 2.35) on glass
    that minimizes the mean reflectance between 400 and 800 nm at normal incidence.
    """
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Strategy: Use a periodic sequence of alternating L and H layers.
    # Parameterize by:
    # 1. Number of layers N (odd, to start and end with same type, or even)
    #    Actually, let's try a symmetric stack: L H L H ... L or H L H L ... H
    #    To enforce smooth transition, we can vary the thicknesses linearly.
    #    Let's define the stack as alternating L and H.
    #    Let the number of pairs be P. Total layers N = 2*P or 2*P+1.
    #    Let's try N up to 20.
    
    # The directive says: "Represent the coating stack as a periodic sequence of alternating L and H layers, 
    # parameterized by a single base thickness and a linear gradient factor to enforce a smooth transition 
    # from low to high optical density, then optimize these two scalar parameters using a global differential evolution algorithm."
    
    # So we have 2 parameters: base_thickness (t0) and gradient_factor (g).
    # The thickness of the i-th layer (0-indexed) is t_i = t0 * (1 + g * i) or something similar.
    # But we need to ensure t_i is within [0, max_layer_nm] and sum(t_i) <= max_total_nm.
    
    # Let's define the stack structure first.
    # We'll try different numbers of layers and find the best one.
    # But differential evolution needs a fixed dimension.
    # We can loop over N (number of layers) from 1 to 20, and for each N, run DE on (t0, g).
    
    # However, running DE 20 times might be slow. 60s CPU budget.
    # Let's try a few representative N values, e.g., 5, 7, 9, 11, 13, 15, 17, 19.
    # Or we can just pick a good N, like 11 or 13, and optimize t0 and g.
    
    # Let's try N = 11 (odd, starts and ends with L, assuming we start with L).
    # Labels: L H L H L H L H L H L
    # Thicknesses: t_i = t0 * (1 + g * i) for i=0..10
    # Constraints:
    # 0 <= t_i <= max_layer_nm
    # sum(t_i) <= max_total_nm
    
    # Let's define the objective function for a fixed N and label pattern.
    
    def objective(params, N, labels):
        t0, g = params
        # Compute thicknesses
        thicknesses = np.zeros(N)
        for i in range(N):
            t_i = t0 * (1.0 + g * i)
            if t_i < 0 or t_i > max_layer_nm:
                return 100.0  # Penalty
            thicknesses[i] = t_i
        
        total = np.sum(thicknesses)
        if total > max_total_nm:
            return 100.0  # Penalty
        
        # Compute reflectance
        return _compute_reflectance(labels, thicknesses, band_nm, ambient_index, substrate_index, materials_dict)
    
    # Try different N values
    best_result = (100.0, None, None)  # (score, labels, thicknesses)
    
    # We'll try N = 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
    # For each N, define the label pattern.
    # Let's start with L and alternate.
    for N in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
        labels = ['L' if i % 2 == 0 else 'H' for i in range(N)]
        
        # Bounds for t0 and g
        # t0 > 0, g can be negative or positive.
        # To keep t_i in [0, 400], we need to bound g carefully.
        # If g is positive, t_{N-1} = t0*(1 + g*(N-1)) <= 400
        # If g is negative, t_0 = t0 >= 0, and t_{N-1} = t0*(1 + g*(N-1)) >= 0 => 1 + g*(N-1) >= 0 => g >= -1/(N-1)
        # Let's set bounds:
        # t0 in [10, 200] (nm)
        # g in [-0.5, 0.5]
        # This is a heuristic. We might need to adjust.
        
        bounds = [(10.0, 200.0), (-0.5, 0.5)]
        
        try:
            result = differential_evolution(objective, bounds, args=(N, labels), 
                                             maxiter=100, popsize=10, tol=1e-4, 
                                             seed=42, polish=False)
            if result.fun < best_result[0]:
                t0, g = result.x
                thicknesses = [t0 * (1.0 + g * i) for i in range(N)]
                # Check constraints again
                valid = True
                for t in thicknesses:
                    if t < 0 or t > max_layer_nm:
                        valid = False
                        break
                if np.sum(thicknesses) > max_total_nm:
                    valid = False
                if valid:
                    best_result = (result.fun, labels, thicknesses)
        except Exception as e:
            continue
    
    if best_result[1] is not None:
        return best_result[1], best_result[2]
    else:
        # Fallback: single quarter-wave layer
        centre = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [centre / (4.0 * materials_dict['L'])]

# EVOLVE-BLOCK-END
