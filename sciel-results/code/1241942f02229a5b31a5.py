"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time
import math

def _reflectance_spectrum(materials, thicknesses, band_nm, ambient_index, substrate_index):
    """
    Compute the reflectance spectrum (in percent) using the characteristic-matrix method.
    materials: dict with 'L' and 'H' indices
    thicknesses: list of physical thicknesses in nm
    band_nm: [w_min, w_max]
    ambient_index: float
    substrate_index: float
    Returns: array of reflectance percentages at 401 equally spaced wavelengths.
    """
    w_min, w_max = band_nm
    wavelengths = np.linspace(w_min, w_max, 401)
    n_layers = len(materials)
    
    # We need to know which layers are L and which are H.
    # Actually, the input to design returns (materials_labels, thicknesses).
    # So we need to pass the labels too to this function or reconstruct.
    # Let's modify the helper to take labels as well.
    pass

def _compute_reflectance(labels, thicknesses, wavelengths, materials_dict, n_0, n_s):
    """
    labels: list of 'L' or 'H'
    thicknesses: list of floats (nm)
    wavelengths: array of wavelengths (nm)
    materials_dict: {'L': 1.38, 'H': 2.35}
    n_0: ambient index
    n_s: substrate index
    Returns: array of reflectance percentages.
    """
    n_layers = len(labels)
    R = np.zeros(len(wavelengths))
    
    # Pre-compute indices for each layer
    n_layers_arr = np.array([materials_dict[lbl] for lbl in labels])
    d_arr = np.array(thicknesses)
    
    for i, w in enumerate(wavelengths):
        k = 2 * np.pi / w  # 2*pi / lambda
        
        # Transfer matrix for each layer
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
        # For non-magnetic, non-absorbing media
        # delta_j = (2 * pi / lambda) * n_j * d_j
        
        M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = n_layers_arr[j]
            d_j = d_arr[j]
            delta_j = (2 * np.pi / w) * n_j * d_j
            
            c = np.cos(delta_j)
            s = np.sin(delta_j)
            
            M_j = np.array([
                [c, 1j * s / n_j],
                [1j * n_j * s, c]
            ], dtype=complex)
            
            M_total = M_total @ M_j
        
        # Characteristic matrix M_total = [[M11, M12], [M21, M22]]
        M11, M12 = M_total[0, 0], M_total[0, 1]
        M21, M22 = M_total[1, 0], M_total[1, 1]
        
        # Reflectance
        # Y = (n_0 * M21 + M22) / (n_0 * M11 + M12)
        # Actually, the admittance Y is defined such that:
        # [E]   [M11 M12] [E_s]
        # [H] = [M21 M22] [H_s]
        # At the substrate side, E_s / H_s = n_s (for normal incidence, TE/TM same)
        # So H_s = E_s / n_s
        # [E]   [M11 M12] [1]
        # [H] = [M21 M22] [1/n_s]
        # E = M11 + M12/n_s
        # H = M21 + M22/n_s
        # Y = E/H = (M11 + M12/n_s) / (M21 + M22/n_s)
        
        Y = (M11 + M12 / n_s) / (M21 + M22 / n_s)
        
        # Reflectance R = |(n_0 - Y)/(n_0 + Y)|^2
        r = (n_0 - Y) / (n_0 + Y)
        R[i] = np.abs(r) ** 2 * 100.0
        
    return R

def _objective(x, problem, labels, materials_dict, n_0, n_s, wavelengths):
    """
    Objective function for optimization.
    x: array of thicknesses
    Returns: mean reflectance
    """
    thicknesses = list(x)
    try:
        R = _compute_reflectance(labels, thicknesses, wavelengths, materials_dict, n_0, n_s)
        return np.mean(R)
    except Exception:
        return 100.0

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s limit
    
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    w_min, w_max = band_nm
    wavelengths = np.linspace(w_min, w_max, 401)
    
    # Try different numbers of layers and structures
    best_score = 100.0
    best_labels = None
    best_thicknesses = None
    
    # Start with a reasonable initial guess: quarter-wave stack
    # For a broadband AR coating, a common starting point is a quarter-wave stack
    # centered at the geometric mean wavelength
    w_center = math.sqrt(w_min * w_max)
    
    # Try various layer counts
    for n_layers in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if time.time() > deadline:
            break
            
        # Generate a starting structure: alternating L and H, starting with L
        labels = []
        for i in range(n_layers):
            if i % 2 == 0:
                labels.append('L')
            else:
                labels.append('H')
        
        # Initial thicknesses: quarter-wave at center wavelength
        n_L = materials_dict['L']
        n_H = materials_dict['H']
        init_thicknesses = []
        for lbl in labels:
            n = materials_dict[lbl]
            d = w_center / (4.0 * n)
            # Clamp to max_layer_nm
            d = min(d, max_layer_nm)
            init_thicknesses.append(d)
        
        # Check if total thickness exceeds max
        if sum(init_thicknesses) > max_total_nm:
            # Scale down
            scale = max_total_nm / sum(init_thicknesses)
            init_thicknesses = [d * scale for d in init_thicknesses]
        
        # Define bounds for differential evolution
        bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
        
        # Use differential evolution
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(problem, labels, materials_dict, n_0, n_s, wavelengths),
                seed=42,
                maxiter=50,
                tol=1e-5,
                polish=False,  # Skip Nelder-Mead for speed
                update_interval=10,
                callback=None
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_labels = labels
                best_thicknesses = list(result.x)
                
        except Exception as e:
            pass
    
    # If no valid result found, return a simple quarter-wave layer
    if best_labels is None:
        centre = 0.5 * (w_min + w_max)
        return ["L"], [centre / (4.0 * materials_dict["L"])]
    
    # Check constraints one more time
    if sum(best_thicknesses) > max_total_nm:
        # Scale down
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [d * scale for d in best_thicknesses]
    
    # Clamp individual layers
    best_thicknesses = [min(max(d, 0.0), max_layer_nm) for d in best_thicknesses]
    
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
