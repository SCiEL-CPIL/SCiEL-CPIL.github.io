import numpy as np
import time
from scipy.optimize import differential_evolution

def _compute_reflectance_vectorized(thicknesses, indices, wavelengths, n0, ns):
    """
    Compute mean reflectance percentage for a stack of layers, vectorized over wavelengths.
    """
    N = len(thicknesses)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return (np.abs(r) ** 2) * 100.0

    M = np.ones((len(wavelengths), 2, 2), dtype=complex)
    
    for d, n in zip(thicknesses, indices):
        delta = 2.0 * np.pi * n * d / wavelengths
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        Y = n
        
        old_M11 = M[:, 0, 0]
        old_M12 = M[:, 0, 1]
        old_M21 = M[:, 1, 0]
        old_M22 = M[:, 1, 1]
        
        term1 = 1j * sin_d / Y
        term2 = 1j * Y * sin_d
        
        M[:, 0, 0] = old_M11 * cos_d + old_M12 * term1
        M[:, 0, 1] = old_M11 * term2 + old_M12 * cos_d
        M[:, 1, 0] = old_M21 * cos_d + old_M22 * term1
        M[:, 1, 1] = old_M21 * term2 + old_M22 * cos_d

    Ys = ns
    Y_in = (M[:, 0, 0] + M[:, 0, 1] * Ys) / (M[:, 1, 0] + M[:, 1, 1] * Ys)
    
    r = (n0 - Y_in) / (n0 + Y_in)
    R = np.abs(r) ** 2
    
    return np.mean(R) * 100.0

def design(problem):
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # 9-layer stack: L H L H L H L H L
    materials_seq = ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L']
    indices_seq = [nL, nH, nL, nH, nL, nH, nL, nH, nL]
    N_layers = len(materials_seq)
    
    def objective(d):
        total_thick = np.sum(d)
        if total_thick > max_total_nm:
            return 100.0
        if np.any(d < 0) or np.any(d > max_layer_nm):
            return 100.0
        return _compute_reflectance_vectorized(d, indices_seq, wavelengths, n0, ns)
    
    bounds = [(0.0, max_layer_nm)] * N_layers
    
    start_time = time.time()
    time_limit = 50.0
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=True,
            init='latinhypercube'
        )
        
        best_d = res.x
        best_score = res.fun
        
    except Exception as e:
        centre = 0.5 * (band[0] + band[1])
        best_d = [centre / (4.0 * nL)] * N_layers
        best_score = 100.0
        
    total_thick = np.sum(best_d)
    if total_thick > max_total_nm:
        scale = max_total_nm / total_thick
        best_d = best_d * scale
        
    best_d = np.clip(best_d, 0.0, max_layer_nm)
    
    return materials_seq, list(best_d)
