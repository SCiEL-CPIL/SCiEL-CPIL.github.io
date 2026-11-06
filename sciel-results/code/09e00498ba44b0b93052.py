import numpy as np
from scipy.optimize import minimize
import time

def _calc_reflectance(wavelengths, materials, thicknesses, n0, ns):
    """
    Calculate reflectance spectrum using characteristic matrix method.
    wavelengths: array of wavelengths in nm
    materials: list of indices
    thicknesses: list of thicknesses in nm
    n0: ambient index
    ns: substrate index
    Returns: array of reflectance values (0 to 1)
    """
    N = len(materials)
    if N == 0:
        # Bare substrate
        R = ((n0 - ns) / (n0 + ns)) ** 2
        return np.full_like(wavelengths, R, dtype=float)

    # delta_j = 2 * pi * n_j * t_j / lambda
    # Vectorized: delta (M, N)
    delta = 2.0 * np.pi * np.outer(wavelengths, np.array(materials) * np.array(thicknesses))
    
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    
    # M_total (M, 2, 2)
    M_total = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M_total[:, 0, 0] = 1.0
    M_total[:, 1, 1] = 1.0
    
    for j in range(N):
        n_j = materials[j]
        
        A = cos_delta[:, j]
        B = 1j * sin_delta[:, j] / n_j
        C = 1j * n_j * sin_delta[:, j]
        D = cos_delta[:, j]
        
        A_total = M_total[:, 0, 0]
        B_total = M_total[:, 0, 1]
        C_total = M_total[:, 1, 0]
        D_total = M_total[:, 1, 1]
        
        M_total[:, 0, 0] = A_total * A + B_total * C
        M_total[:, 0, 1] = A_total * B + B_total * D
        M_total[:, 1, 0] = C_total * A + D_total * C
        M_total[:, 1, 1] = C_total * B + D_total * D
        
    A = M_total[:, 0, 0]
    B = M_total[:, 0, 1]
    C = M_total[:, 1, 0]
    D = M_total[:, 1, 1]
    
    Y_in = (C + D * ns) / (A + B * ns)
    
    R = np.abs((n0 - Y_in) / (n0 + Y_in)) ** 2
    
    return R

def design(problem):
    start_time = time.time()
    deadline = start_time + 50.0

    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    center_lambda = 0.5 * (band[0] + band[1])
    
    candidates = [
        [n_H, n_L, n_H, n_L, n_H, n_L],
        [n_L, n_H, n_L, n_H, n_L, n_H],
        [n_H, n_L, n_H, n_L],
        [n_L, n_H, n_L, n_H],
    ]
    
    best_seq = None
    best_t = None
    best_score = float('inf')
    
    for seq in candidates:
        if time.time() > deadline:
            break
            
        t0 = [center_lambda / (4.0 * n) for n in seq]
        if sum(t0) > max_total_nm:
            scale = max_total_nm / sum(t0)
            t0 = [t * scale for t in t0]
            
        bounds = [(0.0, max_layer_nm) for _ in range(len(t0))]
        
        def obj(t_vec, seq=seq):
            if sum(t_vec) > max_total_nm:
                penalty = (sum(t_vec) - max_total_nm) * 100.0
            else:
                penalty = 0.0
            R = _calc_reflectance(wavelengths, seq, t_vec, n0, ns)
            return np.mean(R) * 100.0 + penalty
            
        try:
            res = minimize(obj, t0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_seq = seq
                best_t = res.x
        except Exception:
            score_init = obj(t0)
            if score_init < best_score:
                best_score = score_init
                best_seq = seq
                best_t = t0

    if best_seq is None:
        return ["L"], [center_lambda / (4.0 * n_L)]
        
    labels = []
    for n in best_seq:
        if np.isclose(n, n_L):
            labels.append("L")
        else:
            labels.append("H")
            
    return labels, list(best_t)
