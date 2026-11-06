import numpy as np
from scipy.optimize import minimize
import time

def _mean_reflectance(thicknesses, materials_seq, problem):
    n0 = problem["ambient_index"]
    nS = problem["substrate_index"]
    materials = problem["materials"]
    band = problem["band_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    reflectances = np.zeros(len(wavelengths))
    
    for idx, lam in enumerate(wavelengths):
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for i, mat_label in enumerate(materials_seq):
            n = materials[mat_label]
            d = thicknesses[i]
            delta = 2.0 * np.pi * n * d / lam
            c = np.cos(delta)
            s = np.sin(delta)
            Mi = np.array([[c, 1j * s / n], [1j * n * s, c]], dtype=complex)
            M = M @ Mi
            
        B = M[0, 0] + M[0, 1] * nS
        C = M[1, 0] + M[1, 1] * nS
        Y = B / C
        Gamma = (n0 - Y) / (n0 + Y)
        reflectances[idx] = np.abs(Gamma) ** 2
        
    return np.mean(reflectances)

def design(problem):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n0 = problem["ambient_index"]
    nS = problem["substrate_index"]
    materials = problem["materials"]
    band = problem["band_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambda_c = 0.5 * (band[0] + band[1])
    
    candidates = []
    for n_layers in [3, 5, 7, 9]:
        seq_L = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        seq_H = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        candidates.append(seq_L)
        candidates.append(seq_H)
        
    best_score = float('inf')
    best_design = None
    
    # Fallback: single layer
    init_thick = [lambda_c / (4.0 * materials['L'])]
    init_score = _mean_reflectance(init_thick, ['L'], problem)
    best_score = init_score
    best_design = (['L'], init_thick)
    
    for seq in candidates:
        if time.time() > deadline:
            break
            
        n_layers = len(seq)
        x0 = []
        for label in seq:
            n = materials[label]
            d = lambda_c / (4.0 * n)
            x0.append(d)
        
        bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
        
        def objective(x):
            if np.sum(x) > max_total_nm:
                return _mean_reflectance(x, seq, problem) + 1.0
            return _mean_reflectance(x, seq, problem)
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 150, 'ftol': 1e-9, 'gtol': 1e-7})
            
            if res.success:
                x_opt = res.x
                if np.sum(x_opt) <= max_total_nm:
                    score = _mean_reflectance(x_opt, seq, problem)
                    if score < best_score:
                        best_score = score
                        best_design = (seq, [float(t) for t in x_opt])
                        
        except Exception:
            pass
            
    return best_design
