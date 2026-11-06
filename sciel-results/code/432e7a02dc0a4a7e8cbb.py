import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def design(problem):
    start_time = time.time()
    time_limit = 50.0
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def calc_reflectance_vec(layers, thicknesses, wls):
        n_stack = [nL if l == 'L' else nH for l in layers]
        M = np.array([
            [np.ones(len(wls)) + 0j, np.zeros(len(wls)) + 0j],
            [np.zeros(len(wls)) + 0j, np.ones(len(wls)) + 0j]
        ])
        
        for i in range(len(layers)):
            n = n_stack[i]
            d = thicknesses[i]
            delta = 2.0 * np.pi * n * d / wls
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            Mi00 = cos_d
            Mi01 = 1j * sin_d / n
            Mi10 = 1j * n * sin_d
            Mi11 = cos_d
            
            M00 = M[0,0] * Mi00 + M[0,1] * Mi10
            M01 = M[0,0] * Mi01 + M[0,1] * Mi11
            M10 = M[1,0] * Mi00 + M[1,1] * Mi10
            M11 = M[1,0] * Mi01 + M[1,1] * Mi11
            
            M[0,0] = M00
            M[0,1] = M01
            M[1,0] = M10
            M[1,1] = M11
            
        Y = (M[0,0] + M[0,1] * ns) / (M[1,0] + M[1,1] * ns)
        r = (Y - n0) / (Y + n0)
        return np.abs(r)**2

    best_score = float('inf')
    best_layers = None
    best_thick = None
    
    patterns = []
    for n_layers in [4, 6, 8, 10, 12]:
        if n_layers > max_layers:
            continue
        p1 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        p2 = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        patterns.append(p1)
        patterns.append(p2)
        
    for pattern in patterns:
        if time.time() - start_time > time_limit:
            break
            
        n_layers = len(pattern)
        
        def objective(thick):
            if np.sum(thick) > max_total_nm:
                return 1.0
            r_vals = calc_reflectance_vec(pattern, thick, wavelengths)
            return np.mean(r_vals) * 100.0
        
        bounds = [(0, max_layer_nm)] * n_layers
        
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=10, 
                tol=1e-6, 
                seed=42
            )
            if res.fun < best_score:
                best_score = res.fun
                best_layers = pattern
                best_thick = res.x
        except Exception:
            pass

    if best_layers is not None:
        # Polish with Nelder-Mead
        def objective_nm(thick):
            if np.sum(thick) > max_total_nm:
                return 1.0
            r_vals = calc_reflectance_vec(best_layers, thick, wavelengths)
            return np.mean(r_vals) * 100.0

        try:
            res_nm = minimize(objective_nm, best_thick, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_thick = res_nm.x
        except Exception:
            pass

    if best_layers is None:
        return ["L"], [100.0]
        
    return best_layers, list(best_thick)
