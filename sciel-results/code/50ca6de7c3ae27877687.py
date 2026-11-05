import numpy as np
import time
from scipy.optimize import minimize

def design(problem):
    """
    Design an antireflection coating stack.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    def calc_reflectance(thicknesses, materials_seq):
        """
        Calculate mean reflectance for a given stack.
        """
        n_stack = [n_L if m == 'L' else n_H for m in materials_seq]
        
        d = np.array(thicknesses)
        n_arr = np.array(n_stack)
        
        # Check constraints
        if np.any(d < 0) or np.any(d > max_layer_nm):
            return 1.0
        if np.sum(d) > max_total_nm:
            return 1.0
            
        deltas = 2.0 * np.pi * n_arr[:, None] * d[:, None] / wavelengths[None, :]
        
        cos_d = np.cos(deltas)
        sin_d = np.sin(deltas)
        
        M_tot = np.tile(np.eye(2)[:, :, None], (1, 1, len(wavelengths)))
        
        for i in range(len(materials_seq)):
            n_i = n_arr[i]
            cos_i = cos_d[i]
            sin_i = sin_d[i]
            
            M_i = np.empty((2, 2, len(wavelengths)), dtype=complex)
            M_i[0, 0] = cos_i
            M_i[0, 1] = 1j * sin_i / n_i
            M_i[1, 0] = 1j * n_i * sin_i
            M_i[1, 1] = cos_i
            
            M_tot_new = np.empty_like(M_tot)
            M_tot_new[0, 0] = M_i[0, 0] * M_tot[0, 0] + M_i[0, 1] * M_tot[1, 0]
            M_tot_new[0, 1] = M_i[0, 0] * M_tot[0, 1] + M_i[0, 1] * M_tot[1, 1]
            M_tot_new[1, 0] = M_i[1, 0] * M_tot[0, 0] + M_i[1, 1] * M_tot[1, 0]
            M_tot_new[1, 1] = M_i[1, 0] * M_tot[0, 1] + M_i[1, 1] * M_tot[1, 1]
            
            M_tot = M_tot_new
            
        B = M_tot[0, 0] + M_tot[0, 1] * n_substrate
        C = M_tot[1, 0] + M_tot[1, 1] * n_substrate
        
        Y = C / B
        r = (Y - n_ambient) / (Y + n_ambient)
        R = np.abs(r)**2
        
        return np.mean(R)

    def objective(thicknesses, materials_seq):
        return calc_reflectance(thicknesses, materials_seq)

    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    sequences = [
        ['L', 'H', 'L'],
        ['L', 'H', 'L', 'H', 'L'],
        ['L', 'H', 'L', 'H', 'L', 'H', 'L'],
        ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'],
        ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L']
    ]
    
    for seq in sequences:
        if time.time() > deadline:
            break
            
        N = len(seq)
        if N > max_layers:
            continue
            
        lam0 = 600.0
        init_thick = []
        for m in seq:
            n_mat = n_L if m == 'L' else n_H
            d = lam0 / (4.0 * n_mat)
            init_thick.append(d)
            
        bounds = [(0, max_layer_nm)] * N
        
        try:
            res = minimize(
                lambda x: objective(x, seq),
                init_thick,
                method='Powell',
                bounds=bounds,
                options={'xatol': 0.1, 'fatol': 1e-7, 'maxiter': 200, 'maxfev': 2000}
            )
            
            if res.fun < best_score:
                # Check constraints again
                if np.all(res.x >= 0) and np.all(res.x <= max_layer_nm) and np.sum(res.x) <= max_total_nm:
                    best_score = res.fun
                    best_materials = seq
                    best_thicknesses = res.x
        except Exception:
            pass

    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [600.0 / (4.0 * n_L)]
        
    return best_materials, list(best_thicknesses)
