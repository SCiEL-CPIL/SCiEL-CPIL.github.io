import numpy as np
from scipy.optimize import differential_evolution

def design(problem):
    n_0 = 1.0
    n_s = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wl = np.linspace(band[0], band[1], 401)
    
    def calc_reflectance(materials, thicknesses):
        N = len(materials)
        if N == 0:
            r = (n_0 - n_s) / (n_0 + n_s)
            return np.abs(r)**2 * 100.0

        Rs = np.zeros(len(wl))
        
        for idx, lam in enumerate(wl):
            A = 1.0 + 0j
            B = 0.0 + 0j
            C = 0.0 + 0j
            D = 1.0 + 0j
            
            for j in range(N):
                n_j = materials[j]
                d_j = thicknesses[j]
                delta_j = 2.0 * np.pi * n_j * d_j / lam
                cos_d = np.cos(delta_j)
                sin_d = np.sin(delta_j)
                
                Aj = cos_d
                Bj = 1j * sin_d / n_j
                Cj = 1j * n_j * sin_d
                Dj = cos_d
                
                A_new = A * Aj + B * Cj
                B_new = A * Bj + B * Dj
                C_new = C * Aj + D * Cj
                D_new = C * Bj + D * Dj
                
                A, B, C, D = A_new, B_new, C_new, D_new
            
            num = n_0 * (A + B * n_s) - (C + D * n_s)
            den = n_0 * (A + B * n_s) + (C + D * n_s)
            
            if den == 0:
                Rs[idx] = 100.0
            else:
                Rs[idx] = np.abs(num / den)**2 * 100.0
                
        return np.mean(Rs)

    def objective(params):
        d_base, d_grad, N_cont = params
        
        N = int(round(N_cont))
        N = max(1, min(N, max_layers))
        
        i_vals = np.arange(N)
        thicknesses = d_base + i_vals * d_grad
        thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
        
        total = np.sum(thicknesses)
        if total > max_total_nm:
            if total > 0:
                thicknesses = thicknesses * (max_total_nm / total)
            else:
                thicknesses = np.zeros(N)
        
        materials = []
        for i in range(N):
            if i % 2 == 0:
                materials.append(n_L)
            else:
                materials.append(n_H)
                
        return calc_reflectance(materials, thicknesses)

    bounds = [(0.0, max_layer_nm), (-10.0, 10.0), (1.0, float(max_layers))]
    
    res = differential_evolution(objective, bounds, maxiter=100, tol=1e-4, seed=42, polish=True)
    
    best_params = res.x
    d_base, d_grad, N_cont = best_params
    N = int(round(N_cont))
    N = max(1, min(N, max_layers))
    
    i_vals = np.arange(N)
    thicknesses = d_base + i_vals * d_grad
    thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
    
    total = np.sum(thicknesses)
    if total > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / total)
        
    materials = []
    for i in range(N):
        if i % 2 == 0:
            materials.append('L')
        else:
            materials.append('H')
            
    thicknesses_list = [float(t) for t in thicknesses]
    
    return materials, thicknesses_list
