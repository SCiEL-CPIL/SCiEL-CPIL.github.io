import numpy as np
import scipy.optimize as so
import time

def design(problem):
    ambient_index = problem.get("ambient_index", 1.0)
    substrate_index = problem.get("substrate_index", 1.52)
    materials = problem.get("materials", {"L": 1.38, "H": 2.35})
    band_nm = problem.get("band_nm", [400, 800])
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    nL = materials["L"]
    nH = materials["H"]
    n0 = ambient_index
    ns = substrate_index
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    two_pi_over_lambda = 2.0 * np.pi / wavelengths
    
    def compute_reflectance(mat_types, thicknesses):
        N = len(mat_types)
        if N == 0:
            y_in = 1.0 / ns
            r = (y_in - n0) / (y_in + n0)
            return np.mean(np.abs(r)**2) * 100.0
            
        m11 = np.ones_like(wavelengths)
        m12 = np.zeros_like(wavelengths)
        m21 = np.zeros_like(wavelengths)
        m22 = np.ones_like(wavelengths)
        
        for i in range(N):
            n_j = nL if mat_types[i] == 'L' else nH
            d_j = thicknesses[i]
            if d_j < 1e-6:
                continue
            delta = two_pi_over_lambda * n_j * d_j
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            i_s_over_n = 1j * sin_d / n_j
            i_n_s = 1j * n_j * sin_d
            
            new_m11 = m11 * cos_d + m12 * i_s_over_n
            new_m12 = m11 * i_n_s + m12 * cos_d
            new_m21 = m21 * cos_d + m22 * i_s_over_n
            new_m22 = m21 * i_n_s + m22 * cos_d
            
            m11, m12, m21, m22 = new_m11, new_m12, new_m21, new_m22
            
        y_in = (m11 + m12 * ns) / (m21 + m22 * ns)
        r = (y_in - n0) / (y_in + n0)
        refl = np.abs(r)**2
        return np.mean(refl) * 100.0

    def objective(x):
        thicknesses = x[:max_layers]
        mat_inds = x[max_layers:]
        
        total_th = np.sum(thicknesses)
        if total_th > max_total_nm:
            return 100.0 + 10.0 * (total_th - max_total_nm)
            
        mat_types = []
        valid_ths = []
        for i in range(max_layers):
            t = thicknesses[i]
            m = 'H' if mat_inds[i] > 0.5 else 'L'
            mat_types.append(m)
            valid_ths.append(t)
            
        return compute_reflectance(mat_types, valid_ths)

    bounds = []
    for _ in range(max_layers):
        bounds.append((0.0, max_layer_nm))
    for _ in range(max_layers):
        bounds.append((0.0, 1.0))
        
    start_time = time.time()
    
    try:
        res = so.differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=100, 
            popsize=20, 
            tol=1e-4
        )
    except Exception:
        return ['L', 'H', 'L', 'H'], [108.7, 63.8, 108.7, 63.8]

    best_x = res.x
    best_thicknesses = best_x[:max_layers]
    best_mat_inds = best_x[max_layers:]
    
    best_mat_types = []
    best_ths_list = []
    for i in range(max_layers):
        t = best_thicknesses[i]
        m = 'H' if best_mat_inds[i] > 0.5 else 'L'
        if t > 1.0:
            best_mat_types.append(m)
            best_ths_list.append(t)
            
    if not best_mat_types:
        best_mat_types = ['L', 'H', 'L', 'H']
        best_ths_list = [108.7, 63.8, 108.7, 63.8]
        
    active_mats = best_mat_types
    active_ths = np.array(best_ths_list)
    
    if len(active_mats) > 0:
        def local_obj(th_vec):
            if np.sum(th_vec) > max_total_nm:
                return 100.0 + 10.0 * (np.sum(th_vec) - max_total_nm)
            th_vec = np.maximum(th_vec, 0.0)
            return compute_reflectance(active_mats, th_vec)
        
        try:
            res_local = so.minimize(
                local_obj, 
                active_ths, 
                method='Nelder-Mead', 
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            
            if res_local.fun < compute_reflectance(active_mats, active_ths):
                new_ths = res_local.x
                final_mats = []
                final_ths = []
                for i in range(len(active_mats)):
                    t = new_ths[i]
                    if t > 1.0:
                        final_mats.append(active_mats[i])
                        final_ths.append(t)
                if final_mats:
                    best_mat_types = final_mats
                    best_ths_list = final_ths
        except Exception:
            pass

    return best_mat_types, best_ths_list
