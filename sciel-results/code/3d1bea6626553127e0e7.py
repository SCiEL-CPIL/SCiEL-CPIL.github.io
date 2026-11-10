import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def mean_reflectance(materials, thicknesses):
        n_stack = np.array([nL if m == 'L' else nH for m in materials])
        thicknesses = np.array(thicknesses, dtype=float)
        
        if len(materials) > max_layers:
            return 100.0
        if any(t < -1e-9 or t > max_layer_nm + 1e-9 for t in thicknesses):
            return 100.0
        if sum(thicknesses) > max_total_nm + 1e-9:
            return 100.0
        
        total_R = 0.0
        for wl in wavelengths:
            M = np.array([[1, 0], [0, 1]], dtype=complex)
            for i in range(len(materials)):
                n = n_stack[i]
                d = thicknesses[i]
                delta = 2 * np.pi * n * d / wl
                cos_d = np.cos(delta)
                sin_d = np.sin(delta)
                M_layer = np.array([
                    [cos_d, 1j * sin_d / n],
                    [1j * n * sin_d, cos_d]
                ])
                M = M @ M_layer
            
            Y_in = (M[1, 0] + M[1, 1] * ns) / (M[0, 0] + M[0, 1] * ns)
            R = np.abs((n0 - Y_in) / (n0 + Y_in)) ** 2
            total_R += R
        
        return (total_R / len(wavelengths)) * 100.0
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    from scipy.optimize import minimize
    
    lambda_c = 0.5 * (band[0] + band[1])
    
    for n_layers in [1, 2, 3, 4, 6]:
        if time.time() > deadline:
            break
        
        patterns = []
        if n_layers == 1:
            patterns.append(['L'])
        elif n_layers == 2:
            patterns.append(['L', 'H'])
            patterns.append(['H', 'L'])
        elif n_layers == 3:
            patterns.append(['L', 'H', 'L'])
            patterns.append(['H', 'L', 'H'])
        elif n_layers == 4:
            patterns.append(['L', 'H', 'L', 'H'])
            patterns.append(['H', 'L', 'H', 'L'])
        elif n_layers == 6:
            patterns.append(['L', 'H', 'L', 'H', 'L', 'H'])
            patterns.append(['H', 'L', 'H', 'L', 'H', 'L'])
        
        for pattern in patterns:
            if time.time() > deadline:
                break
            
            init_thicknesses = []
            for m in pattern:
                n = nL if m == 'L' else nH
                init_thicknesses.append(lambda_c / (4 * n))
            
            x0 = np.array(init_thicknesses)
            
            def objective(x):
                return mean_reflectance(pattern, x)
            
            bounds = [(0, max_layer_nm)] * n_layers
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': 300, 'ftol': 1e-12})
                if result.fun < best_score:
                    best_score = result.fun
                    best_materials = pattern[:]
                    best_thicknesses = result.x.tolist()
            except:
                pass
            
            if time.time() > deadline:
                break
            try:
                result2 = minimize(objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-8})
                if result2.fun < best_score:
                    best_score = result2.fun
                    best_materials = pattern[:]
                    best_thicknesses = result2.x.tolist()
            except:
                pass
    
    if best_materials is None:
        centre = 0.5 * (band[0] + band[1])
        return ["L"], [centre / (4.0 * nL)]
    
    best_thicknesses = [max(0.0, min(max_layer_nm, t)) for t in best_thicknesses]
    
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_materials, best_thicknesses
