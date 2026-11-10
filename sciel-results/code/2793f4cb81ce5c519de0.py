"""Broadband AR coating design using characteristic-matrix method."""
import numpy as np
import time
from scipy.optimize import minimize

def _characteristic_matrix(n0, ns, materials, thicknesses, lambdas):
    """
    Compute reflectance using the characteristic matrix method.
    n0: ambient index, ns: substrate index
    materials: list of layer indices (same order as thicknesses)
    thicknesses: list of physical thicknesses in nm
    lambdas: array of wavelengths in nm
    Returns: array of reflectance values (0 to 1)
    """
    N = len(thicknesses)
    M11 = np.ones(len(lambdas), dtype=complex)
    M12 = np.zeros(len(lambdas), dtype=complex)
    M21 = np.zeros(len(lambdas), dtype=complex)
    M22 = np.ones(len(lambdas), dtype=complex)
    
    for j in range(N):
        nj = materials[j]
        dj = thicknesses[j]
        delta = 2.0 * np.pi * nj * dj / lambdas
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        m11 = cos_d
        m12 = 1j * sin_d / nj
        m21 = 1j * nj * sin_d
        m22 = cos_d
        
        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    
    Ys = np.full(len(lambdas), ns, dtype=complex)
    
    numerator = M11 + Ys * M12
    denominator = M21 + Ys * M22
    
    Y = numerator / denominator
    
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r) ** 2
    
    return R


def _mean_reflectance_percent(n0, ns, material_indices, thicknesses, lambdas):
    """Compute mean reflectance in percent."""
    R = _characteristic_matrix(n0, ns, material_indices, thicknesses, lambdas)
    return np.mean(R) * 100.0


def _objective(thicknesses, n0, ns, material_indices, lambdas, max_layer_nm, max_total_nm):
    """Objective function for optimizer with penalty for constraint violations."""
    thicknesses = np.asarray(thicknesses)
    penalty = 0.0
    if np.any(thicknesses < -1e-9) or np.any(thicknesses > max_layer_nm + 1e-9):
        penalty += 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-9:
        penalty += 100.0 * (np.sum(thicknesses) - max_total_nm) / max_total_nm
    t_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    total = np.sum(t_clipped)
    if total > max_total_nm:
        t_clipped = t_clipped * (max_total_nm / total)
    score = _mean_reflectance_percent(n0, ns, material_indices, t_clipped.tolist(), lambdas)
    return score + penalty


def design(problem):
    """
    Design a broadband AR coating minimizing mean reflectance over 400-800 nm.
    Returns (materials_list, thicknesses_list).
    """
    start_time = time.time()
    DEADLINE = 140.0
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    lam_center = np.sqrt(band[0] * band[1])
    
    best_score = np.inf
    best_materials = None
    best_thicknesses = None
    
    def time_check():
        return time.time() - start_time > DEADLINE - 15
    
    # Try various stack configurations
    for n_layers in [3, 5, 7, 9, 11, 13, 15]:
        if n_layers > max_layers or time_check():
            break
        
        materials = []
        for i in range(n_layers):
            if i % 2 == 0:
                materials.append("L")
            else:
                materials.append("H")
        
        material_indices = [nL if m == "L" else nH for m in materials]
        
        # Quarter-wave initialization at center wavelength
        t0 = []
        for m in materials:
            n = nL if m == "L" else nH
            d = lam_center / (4.0 * n)
            d = min(d, max_layer_nm)
            t0.append(d)
        total = sum(t0)
        if total > max_total_nm:
            scale = max_total_nm / total
            t0 = [d * scale for d in t0]
        t0 = np.array(t0)
        
        bounds = [(0.0, max_layer_nm)] * n_layers
        
        # L-BFGS-B optimization
        try:
            result = minimize(
                _objective,
                t0,
                args=(n0, ns, material_indices, lambdas, max_layer_nm, max_total_nm),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
        except Exception:
            continue
        
        if time_check():
            break
        
        t_opt = np.clip(result.x, 0.0, max_layer_nm)
        total = np.sum(t_opt)
        if total > max_total_nm:
            t_opt = t_opt * (max_total_nm / total)
        
        score = _mean_reflectance_percent(n0, ns, material_indices, t_opt.tolist(), lambdas)
        
        if score < best_score:
            best_score = score
            best_materials = materials[:]
            best_thicknesses = t_opt.tolist()
        
        # Nelder-Mead polish
        if not time_check():
            try:
                result2 = minimize(
                    _objective,
                    t_opt,
                    args=(n0, ns, material_indices, lambdas, max_layer_nm, max_total_nm),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                t_nm = np.clip(result2.x, 0.0, max_layer_nm)
                total = np.sum(t_nm)
                if total > max_total_nm:
                    t_nm = t_nm * (max_total_nm / total)
                score_nm = _mean_reflectance_percent(n0, ns, material_indices, t_nm.tolist(), lambdas)
                if score_nm < best_score:
                    best_score = score_nm
                    best_thicknesses = t_nm.tolist()
            except Exception:
                pass
    
    if best_materials is None:
        d = lam_center / (4.0 * nL)
        d = min(d, max_layer_nm)
        return ["L"], [d]
    
    total = sum(best_thicknesses)
    if total > max_total_nm + 1e-6:
        scale = max_total_nm / total
        best_thicknesses = [d * scale for d in best_thicknesses]
    
    best_thicknesses = [max(0.0, min(d, max_layer_nm)) for d in best_thicknesses]
    
    return best_materials, best_thicknesses
