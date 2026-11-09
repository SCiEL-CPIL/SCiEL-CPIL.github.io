import numpy as np
import time
from scipy.optimize import minimize

def _compute_reflectance(materials, thicknesses, n_amb, n_sub, wavelengths):
    """Compute reflectance spectrum using characteristic matrix method."""
    n_L = 1.38
    n_H = 2.35
    n_layers = len(materials)
    n_arr = np.array([n_L if m == 'L' else n_H for m in materials], dtype=float)
    d_arr = np.array(thicknesses, dtype=float)
    n_wav = len(wavelengths)
    R = np.zeros(n_wav)
    
    for iw, lam in enumerate(wavelengths):
        delta = 2.0 * np.pi * n_arr * d_arr / lam
        M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for i in range(n_layers):
            c = np.cos(delta[i])
            s = np.sin(delta[i])
            n_i = n_arr[i]
            M_layer = np.array([[c, 1j * s / n_i],
                                [1j * n_i * s, c]], dtype=complex)
            M_total = M_total @ M_layer
        
        B = M_total[0, 0] + M_total[0, 1] * n_sub
        C = M_total[1, 0] + M_total[1, 1] * n_sub
        r = (n_amb * B - C) / (n_amb * B + C)
        R[iw] = np.abs(r) ** 2
    
    return R


def _mean_reflectance_percent(materials, thicknesses, n_amb, n_sub, band):
    """Compute mean reflectance in percent over the band."""
    lam_min, lam_max = band
    wavelengths = np.linspace(lam_min, lam_max, 401)
    R = _compute_reflectance(materials, thicknesses, n_amb, n_sub, wavelengths)
    return 100.0 * np.mean(R)


def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lam_min, lam_max = band
    lam_center = 0.5 * (lam_min + lam_max)
    
    # Precompute wavelengths
    wavelengths = np.linspace(lam_min, lam_max, 401)
    
    n_L = 1.38
    n_H = 2.35
    
    def obj(thick, materials):
        return _mean_reflectance_percent(materials, list(thick), n_amb, n_sub, band)
    
    best_score = 100.0
    best_materials = ["L"]
    best_thicknesses = [lam_center / (4.0 * n_L)]
    
    # Define candidate patterns
    # For broadband AR, symmetric patterns with alternating L/H are common
    # Try patterns with 2, 4, 6, 8, 10 layers
    
    candidate_patterns = []
    
    # Symmetric patterns (palindromic)
    for n in [2, 4, 6, 8, 10]:
        if n % 2 == 0:
            half = n // 2
            # L H L H ... pattern for first half
            first_half = ['L' if i % 2 == 0 else 'H' for i in range(half)]
            pattern = first_half + first_half[::-1]
            candidate_patterns.append(pattern)
        else:
            half = n // 2
            first_half = ['L' if i % 2 == 0 else 'H' for i in range(half)]
            pattern = first_half + ['L'] + first_half[::-1]
            candidate_patterns.append(pattern)
    
    # Non-symmetric: alternating starting with L
    for n in [2, 4, 6, 8, 10, 12]:
        pattern = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        candidate_patterns.append(pattern)
    
    # Non-symmetric: alternating starting with H
    for n in [2, 4, 6, 8, 10, 12]:
        pattern = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        candidate_patterns.append(pattern)
    
    # Remove duplicates
    seen = set()
    unique_patterns = []
    for p in candidate_patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    # Sort by number of layers
    unique_patterns.sort(key=lambda x: len(x))
    
    for materials in unique_patterns:
        if time.time() > deadline - 20:
            break
        
        n_layers = len(materials)
        n_arr = [n_L if m == 'L' else n_H for m in materials]
        
        # Initial guess: quarter-wave at center wavelength
        init_thick = [lam_center / (4.0 * n) for n in n_arr]
        
        # Scale if needed
        total = sum(init_thick)
        if total > max_total_nm:
            scale = max_total_nm / total
            init_thick = [t * scale for t in init_thick]
        
        bounds = [(0.0, max_layer_nm)] * n_layers
        
        # Optimize
        try:
            result = minimize(
                lambda x: obj(x, materials),
                init_thick,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 1500, 'xatol': 0.1, 'fatol': 1e-5, 'adaptive': True}
            )
            
            if time.time() > deadline - 10:
                break
            
            thick = np.clip(result.x, 0.0, max_layer_nm)
            if sum(thick) <= max_total_nm + 1e-6:
                score = obj(thick, materials)
                if score < best_score:
                    best_score = score
                    best_materials = list(materials)
                    best_thicknesses = list(thick)
        except Exception:
            pass
    
    # Final polish
    if best_score < 50.0 and time.time() < deadline - 10:
        try:
            bounds = [(0.0, max_layer_nm)] * len(best_materials)
            result = minimize(
                lambda x: obj(x, best_materials),
                best_thicknesses,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 3000, 'xatol': 0.01, 'fatol': 1e-7, 'adaptive': True}
            )
            thick = np.clip(result.x, 0.0, max_layer_nm)
            if sum(thick) <= max_total_nm + 1e-6:
                score = obj(thick, best_materials)
                if score < best_score:
                    best_thicknesses = list(thick)
                    best_score = score
        except Exception:
            pass
    
    # Ensure validity
    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_materials, best_thicknesses
