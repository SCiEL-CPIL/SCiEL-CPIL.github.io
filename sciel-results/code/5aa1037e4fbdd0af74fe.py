"""Broadband antireflection coating design: multi-layer stack optimization."""
import time
import numpy as np
from scipy.optimize import minimize

def _reflectance_spectrum(materials, thicknesses, band_nm, ambient_index, substrate_index, n_wavelengths=401):
    """Compute mean reflectance (percent) over the band using characteristic matrix method."""
    n_L = materials['L']
    n_H = materials['H']
    n_sub = substrate_index
    n_amb = ambient_index
    
    lam0 = band_nm[0]
    lam1 = band_nm[1]
    wavelengths = np.linspace(lam0, lam1, n_wavelengths)
    
    n_layers = len(materials)
    if n_layers == 0:
        # Bare substrate
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return 100.0 * r * r
    
    # Build the stack: air | layer1 | layer2 | ... | layerN | substrate
    # Characteristic matrix for each layer
    # M_j = [[cos(d_j), i*sin(d_j)/n_j], [i*n_j*sin(d_j), cos(d_j)]]
    
    total_R = 0.0
    for lam in wavelengths:
        # Optical thicknesses in radians: delta = 2*pi*n*d/lam
        M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = n_L if materials[j] == 'L' else n_H
            d_j = thicknesses[j]
            delta = 2.0 * np.pi * n_j * d_j / lam
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            M_j = np.array([[cos_d, 1j * sin_d / n_j],
                            [1j * n_j * sin_d, cos_d]], dtype=complex)
            M_total = M_total @ M_j
        
        # Substrate admittance
        Y_sub = n_sub
        
        # Input admittance
        B, C = M_total[0, 0], M_total[1, 0]
        Y_in = Y_sub / (B + C * Y_sub)  # Wait, let me re-derive
        
        # Actually, the characteristic matrix gives:
        # [E]   [M11 M12] [E_sub]
        # [H] = [M21 M22] [H_sub]
        # Where H = Y * E (for TE) or H = E/Z (for TM)
        # At normal incidence, Y_sub = n_sub (for TE, which is all at normal)
        # Y_in = Y_sub / (M11 + M12 * Y_sub) ... no
        
        # Standard formula:
        # Y_in = (M11 * Y_sub + M21) / (M12 * Y_sub + M22)
        # Wait, let me be careful.
        
        # The characteristic matrix M relates fields at front to back:
        # [E_front]   [M11 M12] [E_back]
        # [H_front] = [M21 M22] [H_back]
        
        # At the back (substrate side), for a plane wave in substrate:
        # H_back = Y_sub * E_back (where Y_sub = n_sub for normal incidence)
        
        # So:
        # E_front = M11 * E_back + M12 * H_back = (M11 + M12 * Y_sub) * E_back
        # H_front = M21 * E_back + M22 * H_back = (M21 + M22 * Y_sub) * E_back
        
        # Y_in = H_front / E_front = (M21 + M22 * Y_sub) / (M11 + M12 * Y_sub)
        
        Y_in = (M_total[1, 0] + M_total[1, 1] * Y_sub) / (M_total[0, 0] + M_total[0, 1] * Y_sub)
        
        # Reflection coefficient at the air-coating interface
        r = (n_amb - Y_in) / (n_amb + Y_in)
        R = np.abs(r) ** 2
        total_R += R * 100.0
    
    return total_R / n_wavelengths


def design(problem):
    """Design a broadband antireflection coating stack."""
    t_start = time.time()
    deadline = t_start + 120.0  # Leave 30s safety margin
    
    band = problem["band_nm"]
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    mat_dict = problem["materials"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    n_L = mat_dict['L']
    n_H = mat_dict['H']
    
    def _score(materials, thicknesses):
        """Compute the objective: mean reflectance in percent."""
        if len(materials) == 0:
            r = (n_amb - n_sub) / (n_amb + n_sub)
            return 100.0 * r * r
        if sum(thicknesses) > max_total + 1e-9:
            return 100.0
        for t in thicknesses:
            if t < -1e-9 or t > max_layer + 1e-9:
                return 100.0
        # Clamp for safety
        t_clamped = [max(0.0, min(max_layer, t)) for t in thicknesses]
        return _reflectance_spectrum(materials, t_clamped, band, n_amb, n_sub)
    
    best_score = 100.0
    best_materials = ["L"]
    best_thicknesses = [100.0]
    
    def _update_best(materials, thicknesses, score):
        nonlocal best_score, best_materials, best_thicknesses
        if score < best_score:
            best_score = score
            best_materials = list(materials)
            best_thicknesses = list(thicknesses)
    
    # Try several layer counts and use Nelder-Mead optimization
    # Start with promising initial guesses based on quarter-wave designs
    
    for n_layers in [3, 5, 7, 9, 11, 13, 15]:
        if n_layers > max_layers:
            break
        if time.time() > deadline - 15.0:
            break
        
        # Multiple starting points
        n_starts = 5
        for start_idx in range(n_starts):
            if time.time() > deadline - 10.0:
                break
            
            # Generate initial guess
            if start_idx == 0:
                # Quarter-wave stack centered at 600nm
                centre = 600.0
                if n_layers % 2 == 1:
                    # Start and end with L
                    materials = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
                else:
                    materials = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
                thicknesses = []
                for m in materials:
                    n_m = n_L if m == 'L' else n_H
                    thicknesses.append(centre / (4.0 * n_m))
            elif start_idx == 1:
                # Quarter-wave centered at 550nm
                centre = 550.0
                materials = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
                thicknesses = []
                for m in materials:
                    n_m = n_L if m == 'L' else n_H
                    thicknesses.append(centre / (4.0 * n_m))
            elif start_idx == 2:
                # Quarter-wave centered at 650nm
                centre = 650.0
                materials = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
                thicknesses = []
                for m in materials:
                    n_m = n_L if m == 'L' else n_H
                    thicknesses.append(centre / (4.0 * n_m))
            elif start_idx == 3:
                # Random but bounded
                np.random.seed(42 + n_layers * 100 + start_idx)
                materials = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
                thicknesses = np.random.uniform(50, 200, n_layers).tolist()
            else:
                # Another random
                np.random.seed(1000 + n_layers * 100 + start_idx)
                materials = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
                thicknesses = np.random.uniform(30, 300, n_layers).tolist()
            
            # Check if total thickness is within bounds
            if sum(thicknesses) > max_total:
                # Scale down
                scale = max_total / sum(thicknesses)
                thicknesses = [t * scale for t in thicknesses]
            
            # Clamp individual layers
            thicknesses = [max(0.0, min(max_layer, t)) for t in thicknesses]
            
            # Evaluate initial
            init_score = _score(materials, thicknesses)
            _update_best(materials, thicknesses, init_score)
            
            # Run Nelder-Mead optimization
            def objective(x):
                # x is the vector of thicknesses
                t_clamped = [max(0.0, min(max_layer, xi)) for xi in x]
                if sum(t_clamped) > max_total:
                    return 100.0
                return _score(materials, t_clamped)
            
            try:
                result = minimize(
                    objective,
                    thicknesses,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'maxfev': 200,
                        'xatol': 0.5,
                        'fatol': 0.01
                    }
                )
                opt_thicknesses = [max(0.0, min(max_layer, t)) for t in result.x]
                if sum(opt_thicknesses) <= max_total:
                    opt_score = _score(materials, opt_thicknesses)
                    _update_best(materials, opt_thicknesses, opt_score)
            except Exception:
                pass
            
            # Also try with reversed layer order (H first)
            if start_idx == 0 and n_layers > 1:
                materials_rev = list(reversed(materials))
                thicknesses_rev = list(thicknesses)
                if sum(thicknesses_rev) > max_total:
                    scale = max_total / sum(thicknesses_rev)
                    thicknesses_rev = [t * scale for t in thicknesses_rev]
                thicknesses_rev = [max(0.0, min(max_layer, t)) for t in thicknesses_rev]
                
                init_score_rev = _score(materials_rev, thicknesses_rev)
                _update_best(materials_rev, thicknesses_rev, init_score_rev)
                
                def objective_rev(x):
                    t_clamped = [max(0.0, min(max_layer, xi)) for xi in x]
                    if sum(t_clamped) > max_total:
                        return 100.0
                    return _score(materials_rev, t_clamped)
                
                try:
                    result_rev = minimize(
                        objective_rev,
                        thicknesses_rev,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 500,
                            'maxfev': 200,
                            'xatol': 0.5,
                            'fatol': 0.01
                        }
                    )
                    opt_thicknesses_rev = [max(0.0, min(max_layer, t)) for t in result_rev.x]
                    if sum(opt_thicknesses_rev) <= max_total:
                        opt_score_rev = _score(materials_rev, opt_thicknesses_rev)
                        _update_best(materials_rev, opt_thicknesses_rev, opt_score_rev)
                except Exception:
                    pass
    
    # Try a few more patterns with different layer count parity
    for n_layers in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            break
        if time.time() > deadline - 10.0:
            break
        
        # H-first pattern
        materials = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        centre = 600.0
        thicknesses = []
        for m in materials:
            n_m = n_L if m == 'L' else n_H
            thicknesses.append(centre / (4.0 * n_m))
        
        if sum(thicknesses) > max_total:
            scale = max_total / sum(thicknesses)
            thicknesses = [t * scale for t in thicknesses]
        thicknesses = [max(0.0, min(max_layer, t)) for t in thicknesses]
        
        init_score = _score(materials, thicknesses)
        _update_best(materials, thicknesses, init_score)
        
        def objective_h(x):
            t_clamped = [max(0.0, min(max_layer, xi)) for xi in x]
            if sum(t_clamped) > max_total:
                return 100.0
            return _score(materials, t_clamped)
        
        try:
            result = minimize(
                objective_h,
                thicknesses,
                method='Nelder-Mead',
                options={
                    'maxiter': 400,
                    'maxfev': 150,
                    'xatol': 0.5,
                    'fatol': 0.01
                }
            )
            opt_thicknesses = [max(0.0, min(max_layer, t)) for t in result.x]
            if sum(opt_thicknesses) <= max_total:
                opt_score = _score(materials, opt_thicknesses)
                _update_best(materials, opt_thicknesses, opt_score)
        except Exception:
            pass
    
    # Final refinement on the best solution found
    if time.time() < deadline - 5.0 and best_materials:
        def objective_final(x):
            t_clamped = [max(0.0, min(max_layer, xi)) for xi in x]
            if sum(t_clamped) > max_total:
                return 100.0
            return _score(best_materials, t_clamped)
        
        try:
            result = minimize(
                objective_final,
                best_thicknesses,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'maxfev': 100,
                    'xatol': 0.1,
                    'fatol': 0.005
                }
            )
            opt_thicknesses = [max(0.0, min(max_layer, t)) for t in result.x]
            if sum(opt_thicknesses) <= max_total:
                opt_score = _score(best_materials, opt_thicknesses)
                _update_best(best_materials, opt_thicknesses, opt_score)
        except Exception:
            pass
    
    # Ensure validity of the final result
    final_thicknesses = [max(0.0, min(max_layer, t)) for t in best_thicknesses]
    if sum(final_thicknesses) > max_total:
        # Scale down
        scale = max_total / sum(final_thicknesses)
        final_thicknesses = [t * scale for t in final_thicknesses]
    
    return best_materials, final_thicknesses
