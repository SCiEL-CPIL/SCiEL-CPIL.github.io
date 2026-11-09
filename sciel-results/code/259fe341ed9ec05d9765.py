"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _reflectance_spectrum(thicknesses, materials, ambient_index, substrate_index, band_nm):
    """Compute reflectance spectrum using characteristic matrix method.
    thicknesses: list of physical thicknesses in nm (air to substrate order)
    materials: list of refractive indices corresponding to each layer
    Returns: array of reflectance values (0 to 1) at wavelengths in band_nm
    """
    n_layers = len(thicknesses)
    wl = np.linspace(band_nm[0], band_nm[1], 401)
    k = 2.0 * np.pi / wl  # wave number in vacuum
    
    # Build characteristic matrices for each layer
    # M_j = [[cos(delta_j), (i/sin(delta_j))/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # For normal incidence, delta_j = k * n_j * d_j
    
    # We'll compute M_total = M_1 * M_2 * ... * M_N
    # Then Y = (m11 + m12 * n_s) / (m21 + m22 * n_s) where n_s is substrate index
    # R = |(Y - n_0)/(Y + n_0)|^2
    
    reflectances = np.zeros(len(wl))
    
    for i, w in enumerate(wl):
        ki = 2.0 * np.pi / w
        # Start with identity matrix
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            nj = materials[j]
            dj = thicknesses[j]
            delta = ki * nj * dj
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            # Layer matrix
            Mj = np.array([
                [cos_d, 1j * sin_d / nj],
                [1j * nj * sin_d, cos_d]
            ], dtype=complex)
            
            M = M @ Mj
        
        ns = substrate_index
        m11, m12 = M[0, 0], M[0, 1]
        m21, m22 = M[1, 0], M[1, 1]
        
        Y = (m11 + m12 * ns) / (m21 + m22 * ns)
        n0 = ambient_index
        r = (Y - n0) / (Y + n0)
        reflectances[i] = np.abs(r)**2
    
    return reflectances

def _mean_reflectance_percent(thicknesses, materials, problem):
    """Compute mean reflectance in percent."""
    refl = _reflectance_spectrum(thicknesses, materials, 
                                  problem["ambient_index"], 
                                  problem["substrate_index"],
                                  problem["band_nm"])
    return np.mean(refl) * 100.0

def _quarter_wave_thicknesses(n_layers, materials, problem):
    """Generate quarter-wave optical thickness design."""
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    thicknesses = []
    for i in range(n_layers):
        n = materials[i]
        d = center_wl / (4.0 * n)
        thicknesses.append(d)
    return thicknesses

def _make_layer_sequence(n_layers, start_with_L=True):
    """Create alternating L/H layer sequence."""
    if start_with_L:
        return ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    else:
        return ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]

def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    nL = materials_dict['L']
    nH = materials_dict['H']
    
    best_score = float('inf')
    best_layers = None
    best_thicknesses = None
    
    # Try different numbers of layers and sequences
    layer_counts = [5, 7, 9, 11, 13, 15]
    
    for n_layers in layer_counts:
        if time.time() > deadline:
            break
        
        for start_L in [True, False]:
            if time.time() > deadline:
                break
            
            layers = _make_layer_sequence(n_layers, start_L)
            n_vals = [nL if l == 'L' else nH for l in layers]
            
            # Initial guess: quarter-wave at center wavelength
            init_thick = _quarter_wave_thicknesses(n_layers, n_vals, problem)
            
            # Check constraints
            if sum(init_thick) > max_total_nm:
                # Scale down
                scale = max_total_nm / sum(init_thick)
                init_thick = [t * scale for t in init_thick]
            
            # Objective function with penalty for constraint violation
            def objective(x):
                if time.time() > deadline:
                    return float('inf')
                # Penalty for out-of-range
                pen = 0.0
                for t in x:
                    if t < 0 or t > max_layer_nm:
                        pen += 100.0 * abs(min(0, t) + max(0, t - max_layer_nm))
                total = sum(x)
                if total > max_total_nm:
                    pen += 100.0 * (total - max_total_nm)
                
                if pen > 1.0:
                    return pen
                
                return _mean_reflectance_percent(list(x), n_vals, problem)
            
            # Nelder-Mead optimization
            try:
                result = minimize(objective, init_thick, method='Nelder-Mead',
                                  options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6,
                                           'adaptive': True})
                
                if time.time() > deadline:
                    break
                    
                final_thick = result.x
                # Clip to valid range
                final_thick = np.clip(final_thick, 0, max_layer_nm)
                total = sum(final_thick)
                if total > max_total_nm:
                    final_thick = final_thick * (max_total_nm / total)
                
                score = _mean_reflectance_percent(list(final_thick), n_vals, problem)
                
                if score < best_score:
                    best_score = score
                    best_layers = layers.copy()
                    best_thicknesses = list(final_thick)
                    
            except Exception:
                pass
    
    # If no valid result found, use a simple quarter-wave stack
    if best_layers is None:
        layers = _make_layer_sequence(5, True)
        n_vals = [nL if l == 'L' else nH for l in layers]
        thicknesses = _quarter_wave_thicknesses(len(layers), n_vals, problem)
        total = sum(thicknesses)
        if total > max_total_nm:
            thicknesses = [t * (max_total_nm / total) for t in thicknesses]
        best_layers = layers
        best_thicknesses = thicknesses
    
    return best_layers, best_thicknesses
# EVOLVE-BLOCK-END
