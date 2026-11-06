"""Rewritten program for broadband antireflection coating design."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START

def _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm):
    """Compute mean reflectance in percent over the band using characteristic matrix method."""
    if len(labels) == 0:
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return r0 * r0 * 100.0
    
    n_L = materials['L']
    n_H = materials['H']
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    n_layers = len(labels)
    
    # Precompute layer indices
    n_indices = np.empty(n_layers, dtype=float)
    for i, lab in enumerate(labels):
        n_indices[i] = n_L if lab == 'L' else n_H
    
    d = np.array(thicknesses, dtype=float)
    
    # Vectorize over wavelengths: compute delta for all layers at all wavelengths
    # delta shape: (n_wavelengths, n_layers)
    # delta[j, k] = 2*pi * n_indices[k] * d[k] / wavelengths[j]
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = 2.0 * np.pi * np.outer(wavelengths, n_indices * d) / wavelengths[:, np.newaxis]
        # Wait, that's not right. Let me fix:
        # delta[j, k] = 2*pi * n_indices[k] * d[k] / wavelengths[j]
        delta = 2.0 * np.pi * (n_indices * d)[np.newaxis, :] / wavelengths[:, np.newaxis]
    
    cos_d = np.cos(delta)  # (401, n_layers)
    sin_d = np.sin(delta)  # (401, n_layers)
    
    # For each wavelength, we need to multiply the layer matrices.
    # This is inherently sequential over layers, but we can vectorize over wavelengths.
    # Initialize M as identity for all wavelengths
    m11 = np.ones(401, dtype=complex)
    m12 = np.zeros(401, dtype=complex)
    m21 = np.zeros(401, dtype=complex)
    m22 = np.ones(401, dtype=complex)
    
    for k in range(n_layers):
        n_k = n_indices[k]
        lm11 = cos_d[:, k]
        lm12 = 1j * sin_d[:, k] / n_k
        lm21 = 1j * n_k * sin_d[:, k]
        lm22 = cos_d[:, k]
        
        new_m11 = m11 * lm11 + m12 * lm21
        new_m12 = m11 * lm12 + m12 * lm22
        new_m21 = m21 * lm11 + m22 * lm21
        new_m22 = m21 * lm12 + m22 * lm22
        
        m11 = new_m11
        m12 = new_m12
        m21 = new_m21
        m22 = new_m22
    
    # Compute admittance Y = (m21 + m22 * n_s) / (m11 + m12 * n_s)
    Y = (m21 + m22 * substrate_index) / (m11 + m12 * substrate_index)
    
    # Reflectance R = |(n_0 - Y)/(n_0 + Y)|^2
    r = (ambient_index - Y) / (ambient_index + Y)
    reflectances = np.abs(r) ** 2
    
    return np.mean(reflectances) * 100.0


def _objective(params, labels, materials, ambient_index, substrate_index, band_nm, max_total_nm, max_layer_nm):
    """Objective function for optimizer: mean reflectance with constraint penalty."""
    thicknesses = np.clip(params, 0, max_layer_nm)
    total = np.sum(thicknesses)
    if total > max_total_nm:
        return 100.0  # Large penalty
    return _compute_reflectance(labels, thicknesses.tolist(), materials, ambient_index, substrate_index, band_nm)


def _quarter_wave_thickness(n, center_wl):
    return center_wl / (4.0 * n)


def _generate_initial_designs(materials, ambient_index, substrate_index, band_nm, max_layers, max_total_nm, max_layer_nm):
    """Generate a set of initial designs to evaluate."""
    designs = []
    n_L = materials['L']
    n_H = materials['H']
    center = 0.5 * (band_nm[0] + band_nm[1])
    
    # Quarter-wave designs with various numbers of layers
    for n_layers in range(1, min(max_layers, 11) + 1):
        # Alternating starting with H (higher index closer to air is typical for AR)
        for start in ['H', 'L']:
            labels = []
            thicknesses = []
            for i in range(n_layers):
                if (i % 2 == 0 and start == 'H') or (i % 2 == 1 and start == 'L'):
                    labels.append('H')
                    thicknesses.append(_quarter_wave_thickness(n_H, center))
                else:
                    labels.append('L')
                    thicknesses.append(_quarter_wave_thickness(n_L, center))
            
            # Check constraints
            if sum(thicknesses) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicknesses):
                designs.append((labels, thicknesses))
    
    # Also try with center wavelengths at different points
    for center_wl in [500, 550, 600, 650, 700]:
        for n_layers in [2, 3, 4, 5]:
            for start in ['H', 'L']:
                labels = []
                thicknesses = []
                for i in range(n_layers):
                    if (i % 2 == 0 and start == 'H') or (i % 2 == 1 and start == 'L'):
                        labels.append('H')
                        thicknesses.append(_quarter_wave_thickness(n_H, center_wl))
                    else:
                        labels.append('L')
                        thicknesses.append(_quarter_wave_thickness(n_L, center_wl))
                
                if sum(thicknesses) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicknesses):
                    designs.append((labels, thicknesses))
    
    return designs


def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    materials = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials['L']
    n_H = materials['H']
    
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    def update_best(labels, thicknesses, score):
        nonlocal best_labels, best_thicknesses, best_score
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
    
    # Step 1: Evaluate initial designs
    initial_designs = _generate_initial_designs(materials, ambient_index, substrate_index, band_nm, max_layers, max_total_nm, max_layer_nm)
    
    for labels, thicknesses in initial_designs:
        if time.time() > deadline - 30:
            break
        score = _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm)
        update_best(labels, thicknesses, score)
    
    # Step 2: Optimize thicknesses for the best few structures
    if best_labels is None:
        best_labels = ['L']
        best_thicknesses = [_quarter_wave_thickness(n_L, 600)]
        best_score = _compute_reflectance(best_labels, best_thicknesses, materials, ambient_index, substrate_index, band_nm)
    
    # Try optimizing the best structures found so far
    # Keep top 5 unique structures by score
    evaluated = []
    for labels, thicknesses in initial_designs:
        if time.time() > deadline - 30:
            break
        score = _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm)
        evaluated.append((score, labels, thicknesses))
    
    evaluated.sort(key=lambda x: x[0])
    
    # Optimize top structures
    top_structures = []
    seen = set()
    for score, labels, thicknesses in evaluated:
        key = tuple(labels)
        if key not in seen:
            seen.add(key)
            top_structures.append((labels, thicknesses))
        if len(top_structures) >= 5:
            break
    
    for labels, thicknesses in top_structures:
        if time.time() > deadline - 20:
            break
        
        n_layers = len(labels)
        bounds = [(0, max_layer_nm)] * n_layers
        
        # Use differential evolution for global optimization
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(labels, materials, ambient_index, substrate_index, band_nm, max_total_nm, max_layer_nm),
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=True
            )
            
            if result.success or result.fun < best_score:
                opt_thicknesses = np.clip(result.x, 0, max_layer_nm)
                if np.sum(opt_thicknesses) <= max_total_nm:
                    score = _compute_reflectance(labels, opt_thicknesses.tolist(), materials, ambient_index, substrate_index, band_nm)
                    update_best(labels, opt_thicknesses.tolist(), score)
        except Exception:
            pass
    
    # Step 3: Try optimizing with more layers if time permits
    if time.time() < deadline - 40:
        # Try 6-8 layer designs
        for n_layers in [6, 7, 8]:
            if time.time() > deadline - 30:
                break
            
            # Start with quarter-wave pattern
            center = 600
            labels = []
            thicknesses = []
            for i in range(n_layers):
                if i % 2 == 0:
                    labels.append('H')
                    thicknesses.append(_quarter_wave_thickness(n_H, center))
                else:
                    labels.append('L')
                    thicknesses.append(_quarter_wave_thickness(n_L, center))
            
            if sum(thicknesses) > max_total_nm:
                continue
            
            bounds = [(0, max_layer_nm)] * n_layers
            try:
                result = differential_evolution(
                    _objective,
                    bounds,
                    args=(labels, materials, ambient_index, substrate_index, band_nm, max_total_nm, max_layer_nm),
                    maxiter=40,
                    popsize=12,
                    tol=1e-4,
                    seed=42,
                    polish=True
                )
                
                if result.success or result.fun < best_score:
                    opt_thicknesses = np.clip(result.x, 0, max_layer_nm)
                    if np.sum(opt_thicknesses) <= max_total_nm:
                        score = _compute_reflectance(labels, opt_thicknesses.tolist(), materials, ambient_index, substrate_index, band_nm)
                        update_best(labels, opt_thicknesses.tolist(), score)
            except Exception:
                pass
    
    # Step 4: Local refinement using Nelder-Mead or BFGS on the best design
    if best_labels is not None and time.time() < deadline - 10:
        try:
            n_layers = len(best_labels)
            x0 = np.array(best_thicknesses, dtype=float)
            
            def local_obj(x):
                x = np.clip(x, 0, max_layer_nm)
                if np.sum(x) > max_total_nm:
                    return 100.0
                return _compute_reflectance(best_labels, x.tolist(), materials, ambient_index, substrate_index, band_nm)
            
            res = minimize(local_obj, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 0.1, 'fatol': 1e-6})
            
            if res.fun < best_score:
                opt_t = np.clip(res.x, 0, max_layer_nm)
                if np.sum(opt_t) <= max_total_nm:
                    score = _compute_reflectance(best_labels, opt_t.tolist(), materials, ambient_index, substrate_index, band_nm)
                    update_best(best_labels, opt_t.tolist(), score)
        except Exception:
            pass
    
    # Final validation and cleanup
    if best_labels is None:
        best_labels = ['L']
        best_thicknesses = [_quarter_wave_thickness(n_L, 600)]
    
    # Ensure constraints are satisfied
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    for i in range(len(best_thicknesses)):
        best_thicknesses[i] = max(0, min(max_layer_nm, best_thicknesses[i]))
    
    # Remove zero-thickness layers
    final_labels = []
    final_thicknesses = []
    for i in range(len(best_labels)):
        if best_thicknesses[i] > 1e-6:
            final_labels.append(best_labels[i])
            final_thicknesses.append(best_thicknesses[i])
    
    if len(final_labels) == 0:
        final_labels = ['L']
        final_thicknesses = [_quarter_wave_thickness(n_L, 600)]
    
    return final_labels, final_thicknesses

# EVOLVE-BLOCK-END
