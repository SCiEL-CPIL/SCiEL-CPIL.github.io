"""Broadband antireflection coating design via characteristic-matrix model + timed search."""
import time
import numpy as np
from scipy.optimize import differential_evolution

def _compute_reflectance(wavelengths, materials_list, thicknesses, n_ambient, n_substrate, n_materials):
    """Compute reflectance spectrum using characteristic matrix method (vectorized)."""
    N = len(wavelengths)
    n_layers = len(materials_list)
    
    if n_layers == 0:
        # Bare substrate
        R = ((n_ambient - n_substrate) / (n_ambient + n_substrate)) ** 2
        return np.full(N, R, dtype=float)
    
    k = 2.0 * np.pi / wavelengths  # (N,)
    
    # Initialize as identity
    M00 = np.ones(N, dtype=np.complex128)
    M01 = np.zeros(N, dtype=np.complex128)
    M10 = np.zeros(N, dtype=np.complex128)
    M11 = np.ones(N, dtype=np.complex128)
    
    for j in range(n_layers):
        n_j = n_materials[materials_list[j]]
        d_j = thicknesses[j]
        delta_j = k * n_j * d_j  # (N,)
        
        cos_d = np.cos(delta_j)
        sin_d = np.sin(delta_j)
        
        # Layer matrix
        L00 = cos_d
        L01 = 1j * sin_d / n_j
        L10 = 1j * n_j * sin_d
        L11 = cos_d
        
        # M_new = M_old * L
        new_M00 = M00 * L00 + M01 * L10
        new_M01 = M00 * L01 + M01 * L11
        new_M10 = M10 * L00 + M11 * L10
        new_M11 = M10 * L01 + M11 * L11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    # Admittance at front
    Y = M01 / M00
    R = np.abs((n_ambient - Y) / (n_ambient + Y)) ** 2
    
    return R

def _mean_reflectance_pct(materials_list, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_materials = problem["materials"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    R = _compute_reflectance(wavelengths, materials_list, thicknesses, n_ambient, n_substrate, n_materials)
    
    return float(np.mean(R) * 100.0)

def _is_valid(materials_list, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials_list) != len(thicknesses):
        return False
    if len(materials_list) > max_layers:
        return False
    if len(materials_list) == 0:
        return False
    
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    
    if sum(thicknesses) > max_total_nm:
        return False
    
    for m in materials_list:
        if m not in ('L', 'H'):
            return False
    
    return True

def _generate_patterns(problem):
    """Generate candidate alternating patterns."""
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_total_nm = problem["max_total_nm"]
    
    d_L = centre / (4.0 * n_L)
    d_H = centre / (4.0 * n_H)
    
    patterns = []
    
    # For each number of layers, try alternating patterns starting with L or H
    for n_layers in range(1, 21):
        # Start with L
        mats_L = []
        thicks_L = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats_L.append("L")
                thicks_L.append(d_L)
            else:
                mats_L.append("H")
                thicks_L.append(d_H)
        if sum(thicks_L) <= max_total_nm:
            patterns.append((mats_L, thicks_L))
        
        # Start with H
        mats_H = []
        thicks_H = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats_H.append("H")
                thicks_H.append(d_H)
            else:
                mats_H.append("L")
                thicks_H.append(d_L)
        if sum(thicks_H) <= max_total_nm:
            patterns.append((mats_H, thicks_H))
    
    return patterns

def _optimize_thicknesses(materials_list, thicknesses, problem, time_limit):
    """Optimize thicknesses using differential evolution with time limit."""
    start_time = time.time()
    n_layers = len(thicknesses)
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    # Define objective function
    def objective(x):
        # Check constraints
        if any(t < 0 or t > max_layer_nm for t in x):
            return 100.0
        if sum(x) > max_total_nm:
            return 100.0
        return _mean_reflectance_pct(materials_list, list(x), problem)
    
    # Bounds for each thickness
    bounds = [(0, max_layer_nm)] * n_layers
    
    # Use differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-6,
            polish=True
        )
        
        if result.success or result.fun < 100.0:
            opt_thicks = list(result.x)
            opt_score = result.fun
            # Verify validity
            if _is_valid(materials_list, opt_thicks, problem):
                return opt_thicks, opt_score
    except Exception:
        pass
    
    # Fallback: return original
    return list(thicknesses), _mean_reflectance_pct(materials_list, thicknesses, problem)

def design(problem):
    """Design a broadband antireflection coating."""
    deadline = time.time() + 140.0  # Safety margin
    
    # Generate patterns
    patterns = _generate_patterns(problem)
    
    # Evaluate all patterns and pick top few
    scored_patterns = []
    for mats, thicks in patterns:
        if _is_valid(mats, thicks, problem):
            score = _mean_reflectance_pct(mats, thicks, problem)
            scored_patterns.append((score, mats, thicks))
    
    scored_patterns.sort(key=lambda x: x[0])
    
    # Start with the best pattern
    if scored_patterns:
        best_score, best_materials, best_thicknesses = scored_patterns[0]
    else:
        # Fallback
        band = problem["band_nm"]
        centre = 0.5 * (band[0] + band[1])
        n_L = problem["materials"]["L"]
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
        best_score = _mean_reflectance_pct(best_materials, best_thicknesses, problem)
    
    # Optimize top patterns
    top_k = min(10, len(scored_patterns))
    
    for i in range(top_k):
        if time.time() > deadline - 15.0:
            break
        
        score, mats, thicks = scored_patterns[i]
        
        # Only optimize if it's not already the best
        if i == 0:
            # Skip the first one (already best)
            continue
        
        # Local optimization with time limit
        remaining_time = deadline - 15.0 - time.time()
        if remaining_time < 5.0:
            break
        
        opt_thicks, opt_score = _optimize_thicknesses(mats, thicks, problem, remaining_time)
        
        if opt_score < best_score:
            best_score = opt_score
            best_materials = list(mats)
            best_thicknesses = list(opt_thicks)
    
    # If we have time, also optimize the best pattern
    if time.time() < deadline - 10.0:
        remaining_time = deadline - 10.0 - time.time()
        if remaining_time > 5.0:
            opt_thicks, opt_score = _optimize_thicknesses(best_materials, best_thicknesses, problem, remaining_time)
            if opt_score < best_score:
                best_score = opt_score
                best_thicknesses = list(opt_thicks)
    
    # Final check
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Fallback to simple design
        band = problem["band_nm"]
        centre = 0.5 * (band[0] + band[1])
        n_L = problem["materials"]["L"]
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
