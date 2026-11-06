"""Broadband antireflection coating design using characteristic-matrix method."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    """Compute reflectance (0-1) for a stack using the characteristic matrix method."""
    n = len(n_layers)
    if n == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.abs(r) ** 2
    
    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)
    
    for i in range(n):
        ni = n_layers[i]
        di = d_layers[i]
        delta = 2.0 * np.pi * ni * di / wavelengths
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        L11 = cos_d
        L12 = 1j * sin_d / ni
        L21 = 1j * ni * sin_d
        L22 = cos_d
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    
    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    return R


def _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    """Compute mean reflectance in percent."""
    R = _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers)
    return np.mean(R) * 100.0


def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return True
    
    total = sum(thicknesses)
    if total > max_total_nm:
        return False
    
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    
    for m in materials:
        if m not in problem["materials"]:
            return False
    
    return True


def _evaluate_design(materials, thicknesses, problem, wavelengths):
    """Evaluate a design and return mean reflectance in percent."""
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_layers = np.array([problem["materials"][m] for m in materials], dtype=float)
    d_layers = np.array(thicknesses, dtype=float)
    
    return _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers)


def _objective(x, problem, wavelengths, n_layers_arr, max_total_nm, max_layer_nm):
    """Objective function for scipy optimizer. Returns mean reflectance percent."""
    if np.any(x < -1e-9) or np.any(x > max_layer_nm + 1e-9):
        return 100.0
    if np.sum(x) > max_total_nm + 1e-6:
        return 100.0
    return _mean_reflectance_percent(wavelengths, problem["ambient_index"], 
                                      problem["substrate_index"], n_layers_arr, x)


def _optimize_with_scipy(mats, thicks, problem, wavelengths, deadline, max_time=20.0):
    """Optimize thicknesses using scipy's Nelder-Mead with time limit."""
    start_time = time.time()
    n = len(mats)
    if n == 0:
        return mats, list(thicks), _evaluate_design(mats, thicks, problem, wavelengths)
    
    n_layers_arr = np.array([problem["materials"][m] for m in mats], dtype=float)
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    x0 = np.array(thicks, dtype=float)
    
    # Bounds
    bounds = [(0.0, max_layer_nm)] * n
    
    # Try multiple restarts with different initial points
    best_score = float('inf')
    best_x = x0.copy()
    
    initial_points = [x0.copy()]
    
    # Add perturbed initial points
    for _ in range(3):
        perturbed = x0.copy()
        for i in range(n):
            perturbed[i] = max(0.0, min(max_layer_nm, perturbed[i] * (0.8 + 0.4 * np.random.random())))
        if np.sum(perturbed) <= max_total_nm:
            initial_points.append(perturbed)
    
    for x_start in initial_points:
        if time.time() > deadline or (time.time() - start_time) > max_time:
            break
        
        try:
            result = minimize(
                lambda x: _objective(x, problem, wavelengths, n_layers_arr, max_total_nm, max_layer_nm),
                x_start,
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 5000,
                    'xatol': 0.01,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Clip to valid range
    best_x = np.clip(best_x, 0.0, max_layer_nm)
    
    # Ensure total thickness constraint
    if np.sum(best_x) > max_total_nm:
        scale = max_total_nm / np.sum(best_x)
        best_x = best_x * scale
    
    best_thicks = list(best_x)
    best_score = _evaluate_design(mats, best_thicks, problem, wavelengths)
    
    return mats, best_thicks, best_score


def _polish_thicknesses(mats, thicks, problem, wavelengths, deadline, max_iter=10000):
    """High-precision polish using coordinate descent with adaptive steps."""
    current_thicks = list(thicks)
    current_score = _evaluate_design(mats, current_thicks, problem, wavelengths)
    
    n = len(mats)
    if n == 0:
        return mats, current_thicks, current_score
    
    step_sizes = [1.0] * n
    min_step = 0.001
    max_step = 5.0
    no_improve_count = 0
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            best_local_score = current_score
            best_local_thick = current_thicks[i]
            
            for factor in [1.0, 0.5, 0.25, 0.1, 2.0]:
                step = step_sizes[i] * factor
                
                for direction in [1, -1]:
                    new_thick = current_thicks[i] + direction * step
                    if new_thick < 0.0 or new_thick > 400.0:
                        continue
                    
                    new_thicks = list(current_thicks)
                    new_thicks[i] = new_thick
                    if _is_valid(mats, new_thicks, problem):
                        new_score = _evaluate_design(mats, new_thicks, problem, wavelengths)
                        if new_score < best_local_score:
                            best_local_score = new_score
                            best_local_thick = new_thick
            
            if best_local_score < current_score - 1e-10:
                current_thicks[i] = best_local_thick
                current_score = best_local_score
                improved = True
                no_improve_count = 0
        
        if not improved:
            no_improve_count += 1
            step_sizes = [s * 0.7 for s in step_sizes]
            if min(step_sizes) < min_step or no_improve_count > 5:
                break
        else:
            step_sizes = [min(s * 1.3, max_step) for s in step_sizes]
    
    return mats, current_thicks, current_score


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    # Generate diverse initial designs
    initial_designs = []
    
    # Quarter-wave thicknesses
    qw_L = centre / (4.0 * n_L)
    qw_H = centre / (4.0 * n_H)
    
    # Single layer designs
    initial_designs.append((["L"], [qw_L]))
    initial_designs.append((["H"], [qw_H]))
    
    # Two-layer designs
    initial_designs.append((["L", "H"], [qw_L, qw_H]))
    initial_designs.append((["H", "L"], [qw_H, qw_L]))
    
    # Three-layer designs
    initial_designs.append((["H", "L", "H"], [qw_H, qw_L, qw_H]))
    initial_designs.append((["L", "H", "L"], [qw_L, qw_H, qw_L]))
    
    # Four-layer designs
    initial_designs.append((["H", "L", "H", "L"], [qw_H, qw_L, qw_H, qw_L]))
    initial_designs.append((["L", "H", "L", "H"], [qw_L, qw_H, qw_L, qw_H]))
    
    # Five-layer designs
    initial_designs.append((["H", "L", "H", "L", "H"], 
                           [qw_H, qw_L, qw_H, qw_L, qw_H]))
    initial_designs.append((["L", "H", "L", "H", "L"], 
                           [qw_L, qw_H, qw_L, qw_H, qw_L]))
    
    # Six-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L"],
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    
    # Seven-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H"],
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Eight-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L"],
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    
    # Nine-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H"],
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Ten-layer designs - classic broadband design
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    
    # Eleven-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"],
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Twelve-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    
    # Non-standard patterns with varying thicknesses
    # HLLH pattern
    initial_designs.append((["H", "L", "L", "H"], [qw_H, qw_L, qw_L, qw_H]))
    initial_designs.append((["H", "L", "L", "H", "L", "L", "H"], 
                           [qw_H, qw_L, qw_L, qw_H, qw_L, qw_L, qw_H]))
    
    # LHHL pattern
    initial_designs.append((["L", "H", "H", "L"], [qw_L, qw_H, qw_H, qw_L]))
    
    # Designs with different thickness ratios
    # Try some designs where thicknesses are not exactly quarter-wave
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        initial_designs.append((["H", "L", "H", "L", "H"], 
                               [qw_H * scale, qw_L * scale, qw_H * scale, qw_L * scale, qw_H * scale]))
    
    # Evaluate all initial designs and keep the best few
    scored_designs = []
    for mats, thicks in initial_designs:
        if not _is_valid(mats, thicks, problem):
            continue
        score = _evaluate_design(mats, thicks, problem, wavelengths)
        scored_designs.append((score, mats, thicks))
    
    scored_designs.sort(key=lambda x: x[0])
    
    if not scored_designs:
        # Fallback
        best_materials = ["L"]
        best_thicknesses = [qw_L]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem, wavelengths)
    else:
        best_score, best_materials, best_thicknesses = scored_designs[0]
    
    # Optimize the top designs using scipy
    top_n = min(5, len(scored_designs))
    optimized_results = []
    
    for i in range(top_n):
        if time.time() > deadline - 30.0:
            break
        
        score, mats, thicks = scored_designs[i]
        max_time_for_this = min(15.0, max(5.0, (deadline - time.time()) / max(1, top_n - i)))
        
        mats_opt, thicks_opt, score_opt = _optimize_with_scipy(
            mats, thicks, problem, wavelengths, deadline, max_time=max_time_for_this
        )
        optimized_results.append((score_opt, mats_opt, thicks_opt))
    
    # Find the best optimized result
    if optimized_results:
        optimized_results.sort(key=lambda x: x[0])
        best_score, best_materials, best_thicknesses = optimized_results[0]
    
    # Try additional candidate sequences if time permits
    if time.time() < deadline - 40.0:
        candidate_sequences = []
        
        # Alternating sequences of various lengths
        for n_layers in range(3, min(15, max_layers + 1)):
            seq = []
            for i in range(n_layers):
                seq.append("H" if i % 2 == 0 else "L")
            candidate_sequences.append(seq)
            
            seq = []
            for i in range(n_layers):
                seq.append("L" if i % 2 == 0 else "H")
            candidate_sequences.append(seq)
        
        # Some non-alternating patterns
        candidate_sequences.append(["H", "L", "L", "H", "L", "L", "H"])
        candidate_sequences.append(["L", "H", "H", "L", "H", "H", "L"])
        candidate_sequences.append(["H", "L", "H", "L", "L", "H", "L", "H"])
        candidate_sequences.append(["L", "H", "L", "H", "H", "L", "H", "L"])
        
        # Deduplicate
        seen = set()
        unique_candidates = []
        for seq in candidate_sequences:
            if len(seq) > max_layers:
                continue
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(seq)
        
        # Evaluate and optimize promising candidates
        for seq in unique_candidates:
            if time.time() > deadline - 20.0:
                break
            
            if seq == best_materials:
                continue
            
            thicks = [centre / (4.0 * materials_dict[m]) for m in seq]
            
            if not _is_valid(seq, thicks, problem):
                continue
            
            score = _evaluate_design(seq, thicks, problem, wavelengths)
            
            # Only optimize if promising
            if score < best_score + 2.0:
                mats, thicks, score = _optimize_with_scipy(
                    seq, thicks, problem, wavelengths, deadline, max_time=10.0
                )
                if score < best_score:
                    best_score = score
                    best_materials = list(seq)
                    best_thicknesses = list(thicks)
    
    # Final refinement on the best design
    if time.time() < deadline - 15.0:
        best_materials, best_thicknesses, best_score = _optimize_with_scipy(
            best_materials, best_thicknesses, problem, wavelengths, deadline, max_time=10.0
        )
    
    # High-precision polish pass
    if time.time() < deadline - 5.0:
        best_materials, best_thicknesses, best_score = _polish_thicknesses(
            best_materials, best_thicknesses, problem, wavelengths, deadline, max_iter=5000
        )
    
    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [qw_L]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
