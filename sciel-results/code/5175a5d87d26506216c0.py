"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _char_matrix(n, d, wl):
    """Compute the characteristic matrix for a single layer at wavelength wl."""
    beta = 2.0 * np.pi * n * d / wl
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    m = np.array([
        [cos_b, 1.0j * sin_b / n],
        [n * 1.0j * sin_b, cos_b]
    ])
    return m

def _mean_reflectance_vec(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band using vectorized numpy."""
    if len(materials) == 0:
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r
    
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    mat_dict = problem["materials"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    
    # Initialize total matrix for all wavelengths
    # M is shape (401, 2, 2)
    M = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for mat, thick in zip(materials, thicknesses):
        n = mat_dict[mat]
        d = thick
        if d == 0:
            continue
        beta = 2.0 * np.pi * n * d / wavelengths
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Layer matrix
        m00 = cos_b
        m01 = 1.0j * sin_b / n
        m10 = n * 1.0j * sin_b
        m11 = cos_b
        
        # Multiply M @ m
        # M_new[i, 0, 0] = M[i, 0, 0]*m00 + M[i, 0, 1]*m10
        # M_new[i, 0, 1] = M[i, 0, 0]*m01 + M[i, 0, 1]*m11
        # M_new[i, 1, 0] = M[i, 1, 0]*m00 + M[i, 1, 1]*m10
        # M_new[i, 1, 1] = M[i, 1, 0]*m01 + M[i, 1, 1]*m11
        
        M_new = np.empty_like(M)
        M_new[:, 0, 0] = M[:, 0, 0] * m00 + M[:, 0, 1] * m10
        M_new[:, 0, 1] = M[:, 0, 0] * m01 + M[:, 0, 1] * m11
        M_new[:, 1, 0] = M[:, 1, 0] * m00 + M[:, 1, 1] * m10
        M_new[:, 1, 1] = M[:, 1, 0] * m01 + M[:, 1, 1] * m11
        M = M_new
        
    # Substrate admittance
    Ys = ns
    
    # Total admittance Y = M[1, 1] / M[0, 1]
    Y = M[:, 1, 1] / M[:, 0, 1]
    
    # Reflectance
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r) ** 2 * 100.0
    
    return np.mean(R)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    if sum(thicknesses) > max_total_nm:
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    return True

def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    
    dL = centre / (4.0 * nL)
    dH = centre / (4.0 * nH)
    
    # Generate candidate structures
    candidates = []
    
    # 1-layer L
    candidates.append((["L"], [dL]))
    
    # 3-layer L-H-L
    candidates.append((["L", "H", "L"], [dL, dH, dL]))
    
    # 5-layer L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L"], [dL, dH, dL, dH, dL]))
    
    # 7-layer L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L"], [dL, dH, dL, dH, dL, dH, dL]))
    
    # 9-layer L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 11-layer L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 13-layer L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 15-layer L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 17-layer L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 19-layer L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    best_score = float('inf')
    best_result = candidates[0]
    
    # Evaluate initial candidates
    for mats, thicks in candidates:
        if not _is_valid(mats, thicks, problem):
            continue
        score = _mean_reflectance_vec(mats, thicks, problem)
        if score < best_score:
            best_score = score
            best_result = (mats, thicks)
        if time.time() > deadline:
            break
    
    # Local optimization: Nelder-Mead or simple coordinate descent
    # We will use a simple iterative improvement strategy on the best candidate
    # and potentially try others if time permits.
    
    def optimize(materials, thicknesses, problem, time_limit):
        """Simple local optimization using coordinate descent with adaptive step size."""
        current_thicks = list(thicknesses)
        current_score = _mean_reflectance_vec(materials, current_thicks, problem)
        
        n_params = len(current_thicks)
        # Initial step sizes
        steps = [10.0] * n_params
        
        # Coordinate descent
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n_params):
                # Try positive and negative steps
                for direction in [1.0, -1.0]:
                    trial = list(current_thicks)
                    trial[i] += direction * steps[i]
                    
                    if _is_valid(materials, trial, problem):
                        trial_score = _mean_reflectance_vec(materials, trial, problem)
                        if trial_score < current_score:
                            current_score = trial_score
                            current_thicks = trial
                            improved = True
                            # Increase step size slightly
                            steps[i] *= 1.2
                        else:
                            # Decrease step size
                            steps[i] *= 0.8
                # Check time
                if time.time() > time_limit:
                    break
        
        return current_thicks, current_score

    # Optimize the best candidate found so far
    if time.time() < deadline - 5.0:
        mats, thicks = best_result
        opt_thicks, opt_score = optimize(mats, thicks, problem, deadline - 5.0)
        if opt_score < best_score:
            best_score = opt_score
            best_result = (mats, opt_thicks)
    
    # If time remains, try optimizing other top candidates
    # Sort candidates by initial score
    scored_candidates = []
    for mats, thicks in candidates:
        if not _is_valid(mats, thicks, problem):
            continue
        # Re-evaluate if not already done, or just use a rough estimate
        # To save time, we only optimize the top few distinct structures
        # We already evaluated them all in the first loop.
        # Let's just pick the second best structure if it's different
        pass
    
    # A more robust approach: try optimizing a few different structures
    # We'll pick the best 3 distinct structures from the initial evaluation
    # and optimize them if time permits.
    
    # For simplicity and robustness within time limits, we stick to optimizing the single best initial candidate
    # unless we have significant time left.
    
    if time.time() < deadline - 10.0:
        # Try a different structure: 5-layer stack
        alt_mats = ["L", "H", "L", "H", "L"]
        alt_thicks = [dL, dH, dL, dH, dL]
        if _is_valid(alt_mats, alt_thicks, problem):
            opt_thicks, opt_score = optimize(alt_mats, alt_thicks, problem, deadline - 10.0)
            if opt_score < best_score:
                best_score = opt_score
                best_result = (alt_mats, opt_thicks)

    return best_result
# EVOLVE-BLOCK-END
