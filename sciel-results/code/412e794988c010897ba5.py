"""Broadband antireflection coating design using characteristic-matrix method."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _char_matrix(n, d, wl):
    """
    Characteristic matrix for a single lossless layer at normal incidence.
    Returns a 2x2 complex numpy array.
    M = [[cos(beta), (1/(i*n))*sin(beta)],
         [i*n*sin(beta), cos(beta)]]
    """
    beta = 2.0 * np.pi * n * d / wl
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    # 1/(i*n) = -i/n
    # i*n
    m11 = cos_b
    m12 = -1j * sin_b / n
    m21 = 1j * n * sin_b
    m22 = cos_b
    return np.array([[m11, m12], [m21, m22]], dtype=complex)


def _compute_reflectance(materials, thicknesses, wl, ambient_index, substrate_index):
    """
    Compute the reflectance (0 to 1) at a single wavelength using the
    characteristic-matrix method.
    """
    # Start with the substrate admittance
    # The characteristic matrix of the stack
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    for i in range(len(materials)):
        n = materials[i]
        d = thicknesses[i]
        Mi = _char_matrix(n, d, wl)
        M = M @ Mi
    
    # The admittance at the front surface
    # Y = [m12, m22] @ [1, ns] / [m11, m21] @ [1, ns] ... 
    # Actually, the standard formula for normal incidence:
    # Y = (m11 * ns + m12) / (m21 * ns + m22)  -- wait, let me recall.
    
    # The characteristic matrix relates the fields at the front and back:
    # [E]   [m11 m12] [E']
    # [H] = [m21 m22] [H']
    # At the back (substrate), H' = E' / ns (for a wave going into substrate)
    # So [E]   [m11 m12] [E']
    #    [H] = [m21 m22] [E'/ns]
    # E = E' (m11 + m12/ns)
    # H = E' (m21 + m22/ns)
    # Y = H/E = (m21 + m22/ns) / (m11 + m12/ns)
    
    ns = substrate_index
    Y = (M[1, 0] + M[1, 1] / ns) / (M[0, 0] + M[0, 1] / ns)
    
    # Reflectance at the front surface (air to coating)
    # r = (Ya - Y) / (Ya + Y) where Ya = ambient_index (since H = E/Ya in air for incident+reflected)
    # Wait, the admittance Y computed above is the ratio H/E at the front surface.
    # In the air, the total field is E_inc + E_ref, and H = (E_inc - E_ref)/na.
    # So H/E = (1 - r)/(1 + r) * (1/na) ... 
    # Actually, Y = (E_inc - E_ref) / (E_inc + E_ref) * na
    # So Y/na = (1-r)/(1+r)
    # r = (1 - Y/na) / (1 + Y/na)
    
    na = ambient_index
    r = (1.0 - Y / na) / (1.0 + Y / na)
    R = np.abs(r) ** 2
    return R


def _mean_reflectance_percent(materials, thicknesses, problem):
    """
    Compute the mean reflectance in percent over the band.
    """
    band = problem["band_nm"]
    wl_min, wl_max = band[0], band[1]
    wavelengths = np.linspace(wl_min, wl_max, 401)
    
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    mat_dict = problem["materials"]
    
    # Convert material labels to indices
    n_list = [mat_dict[m] for m in materials]
    
    total_R = 0.0
    for wl in wavelengths:
        R = _compute_reflectance(n_list, thicknesses, wl, ambient_index, substrate_index)
        total_R += R
    
    mean_R = total_R / len(wavelengths)
    return 100.0 * mean_R


def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if len(materials) == 0:
        return False
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    total = 0.0
    for d in thicknesses:
        if d < 0 or d > max_layer_nm:
            return False
        total += d
    if total > max_total_nm + 1e-9:
        return False
    for m in materials:
        if m not in problem["materials"]:
            return False
    return True


def _initial_design(problem):
    """
    Generate an initial design based on a quarter-wave stack at the center wavelength.
    We use a 4-layer stack: L H L H or H L H L, etc.
    Let's try a 4-layer stack centered at 600 nm.
    """
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    mat_dict = problem["materials"]
    nL = mat_dict["L"]
    nH = mat_dict["H"]
    
    # Quarter wave thicknesses
    dL = center_wl / (4.0 * nL)
    dH = center_wl / (4.0 * nH)
    
    # Try a few common starting patterns
    # Pattern 1: L H L H
    # Pattern 2: H L H L
    # Pattern 3: L H H L
    # Pattern 4: H L L H
    
    patterns = [
        (["L", "H", "L", "H"], [dL, dH, dL, dH]),
        (["H", "L", "H", "L"], [dH, dL, dH, dL]),
        (["L", "H", "H", "L"], [dL, dH, dH, dL]),
        (["H", "L", "L", "H"], [dH, dL, dL, dH]),
    ]
    
    best_pattern = None
    best_score = float('inf')
    
    for mats, thicks in patterns:
        if _is_valid(mats, thicks, problem):
            score = _mean_reflectance_percent(mats, thicks, problem)
            if score < best_score:
                best_score = score
                best_pattern = (mats, thicks)
    
    if best_pattern is None:
        # Fallback
        return ["L"], [dL]
    
    return best_pattern


def _coordinate_descent(materials, thicknesses, problem, deadline):
    """
    Perform coordinate descent optimization on the thicknesses.
    """
    best_mats = list(materials)
    best_thicks = list(thicknesses)
    best_score = _mean_reflectance_percent(best_mats, best_thicks, problem)
    
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    # Number of iterations
    max_iters = 200
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # For each layer, try to optimize its thickness
        for i in range(len(best_thicks)):
            if time.time() > deadline:
                break
            
            # Try a few step sizes
            step_sizes = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
            
            best_d = best_thicks[i]
            best_local_score = best_score
            
            for step in step_sizes:
                for sign in [1, -1]:
                    new_d = best_thicks[i] + sign * step
                    
                    # Check bounds
                    if new_d < 0 or new_d > max_layer_nm:
                        continue
                    
                    # Check total thickness
                    new_total = sum(best_thicks) - best_thicks[i] + new_d
                    if new_total > max_total_nm + 1e-9:
                        continue
                    
                    new_thicks = list(best_thicks)
                    new_thicks[i] = new_d
                    
                    score = _mean_reflectance_percent(best_mats, new_thicks, problem)
                    if score < best_local_score:
                        best_local_score = score
                        best_d = new_d
                        improved = True
            
            if best_d != best_thicks[i]:
                best_thicks[i] = best_d
                best_score = best_local_score
        
        if not improved:
            break
    
    return best_mats, best_thicks, best_score


def _try_add_layer(materials, thicknesses, problem, deadline):
    """
    Try adding a layer and see if it improves the score.
    """
    best_mats = list(materials)
    best_thicks = list(thicknesses)
    best_score = _mean_reflectance_percent(best_mats, best_thicks, problem)
    
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    if len(best_mats) >= max_layers:
        return best_mats, best_thicks, best_score
    
    current_total = sum(best_thicks)
    remaining = max_total_nm - current_total
    if remaining < 1.0:
        return best_mats, best_thicks, best_score
    
    # Try adding a layer at different positions
    # Positions: 0 to len(best_mats)
    for pos in range(len(best_mats) + 1):
        if time.time() > deadline:
            break
        
        for mat_label in ["L", "H"]:
            if time.time() > deadline:
                break
            
            # Try a few thicknesses for the new layer
            for d in [50, 100, 150, 200, 250, 300, 350, 400]:
                if d > remaining or d > max_layer_nm:
                    continue
                
                new_mats = list(best_mats)
                new_thicks = list(best_thicks)
                new_mats.insert(pos, mat_label)
                new_thicks.insert(pos, d)
                
                score = _mean_reflectance_percent(new_mats, new_thicks, problem)
                if score < best_score:
                    best_score = score
                    best_mats = new_mats
                    best_thicks = new_thicks
    
    return best_mats, best_thicks, best_score


def _try_remove_layer(materials, thicknesses, problem, deadline):
    """
    Try removing a layer and see if it improves the score.
    """
    best_mats = list(materials)
    best_thicks = list(thicknesses)
    best_score = _mean_reflectance_percent(best_mats, best_thicks, problem)
    
    if len(best_mats) <= 1:
        return best_mats, best_thicks, best_score
    
    for i in range(len(best_mats)):
        if time.time() > deadline:
            break
        
        new_mats = list(best_mats)
        new_thicks = list(best_thicks)
        new_mats.pop(i)
        new_thicks.pop(i)
        
        score = _mean_reflectance_percent(new_mats, new_thicks, problem)
        if score < best_score:
            best_score = score
            best_mats = new_mats
            best_thicks = new_thicks
    
    return best_mats, best_thicks, best_score


def _swap_layers(materials, thicknesses, problem, deadline):
    """
    Try swapping adjacent layers.
    """
    best_mats = list(materials)
    best_thicks = list(thicknesses)
    best_score = _mean_reflectance_percent(best_mats, best_thicks, problem)
    
    if len(best_mats) < 2:
        return best_mats, best_thicks, best_score
    
    for i in range(len(best_mats) - 1):
        if time.time() > deadline:
            break
        
        # Swap layer i and i+1
        new_mats = list(best_mats)
        new_thicks = list(best_thicks)
        new_mats[i], new_mats[i+1] = new_mats[i+1], new_mats[i]
        new_thicks[i], new_thicks[i+1] = new_thicks[i+1], new_thicks[i]
        
        score = _mean_reflectance_percent(new_mats, new_thicks, problem)
        if score < best_score:
            best_score = score
            best_mats = new_mats
            best_thicks = new_thicks
    
    return best_mats, best_thicks, best_score


def _nelder_mead(materials, thicknesses, problem, deadline):
    """
    Perform Nelder-Mead optimization on the thicknesses.
    """
    best_mats = list(materials)
    best_thicks = list(thicknesses)
    best_score = _mean_reflectance_percent(best_mats, best_thicks, problem)
    
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    n = len(best_thicks)
    if n == 0:
        return best_mats, best_thicks, best_score
    
    # Initialize simplex
    simplex = [list(best_thicks)]
    for i in range(n):
        point = list(best_thicks)
        point[i] = max(0.0, point[i] + 10.0)
        # Check total thickness
        if sum(point) <= max_total_nm + 1e-9:
            simplex.append(point)
        else:
            simplex.append(list(best_thicks))
    
    # Evaluate simplex
    scores = []
    for point in simplex:
        # Clamp to valid range
        valid_point = [min(max(0.0, d), max_layer_nm) for d in point]
        # Check total
        total = sum(valid_point)
        if total > max_total_nm:
            # Scale down
            scale = max_total_nm / total
            valid_point = [d * scale for d in valid_point]
        score = _mean_reflectance_percent(best_mats, valid_point, problem)
        scores.append(score)
    
    # Sort simplex
    for _ in range(n + 1):
        if time.time() > deadline:
            break
        
        order = sorted(range(n + 1), key=lambda i: scores[i])
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Centroid of all but worst
        centroid = [0.0] * n
        for i in range(n):
            for j in range(n):
                centroid[j] += simplex[i][j] / n
        
        # Reflection
        alpha = 1.0
        reflected = [centroid[j] + alpha * (centroid[j] - simplex[n][j]) for j in range(n)]
        
        # Clamp
        valid_reflected = [min(max(0.0, d), max_layer_nm) for d in reflected]
        total = sum(valid_reflected)
        if total > max_total_nm:
            scale = max_total_nm / total
            valid_reflected = [d * scale for d in valid_reflected]
        
        score_reflected = _mean_reflectance_percent(best_mats, valid_reflected, problem)
        
        if score_reflected < scores[0]:
            # Expansion
            gamma = 2.0
            expanded = [centroid[j] + gamma * (centroid[j] - simplex[n][j]) for j in range(n)]
            valid_expanded = [min(max(0.0, d), max_layer_nm) for d in expanded]
            total = sum(valid_expanded)
            if total > max_total_nm:
                scale = max_total_nm / total
                valid_expanded = [d * scale for d in valid_expanded]
            
            score_expanded = _mean_reflectance_percent(best_mats, valid_expanded, problem)
            
            if score_expanded < score_reflected:
                simplex[n] = valid_expanded
                scores[n] = score_expanded
            else:
                simplex[n] = valid_reflected
                scores[n] = score_reflected
        elif score_reflected < scores[n - 1]:
            simplex[n] = valid_reflected
            scores[n] = score_reflected
        else:
            # Contraction
            beta = 0.5
            contracted = [centroid[j] + beta * (centroid[j] - simplex[n][j]) for j in range(n)]
            valid_contracted = [min(max(0.0, d), max_layer_nm) for d in contracted]
            total = sum(valid_contracted)
            if total > max_total_nm:
                scale = max_total_nm / total
                valid_contracted = [d * scale for d in valid_contracted]
            
            score_contracted = _mean_reflectance_percent(best_mats, valid_contracted, problem)
            
            if score_contracted < scores[n]:
                simplex[n] = valid_contracted
                scores[n] = score_contracted
            else:
                # Shrink
                for i in range(1, n + 1):
                    for j in range(n):
                        simplex[i][j] = (simplex[0][j] + simplex[i][j]) / 2.0
                    valid_point = [min(max(0.0, d), max_layer_nm) for d in simplex[i]]
                    total = sum(valid_point)
                    if total > max_total_nm:
                        scale = max_total_nm / total
                        valid_point = [d * scale for d in valid_point]
                    simplex[i] = valid_point
                    scores[i] = _mean_reflectance_percent(best_mats, valid_point, problem)
    
    # Return the best point
    best_idx = np.argmin(scores)
    best_thicks = simplex[best_idx]
    best_score = scores[best_idx]
    
    return best_mats, best_thicks, best_score


def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Leave some margin for the 150s wall-clock limit
    
    # Generate initial design
    materials, thicknesses = _initial_design(problem)
    
    if not _is_valid(materials, thicknesses, problem):
        # Fallback to a simple valid design
        center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        nL = problem["materials"]["L"]
        dL = center_wl / (4.0 * nL)
        if dL > problem["max_layer_nm"]:
            dL = problem["max_layer_nm"]
        return ["L"], [dL]
    
    best_mats = list(materials)
    best_thicks = list(thicknesses)
    best_score = _mean_reflectance_percent(best_mats, best_thicks, problem)
    
    # Main optimization loop
    while time.time() < deadline:
        # Coordinate descent
        mats, thicks, score = _coordinate_descent(best_mats, best_thicks, problem, deadline)
        if score < best_score:
            best_mats = mats
            best_thicks = thicks
            best_score = score
        
        # Nelder-Mead
        mats, thicks, score = _nelder_mead(best_mats, best_thicks, problem, deadline)
        if score < best_score:
            best_mats = mats
            best_thicks = thicks
            best_score = score
        
        # Try adding a layer
        mats, thicks, score = _try_add_layer(best_mats, best_thicks, problem, deadline)
        if score < best_score:
            best_mats = mats
            best_thicks = thicks
            best_score = score
        
        # Try removing a layer
        mats, thicks, score = _try_remove_layer(best_mats, best_thicks, problem, deadline)
        if score < best_score:
            best_mats = mats
            best_thicks = thicks
            best_score = score
        
        # Swap layers
        mats, thicks, score = _swap_layers(best_mats, best_thicks, problem, deadline)
        if score < best_score:
            best_mats = mats
            best_thicks = thicks
            best_score = score
        
        # Final coordinate descent
        mats, thicks, score = _coordinate_descent(best_mats, best_thicks, problem, deadline)
        if score < best_score:
            best_mats = mats
            best_thicks = thicks
            best_score = score
        
        # Check if we're close to the deadline
        if time.time() > deadline - 30.0:
            break
    
    # Ensure validity
    if not _is_valid(best_mats, best_thicks, problem):
        center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        nL = problem["materials"]["L"]
        dL = center_wl / (4.0 * nL)
        if dL > problem["max_layer_nm"]:
            dL = problem["max_layer_nm"]
        return ["L"], [dL]
    
    return best_mats, best_thicks

# EVOLVE-BLOCK-END
