"""Broadband antireflection coating design using characteristic-matrix method."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    """
    Compute reflectance (0-1) for a stack using the characteristic matrix method.
    """
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
        if t < 0 or t > max_layer_nm:
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


def _nelder_mead(objective, x0, bounds, deadline, max_iter=2000, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """
    Nelder-Mead simplex optimization.
    objective: function to minimize
    x0: initial point (list or array)
    bounds: list of (min, max) for each dimension
    deadline: time limit
    Returns: (best_x, best_f)
    """
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    
    # Create initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.01
        xi[i] += step
        xi[i] = np.clip(xi[i], bounds[i][0], bounds[i][1])
        simplex.append(xi)
    
    # Evaluate initial simplex
    fvals = [objective(x) for x in simplex]
    
    best_x = simplex[0].copy()
    best_f = fvals[0]
    for i in range(1, n + 1):
        if fvals[i] < best_f:
            best_f = fvals[i]
            best_x = simplex[i].copy()
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:n], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[n])
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[n])
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = objective(xe)
            if fe < fr:
                simplex[n] = xe
                fvals[n] = fe
            else:
                simplex[n] = xr
                fvals[n] = fr
        elif fr < fvals[n - 1]:
            simplex[n] = xr
            fvals[n] = fr
        else:
            # Contraction
            if fr < fvals[n]:
                # Outside contraction
                xc = centroid + sigma * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - sigma * (xr - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = objective(xc)
            if fc < fvals[n]:
                simplex[n] = xc
                fvals[n] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = objective(simplex[i])
        
        # Update best
        for i in range(n + 1):
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()
    
    return best_x, best_f


def _optimize_thicknesses(mats, thicks, problem, wavelengths, deadline, max_iter=2000):
    """Optimize thicknesses for a given layer sequence using Nelder-Mead."""
    current_thicks = list(thicks)
    current_score = _evaluate_design(mats, current_thicks, problem, wavelengths)
    
    n = len(mats)
    if n == 0:
        return mats, current_thicks, current_score
    
    # Define objective function
    def objective(x):
        x = np.clip(x, 0.0, 400.0)
        if np.sum(x) > problem["max_total_nm"]:
            return current_score + 10.0
        return _evaluate_design(mats, list(x), problem, wavelengths)
    
    bounds = [(0.0, 400.0)] * n
    x0 = np.array(current_thicks, dtype=float)
    
    best_x, best_f = _nelder_mead(objective, x0, bounds, deadline, max_iter=max_iter)
    best_x = np.clip(best_x, 0.0, 400.0)
    
    # Final validation
    if not _is_valid(mats, list(best_x), problem):
        return mats, current_thicks, current_score
    
    return mats, list(best_x), best_f


def _polish_thicknesses(mats, thicks, problem, wavelengths, deadline, max_iter=3000):
    """High-precision polish using Nelder-Mead with smaller initial step."""
    current_thicks = list(thicks)
    current_score = _evaluate_design(mats, current_thicks, problem, wavelengths)
    
    n = len(mats)
    if n == 0:
        return mats, current_thicks, current_score
    
    def objective(x):
        x = np.clip(x, 0.0, 400.0)
        if np.sum(x) > problem["max_total_nm"]:
            return current_score + 10.0
        return _evaluate_design(mats, list(x), problem, wavelengths)
    
    bounds = [(0.0, 400.0)] * n
    x0 = np.array(current_thicks, dtype=float)
    
    # Use smaller step for polish
    best_x, best_f = _nelder_mead(objective, x0, bounds, deadline, max_iter=max_iter, alpha=0.8, gamma=1.5, rho=0.4, sigma=0.4)
    best_x = np.clip(best_x, 0.0, 400.0)
    
    if not _is_valid(mats, list(best_x), problem):
        return mats, current_thicks, current_score
    
    return mats, list(best_x), best_f


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0
    
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
    
    # Single layer designs
    initial_designs.append((["L"], [centre / (4.0 * n_L)]))
    initial_designs.append((["H"], [centre / (4.0 * n_H)]))
    
    # Two-layer designs
    initial_designs.append((["L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Three-layer designs
    initial_designs.append((["H", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Four-layer designs
    initial_designs.append((["H", "L", "H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    initial_designs.append((["L", "H", "L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # Five-layer designs
    initial_designs.append((["H", "L", "H", "L", "H"], 
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), 
                            centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["L", "H", "L", "H", "L"], 
                           [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), 
                            centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Six-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Seven-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H)]))
    
    # Eight-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Nine-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # Ten-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L)]))
    
    # Non-standard patterns
    initial_designs.append((["H", "L", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["H", "L", "L", "H", "L", "L", "H"], 
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), 
                            centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # Evaluate all initial designs and keep the best
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    for mats, thicks in initial_designs:
        if not _is_valid(mats, thicks, problem):
            continue
        score = _evaluate_design(mats, thicks, problem, wavelengths)
        if score < best_score:
            best_score = score
            best_materials = list(mats)
            best_thicknesses = list(thicks)
    
    # Fallback if no valid initial design
    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem, wavelengths)
    
    # Optimize the best initial design
    best_materials, best_thicknesses, best_score = _optimize_thicknesses(
        best_materials, best_thicknesses, problem, wavelengths, deadline, max_iter=2000
    )
    
    # Try other promising sequences with shorter optimization
    candidate_sequences = []
    
    # Alternating sequences of various lengths
    for n_layers in range(1, min(15, max_layers + 1)):
        # Starting with H
        seq = []
        for i in range(n_layers):
            seq.append("H" if i % 2 == 0 else "L")
        candidate_sequences.append(seq)
        
        # Starting with L
        seq = []
        for i in range(n_layers):
            seq.append("L" if i % 2 == 0 else "H")
        candidate_sequences.append(seq)
    
    # Some non-alternating patterns
    candidate_sequences.append(["H", "L", "L", "H", "L", "L", "H"])
    candidate_sequences.append(["L", "H", "H", "L", "H", "H", "L"])
    candidate_sequences.append(["H", "L", "H", "L", "L", "H", "L", "H"])
    candidate_sequences.append(["L", "H", "L", "H", "H", "L", "H", "L"])
    
    # Deduplicate and filter
    seen = set()
    unique_candidates = []
    for seq in candidate_sequences:
        if len(seq) > max_layers:
            continue
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(seq)
    
    # Evaluate and optimize each candidate
    for seq in unique_candidates:
        if time.time() > deadline:
            break
        
        if seq == best_materials:
            continue
        
        thicks = [centre / (4.0 * materials_dict[m]) for m in seq]
        
        if not _is_valid(seq, thicks, problem):
            continue
        
        score = _evaluate_design(seq, thicks, problem, wavelengths)
        
        # Only optimize if promising
        if score < best_score + 3.0:
            mats, thicks, score = _optimize_thicknesses(
                seq, thicks, problem, wavelengths, deadline, max_iter=1500
            )
            if score < best_score:
                best_score = score
                best_materials = list(seq)
                best_thicknesses = list(thicks)
    
    # Final refinement on the best design
    if time.time() < deadline - 5.0:
        best_materials, best_thicknesses, best_score = _optimize_thicknesses(
            best_materials, best_thicknesses, problem, wavelengths, deadline, max_iter=2000
        )
    
    # High-precision polish pass
    if time.time() < deadline - 2.0:
        best_materials, best_thicknesses, best_score = _polish_thicknesses(
            best_materials, best_thicknesses, problem, wavelengths, deadline, max_iter=3000
        )
    
    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
