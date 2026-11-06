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
        M11 = new_M11; M12 = new_M12; M21 = new_M21; M22 = new_M22

    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    r = (n_ambient - Y) / (n_ambient + Y)
    return np.abs(r) ** 2


def _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    R = _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers)
    return float(np.mean(R) * 100.0)


def _is_valid(materials, thicknesses, problem):
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
    if total > max_total_nm + 1e-9:
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    for m in materials:
        if m not in problem["materials"]:
            return False
    return True


def _evaluate_design(materials, thicknesses, problem, wavelengths):
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_layers = np.array([problem["materials"][m] for m in materials], dtype=float)
    d_layers = np.array(thicknesses, dtype=float)
    return _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers)


def _objective_lbgfgs(x, mats, n_layers, problem, wavelengths, max_total_nm, max_layer_nm):
    """Objective function for L-BFGS-B optimizer."""
    x = np.clip(x, 0.0, max_layer_nm)
    total = x.sum()
    if total > max_total_nm:
        penalty = (total - max_total_nm) * 0.5
    else:
        penalty = 0.0
    d_layers = x.copy()
    R = _char_matrix_method(wavelengths, problem["ambient_index"], problem["substrate_index"], n_layers, d_layers)
    return float(np.mean(R) * 100.0) + penalty


def _optimize_lbgfgs(mats, x0, problem, wavelengths, deadline, max_iter=2000):
    """Optimize using L-BFGS-B with bounds."""
    n = len(mats)
    if n == 0:
        return list(x0), _evaluate_design(mats, list(x0), problem, wavelengths)
    
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    n_layers = np.array([problem["materials"][m] for m in mats], dtype=float)
    
    x0 = np.clip(np.asarray(x0, dtype=float), 0.0, max_layer_nm)
    bounds = [(0.0, max_layer_nm)] * n
    
    def obj(x):
        if time.time() > deadline:
            return float('inf')
        return _objective_lbgfgs(x, mats, n_layers, problem, wavelengths, max_total_nm, max_layer_nm)
    
    result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    best_x = np.clip(result.x, 0.0, max_layer_nm)
    best_f = _evaluate_design(mats, list(best_x), problem, wavelengths)
    return list(best_x), best_f


def _polish_thicknesses(mats, thicks, problem, wavelengths, deadline, max_iter=4000):
    """High-precision coordinate polish with adaptive step sizes."""
    current_thicks = list(thicks)
    current_score = _evaluate_design(mats, current_thicks, problem, wavelengths)
    n = len(mats)
    if n == 0:
        return mats, current_thicks, current_score
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    step_sizes = [2.0] * n
    min_step = 0.005
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            best_local_score = current_score
            best_local_thick = current_thicks[i]
            for factor in [1.0, 0.5, 0.25, 0.1]:
                step = step_sizes[i] * factor
                for sign in (1, -1):
                    new_thick = current_thicks[i] + sign * step
                    if new_thick < 0:
                        new_thick = 0.0
                    if new_thick > max_layer_nm:
                        new_thick = max_layer_nm
                    new_thicks = list(current_thicks)
                    new_thicks[i] = new_thick
                    if _is_valid(mats, new_thicks, problem):
                        new_score = _evaluate_design(mats, new_thicks, problem, wavelengths)
                        if new_score < best_local_score:
                            best_local_score = new_score
                            best_local_thick = new_thick
            if best_local_score < current_score:
                current_thicks[i] = best_local_thick
                current_score = best_local_score
                improved = True
        if not improved:
            step_sizes = [s * 0.6 for s in step_sizes]
            if min(step_sizes) < min_step:
                break
        else:
            step_sizes = [min(s * 1.1, 8.0) for s in step_sizes]
    return mats, current_thicks, current_score


def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s wall-clock kill

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

    def qw(m):
        return centre / (4.0 * materials_dict[m])

    # --- Build a rich set of diverse initial seeds ---
    initial_designs = []
    # single layers
    initial_designs.append((["L"], [qw("L")]))
    initial_designs.append((["H"], [qw("H")]))
    # two layers
    initial_designs.append((["L", "H"], [qw("L"), qw("H")]))
    initial_designs.append((["H", "L"], [qw("H"), qw("L")]))
    # three layers
    initial_designs.append((["H", "L", "H"], [qw("H"), qw("L"), qw("H")]))
    initial_designs.append((["L", "H", "L"], [qw("L"), qw("H"), qw("L")]))
    # four layers
    initial_designs.append((["H", "L", "H", "L"], [qw("H"), qw("L"), qw("H"), qw("L")]))
    initial_designs.append((["L", "H", "L", "H"], [qw("L"), qw("H"), qw("L"), qw("H")]))
    # five layers
    initial_designs.append((["H", "L", "H", "L", "H"], [qw("H"), qw("L"), qw("H"), qw("L"), qw("H")]))
    initial_designs.append((["L", "H", "L", "H", "L"], [qw("L"), qw("H"), qw("L"), qw("H"), qw("L")]))
    # six layers
    initial_designs.append((["H", "L", "H", "L", "H", "L"], [qw("H"), qw("L"), qw("H"), qw("L"), qw("H"), qw("L")]))
    # seven layers
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H"], [qw("H"), qw("L"), qw("H"), qw("L"), qw("H"), qw("L"), qw("H")]))
    # eight layers
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L"], [qw("H"), qw("L"), qw("H"), qw("L"), qw("H"), qw("L"), qw("H"), qw("L")]))
    # nine layers
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H"], [qw("H"), qw("L"), qw("H"), qw("L"), qw("H"), qw("L"), qw("H"), qw("L"), qw("H")]))
    # ten layers
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], [qw("H"), qw("L"), qw("H"), qw("L"), qw("H"), qw("L"), qw("H"), qw("L"), qw("H"), qw("L")]))
    # non-alternating patterns
    initial_designs.append((["H", "L", "L", "H"], [qw("H"), qw("L"), qw("L"), qw("H")]))
    initial_designs.append((["L", "H", "H", "L"], [qw("L"), qw("H"), qw("H"), qw("L")]))
    initial_designs.append((["H", "L", "L", "H", "L", "L", "H"], [qw("H"), qw("L"), qw("L"), qw("H"), qw("L"), qw("L"), qw("H")]))
    initial_designs.append((["H", "L", "L"], [qw("H"), qw("L"), qw("L")]))
    # classic 10-layer H|LHLHLHLHL|H quarter-wave broadband design
    inner = ["L", "H", "L", "H", "L", "H", "L", "H", "L"]
    initial_designs.append((["H"] + inner + ["H"], [qw("H")] + [qw(m) for m in inner] + [qw("H")]))

    # --- Evaluate all seeds, keep the best ---
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

    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [qw("L")]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem, wavelengths)

    # --- Optimize the best seed with L-BFGS-B, then polish ---
    best_thicknesses, best_score = _optimize_lbgfgs(best_materials, best_thicknesses, problem, wavelengths, deadline)
    if time.time() < deadline - 3.0:
        _, best_thicknesses, best_score = _polish_thicknesses(best_materials, best_thicknesses, problem, wavelengths, deadline)

    # --- Try other promising sequences with shorter optimization ---
    candidate_sequences = []
    for n_layers in range(1, min(15, max_layers + 1)):
        seq = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        candidate_sequences.append(seq)
        seq = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        candidate_sequences.append(seq)
    for n_layers in [4, 8, 12]:
        if n_layers % 4 == 0:
            seq = [("H" if (i % 4 in (0, 3)) else "L") for i in range(n_layers)]
            candidate_sequences.append(seq)
    candidate_sequences.append(["H", "L", "L", "H", "L", "L", "H"])
    candidate_sequences.append(["L", "H", "H", "L", "H", "H", "L"])
    candidate_sequences.append(["H", "L", "H", "L", "L", "H", "L", "H"])
    candidate_sequences.append(["L", "H", "L", "H", "H", "L", "H", "L"])
    inner = ["L", "H", "L", "H", "L", "H", "L", "H", "L"]
    candidate_sequences.append(["H"] + inner + ["H"])

    seen = set()
    unique_candidates = []
    for seq in candidate_sequences:
        if len(seq) > max_layers:
            continue
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(seq)

    for seq in unique_candidates:
        if time.time() > deadline - 10.0:
            break
        if seq == best_materials:
            continue
        thicks = [qw(m) for m in seq]
        if not _is_valid(seq, thicks, problem):
            continue
        score = _evaluate_design(seq, thicks, problem, wavelengths)
        if score < best_score + 3.0:
            thicks, score = _optimize_lbgfgs(seq, thicks, problem, wavelengths, deadline - 5.0, max_iter=1000)
            if time.time() < deadline - 4.0:
                _, thicks, score = _polish_thicknesses(seq, thicks, problem, wavelengths, deadline - 3.0, max_iter=1500)
            if score < best_score:
                best_score = score
                best_materials = list(seq)
                best_thicknesses = list(thicks)

    # --- Final refinement on the best design ---
    if time.time() < deadline - 5.0:
        best_thicknesses, best_score = _optimize_lbgfgs(best_materials, best_thicknesses, problem, wavelengths, deadline - 2.0, max_iter=1000)
    # High-precision polish pass
    if time.time() < deadline - 2.0:
        _, best_thicknesses, best_score = _polish_thicknesses(best_materials, best_thicknesses, problem, wavelengths, deadline - 0.5, max_iter=4000)

    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [qw("L")]

    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
