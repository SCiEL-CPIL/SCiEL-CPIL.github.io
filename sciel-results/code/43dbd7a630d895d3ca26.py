"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _char_matrix(n, d, wl):
    """Compute the characteristic matrix for a single layer at wavelength wl (vectorized over wl array)."""
    beta = 2.0 * np.pi * n * d / wl
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    # Return array of matrices: shape (N, 2, 2)
    m = np.zeros((len(wl), 2, 2), dtype=complex)
    m[:, 0, 0] = cos_b
    m[:, 0, 1] = 1.0j * sin_b / n
    m[:, 1, 0] = n * 1.0j * sin_b
    m[:, 1, 1] = cos_b
    return m


def _mean_reflectance_vec(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band using vectorized computation."""
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

    # Build total characteristic matrix M for all wavelengths simultaneously
    # M has shape (401, 2, 2)
    M = np.eye(2, dtype=complex)
    M = np.tile(M, (len(wavelengths), 1, 1))

    for mat, thick in zip(materials, thicknesses):
        if thick == 0:
            continue
        n = mat_dict[mat]
        layer_M = _char_matrix(n, thick, wavelengths)
        # M = M @ layer_M for each wavelength
        M = M @ layer_M

    # Substrate admittance Ys = ns
    # Total admittance Y = M[1,1] / M[0,1]
    Y = M[:, 1, 1] / M[:, 0, 1]

    # Reflectance
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r) ** 2

    return float(np.mean(R) * 100.0)


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


def _coordinate_descent(materials, thicknesses, problem, deadline, max_iter=300):
    """Refine thicknesses using coordinate descent with line search."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat_dict = problem["materials"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    n_layers = len(materials)

    if n_layers == 0:
        return materials, thicknesses

    best_thicks = list(thicknesses)
    best_score = _mean_reflectance_vec(materials, best_thicks, problem)

    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]
    current_thicks = list(best_thicks)

    for iteration in range(max_iter):
        if time.time() > deadline:
            break

        improved = False
        for i in range(n_layers):
            if time.time() > deadline:
                break

            for direction in [-1, 1]:
                for frac in [1.0, 0.5, 0.25, 0.1, 0.05]:
                    # Try adjusting thickness i
                    old_val = current_thicks[i]
                    # Use a step proportional to current value, or absolute if small
                    base_step = max(1.0, abs(old_val) * 0.1)
                    delta = direction * base_step * frac

                    new_val = old_val + delta
                    if new_val < 0 or new_val > max_layer_nm:
                        continue

                    # Check total thickness
                    new_total = sum(current_thicks) - old_val + new_val
                    if new_total > max_total_nm:
                        continue

                    new_thicks = list(current_thicks)
                    new_thicks[i] = new_val

                    score = _mean_reflectance_vec(materials, new_thicks, problem)
                    if score < best_score:
                        best_score = score
                        best_thicks = new_thicks
                        current_thicks = new_thicks
                        improved = True
                        break
                else:
                    continue
                break

        if not improved:
            # No improvement in this full pass; try smaller steps or break
            break

    return materials, best_thicks


def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit (evaluator kills at 120s wall, but keep safe)

    mat_dict = problem["materials"]
    nL = mat_dict["L"]
    nH = mat_dict["H"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])

    # Quarter-wave thicknesses
    dL = centre / (4.0 * nL)
    dH = centre / (4.0 * nH)

    candidates = []

    # Various symmetric stacks starting with L and ending with L
    for n_layers in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
        if n_layers > max_layers:
            break
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append("L")
                thicks.append(dL)
            else:
                mats.append("H")
                thicks.append(dH)
        if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
            candidates.append((mats, thicks))

    # Also try some asymmetric patterns
    # L-H-L-H pattern (ending with H)
    for n_layers in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            break
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append("L")
                thicks.append(dL)
            else:
                mats.append("H")
                thicks.append(dH)
        if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
            candidates.append((mats, thicks))

    # Try with slightly different center wavelengths
    for wl_center in [500, 550, 600]:
        dL_c = wl_center / (4.0 * nL)
        dH_c = wl_center / (4.0 * nH)
        for n_layers in [3, 5, 7, 9]:
            if n_layers > max_layers:
                continue
            mats = []
            thicks = []
            for i in range(n_layers):
                if i % 2 == 0:
                    mats.append("L")
                    thicks.append(dL_c)
                else:
                    mats.append("H")
                    thicks.append(dH_c)
            if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
                candidates.append((mats, thicks))

    # Evaluate all candidates and find the best
    best_score = float('inf')
    best_result = candidates[0] if candidates else (["L"], [dL])

    for mats, thicks in candidates:
        if time.time() > deadline:
            break
        if not _is_valid(mats, thicks, problem):
            continue
        score = _mean_reflectance_vec(mats, thicks, problem)
        if score < best_score:
            best_score = score
            best_result = (mats, list(thicks))

    # Coordinate descent refinement on the best candidate
    if time.time() < deadline - 5.0:
        mats, thicks = best_result
        mats_refined, thicks_refined = _coordinate_descent(mats, thicks, problem, deadline)
        score_refined = _mean_reflectance_vec(mats_refined, thicks_refined, problem)
        if score_refined < best_score:
            best_score = score_refined
            best_result = (mats_refined, thicks_refined)

    # Try coordinate descent on the second-best candidate too
    if len(candidates) > 1 and time.time() < deadline - 5.0:
        # Sort candidates by score (re-evaluate to be safe, or just try the first few)
        second_candidates = []
        for mats, thicks in candidates:
            if mats == best_result[0] and thicks == best_result[1]:
                continue
            if not _is_valid(mats, thicks, problem):
                continue
            score = _mean_reflectance_vec(mats, thicks, problem)
            second_candidates.append((score, mats, thicks))
            if time.time() > deadline - 10.0:
                break

        second_candidates.sort(key=lambda x: x[0])
        if second_candidates:
            _, mats2, thicks2 = second_candidates[0]
            mats2_refined, thicks2_refined = _coordinate_descent(mats2, thicks2, problem, deadline)
            score2_refined = _mean_reflectance_vec(mats2_refined, thicks2_refined, problem)
            if score2_refined < best_score:
                best_score = score2_refined
                best_result = (mats2_refined, thicks2_refined)

    return best_result
# EVOLVE-BLOCK-END
