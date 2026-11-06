"""Broadband antireflection coating design: Nelder-Mead + multi-start optimization."""
import time
import numpy as np

# EVOLVE-BLOCK-START


def _mean_reflectance_fast(materials, thicknesses, problem, wavelengths):
    """Vectorized mean reflectance (percent) over given wavelengths."""
    if len(materials) == 0:
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r

    mat_dict = problem["materials"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]

    n_wl = len(wavelengths)

    n_vals = np.array([mat_dict[m] for m in materials], dtype=float)
    d_vals = np.array(thicknesses, dtype=float)

    # Initialize total characteristic matrix as identity for each wavelength
    M11 = np.ones(n_wl, dtype=complex)
    M12 = np.zeros(n_wl, dtype=complex)
    M21 = np.zeros(n_wl, dtype=complex)
    M22 = np.ones(n_wl, dtype=complex)

    for n, d in zip(n_vals, d_vals):
        if d == 0.0:
            continue
        beta = 2.0 * np.pi * n * d / wavelengths
        c = np.cos(beta)
        s = np.sin(beta)

        # Layer matrix: [[c, i*s/n], [i*n*s, c]]
        L11 = c
        L12 = 1j * s / n
        L21 = 1j * n * s
        L22 = c

        # M_total = M_total @ M_layer
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22

        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22

    # Admittance Y = (M11 + M12 * n_sub) / (M21 + M22 * n_sub)
    Y = (M11 + M12 * ns) / (M21 + M22 * ns)
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r) ** 2
    return float(np.mean(R) * 100.0)


def _is_valid(materials, thicknesses, problem):
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    for m in materials:
        if m not in ('L', 'H'):
            return False
    return True


def _nelder_mead(objective, x0, bounds, max_iter=200, tol=1e-8):
    """Simple Nelder-Mead simplex optimization."""
    n = len(x0)
    # Initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        perturbation = np.zeros(n)
        perturbation[i] = 0.1 * (bounds[i][1] - bounds[i][0])
        simplex.append(np.clip(np.array(x0) + perturbation, bounds[:, 0], bounds[:, 1]))

    # Evaluate initial simplex
    values = [objective(v) for v in simplex]

    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    for _ in range(max_iter):
        # Sort by objective value
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        if abs(values[-1] - values[0]) < tol:
            break

        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, bounds[:, 0], bounds[:, 1])
        fr = objective(xr)

        if fr < values[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            xe = np.clip(xe, bounds[:, 0], bounds[:, 1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            xc = np.clip(xc, bounds[:, 0], bounds[:, 1])
            fc = objective(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds[:, 0], bounds[:, 1])
                    values[i] = objective(simplex[i])

    # Return best point
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]


def _coordinate_descent(objective, x0, bounds, max_iter=500, tol=1e-8, step_factor=0.5):
    """Coordinate descent with adaptive step size."""
    n = len(x0)
    x = np.array(x0, dtype=float)
    f = objective(x)
    step = [0.5 * (b[1] - b[0]) for b in bounds]
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            best_f = f
            best_x = x.copy()
            for sign in [-1, 1]:
                for trial in range(10):
                    new_x = x.copy()
                    new_x[i] = x[i] + sign * step[i]
                    new_x[i] = np.clip(new_x[i], bounds[i][0], bounds[i][1])
                    new_f = objective(new_x)
                    if new_f < best_f - tol:
                        best_f = new_f
                        best_x = new_x.copy()
                        improved = True
                    else:
                        step[i] *= step_factor
                        break
            if best_f < f:
                x = best_x
                f = best_f
                step[i] = max(step[i] * 1.5, 0.1)
        if not improved:
            break
    return x, f


def _generate_initial_candidates(problem):
    """Generate diverse initial candidates for optimization."""
    mat_dict = problem["materials"]
    nL = mat_dict["L"]
    nH = mat_dict["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])

    dL_qw = centre / (4.0 * nL)
    dH_qw = centre / (4.0 * nH)

    candidates = []

    # Various alternating patterns
    for n in range(1, min(max_layers, 15) + 1):
        # L-first alternating
        mats = ["L" if i % 2 == 0 else "H" for i in range(n)]
        thicks = [dL_qw if i % 2 == 0 else dH_qw for i in range(n)]
        if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
            candidates.append((mats, list(thicks)))

        # H-first alternating
        if n > 1:
            mats = ["H" if i % 2 == 0 else "L" for i in range(n)]
            thicks = [dH_qw if i % 2 == 0 else dL_qw for i in range(n)]
            if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
                candidates.append((mats, list(thicks)))

    # Geometric thickness variation
    for n in [5, 7, 9, 11]:
        if n > max_layers:
            continue
        mats = ["L" if i % 2 == 0 else "H" for i in range(n)]
        for factor in [0.6, 0.8, 1.2, 1.5]:
            thicks = []
            for i in range(n):
                base = dL_qw if i % 2 == 0 else dH_qw
                pos = i / max(1, n - 1)
                taper = 1.0 + (factor - 1.0) * (1.0 - abs(pos - 0.5) * 2.0)
                t = base * taper
                thicks.append(min(max(t, 0.0), max_layer_nm))
            if sum(thicks) <= max_total_nm:
                candidates.append((mats, thicks))

    # Different center wavelengths
    for wl_c in [450, 500, 550, 600]:
        dL_c = wl_c / (4.0 * nL)
        dH_c = wl_c / (4.0 * nH)
        for n in [5, 7, 9]:
            if n > max_layers:
                continue
            mats = ["L" if i % 2 == 0 else "H" for i in range(n)]
            thicks = [dL_c if i % 2 == 0 else dH_c for i in range(n)]
            if sum(thicks) <= max_total_nm and all(0 <= t <= max_layer_nm for t in thicks):
                candidates.append((mats, list(thicks)))

    # Non-periodic patterns with random-like thicknesses
    rng = np.random.default_rng(42)
    for n in [6, 8, 10, 12]:
        if n > max_layers:
            continue
        for trial in range(3):
            mats = []
            thicks = []
            for i in range(n):
                if i == 0:
                    mat = 'L'
                else:
                    # Avoid same material in adjacent layers
                    if mats[-1] == 'L':
                        mat = 'H'
                    else:
                        mat = 'L'
                mats.append(mat)
                base = dL_qw if mat == 'L' else dH_qw
                t = base * (0.7 + 0.6 * rng.random())
                thicks.append(min(max(t, 0.0), max_layer_nm))
            if sum(thicks) <= max_total_nm:
                candidates.append((mats, thicks))

    # Deduplicate
    seen = set()
    unique_candidates = []
    for mats, thicks in candidates:
        key = tuple(mats) + tuple(round(t, 1) for t in thicks)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((mats, list(thicks)))

    return unique_candidates


def _optimize_candidate(materials, thicknesses, problem, wavelengths, deadline, fast_wl):
    """Optimize a single candidate using Nelder-Mead followed by coordinate descent."""
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses, _mean_reflectance_fast(materials, thicknesses, problem, wavelengths)

    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]

    def objective(thicks):
        if time.time() > deadline:
            return 100.0
        # Check constraints
        if any(t < -1e-9 or t > max_layer_nm + 1e-9 for t in thicks):
            return 100.0
        if sum(thicks) > max_total_nm + 1e-9:
            return 100.0
        return _mean_reflectance_fast(materials, list(thicks), problem, fast_wl)

    bounds = [(0.0, max_layer_nm)] * n_layers
    x0 = np.array(thicknesses, dtype=float)

    try:
        # Phase 1: Nelder-Mead for global-ish search
        x_opt, f_opt = _nelder_mead(objective, x0, bounds, max_iter=150, tol=1e-7)
        x_opt = np.clip(x_opt, 0.0, max_layer_nm)
        total = np.sum(x_opt)
        if total > max_total_nm:
            x_opt *= max_total_nm / total

        # Phase 2: Coordinate descent for fine-tuning
        x_cd, f_cd = _coordinate_descent(objective, x_opt, bounds, max_iter=300, tol=1e-9, step_factor=0.5)
        x_cd = np.clip(x_cd, 0.0, max_layer_nm)
        total = np.sum(x_cd)
        if total > max_total_nm:
            x_cd *= max_total_nm / total

        # Use the better result
        if f_cd < f_opt:
            x_opt = x_cd

        final_score = _mean_reflectance_fast(materials, list(x_opt), problem, wavelengths)
        return materials, list(x_opt), final_score
    except Exception:
        return materials, thicknesses, _mean_reflectance_fast(materials, thicknesses, problem, wavelengths)


def _local_search_with_layer_changes(materials, thicknesses, problem, wavelengths, deadline, fast_wl):
    """Local search that can add/remove layers."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    dL_qw = centre / (4.0 * nL)
    dH_qw = centre / (4.0 * nH)

    best_mats = materials[:]
    best_thicks = thicknesses[:]
    best_score = _mean_reflectance_fast(best_mats, best_thicks, problem, wavelengths)

    for iteration in range(20):
        if time.time() > deadline:
            break
        improved = False
        n_layers = len(best_mats)

        # Try adding a layer
        if n_layers < max_layers:
            for pos in range(n_layers + 1):
                for new_mat in ['L', 'H']:
                    # Avoid same material in adjacent layers
                    if pos > 0 and best_mats[pos - 1] == new_mat:
                        continue
                    if pos < n_layers and best_mats[pos] == new_mat:
                        continue

                    new_mats = best_mats[:]
                    new_mats.insert(pos, new_mat)
                    new_thicks = best_thicks[:]
                    base = dL_qw if new_mat == 'L' else dH_qw
                    new_thicks.insert(pos, base)

                    if sum(new_thicks) > max_total_nm:
                        # Scale down
                        scale = max_total_nm / sum(new_thicks)
                        new_thicks = [t * scale for t in new_thicks]

                    if not _is_valid(new_mats, new_thicks, problem):
                        continue

                    score = _mean_reflectance_fast(new_mats, new_thicks, problem, fast_wl)
                    if score < best_score - 1e-6:
                        best_mats = new_mats
                        best_thicks = new_thicks
                        best_score = score
                        improved = True

        # Try removing a layer
        if n_layers > 1:
            for pos in range(n_layers):
                new_mats = best_mats[:]
                new_thicks = best_thicks[:]
                new_mats.pop(pos)
                new_thicks.pop(pos)

                if not _is_valid(new_mats, new_thicks, problem):
                    continue

                score = _mean_reflectance_fast(new_mats, new_thicks, problem, fast_wl)
                if score < best_score - 1e-6:
                    best_mats = new_mats
                    best_thicks = new_thicks
                    best_score = score
                    improved = True

        if not improved:
            break

    return best_mats, best_thicks, _mean_reflectance_fast(best_mats, best_thicks, problem, wavelengths)


def design(problem):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    mat_dict = problem["materials"]
    nL = mat_dict["L"]
    nH = mat_dict["H"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    centre = 0.5 * (band[0] + band[1])

    # Fast wavelength grid for optimization
    n_wl = len(wavelengths)
    if n_wl > 101:
        step_idx = max(1, n_wl // 101)
        fast_wl = wavelengths[::step_idx]
    else:
        fast_wl = wavelengths

    # Generate initial candidates
    candidates = _generate_initial_candidates(problem)

    if not candidates:
        # Fallback
        dL_qw = centre / (4.0 * nL)
        return ["L"], [dL_qw]

    # Evaluate all candidates
    scored = []
    for mats, thicks in candidates:
        if time.time() > deadline - 30.0:
            break
        if not _is_valid(mats, thicks, problem):
            continue
        score = _mean_reflectance_fast(mats, thicks, problem, wavelengths)
        scored.append((score, mats, thicks))

    scored.sort(key=lambda x: x[0])

    if not scored:
        dL_qw = centre / (4.0 * nL)
        return ["L"], [dL_qw]

    best_score, best_mats, best_thicks = scored[0]

    # Optimize top candidates with Nelder-Mead + coordinate descent
    n_to_optimize = min(6, len(scored))
    for i in range(n_to_optimize):
        if time.time() > deadline - 15.0:
            break
        mats_c, thicks_c = scored[i][1], scored[i][2]
        opt_mats, opt_thicks, opt_score = _optimize_candidate(
            mats_c, thicks_c, problem, wavelengths, deadline - 5.0, fast_wl
        )
        if _is_valid(opt_mats, opt_thicks, problem):
            final_score = _mean_reflectance_fast(opt_mats, opt_thicks, problem, wavelengths)
            if final_score < best_score:
                best_score = final_score
                best_mats = opt_mats
                best_thicks = opt_thicks

    # Local search with layer add/remove on the best design
    if time.time() < deadline - 10.0:
        ls_mats, ls_thicks, ls_score = _local_search_with_layer_changes(
            best_mats, best_thicks, problem, wavelengths, deadline - 5.0, fast_wl
        )
        if _is_valid(ls_mats, ls_thicks, problem) and ls_score < best_score:
            best_score = ls_score
            best_mats = ls_mats
            best_thicks = ls_thicks

    # Final polish: coordinate descent with tighter tolerance
    if time.time() < deadline - 5.0 and len(best_mats) > 0:
        def final_objective(thicks):
            if any(t < -1e-9 or t > max_layer_nm + 1e-9 for t in thicks):
                return 100.0
            if sum(thicks) > max_total_nm + 1e-9:
                return 100.0
            return _mean_reflectance_fast(best_mats, list(thicks), problem, fast_wl)

        bounds = [(0.0, max_layer_nm)] * len(best_mats)
        x0 = np.array(best_thicks, dtype=float)
        try:
            x_polish, f_polish = _coordinate_descent(final_objective, x0, bounds, max_iter=200, tol=1e-10, step_factor=0.5)
            x_polish = np.clip(x_polish, 0.0, max_layer_nm)
            total = np.sum(x_polish)
            if total > max_total_nm:
                x_polish *= max_total_nm / total
            final_score = _mean_reflectance_fast(best_mats, list(x_polish), problem, wavelengths)
            if final_score < best_score:
                best_score = final_score
                best_thicks = list(x_polish)
        except Exception:
            pass

    # Ensure validity
    if not _is_valid(best_mats, best_thicks, problem):
        best_mats, best_thicks = scored[0][1], scored[0][2]

    return best_mats, best_thicks

# EVOLVE-BLOCK-END
