"""Broadband antireflection coating design with DE + Nelder-Mead optimization."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START


def _reflectance_spectrum(n_vals, d_vals, lambdas, n_ambient, n_substrate):
    """Vectorized reflectance computation for a stack of layers."""
    N = len(n_vals)
    W = len(lambdas)

    M11 = np.ones(W, dtype=complex)
    M12 = np.zeros(W, dtype=complex)
    M21 = np.zeros(W, dtype=complex)
    M22 = np.ones(W, dtype=complex)

    for j in range(N):
        n = n_vals[j]
        d = d_vals[j]
        beta = 2.0 * np.pi * n * d / lambdas
        c = np.cos(beta)
        s = np.sin(beta)

        L11 = c
        L12 = 1j * s / n
        L21 = 1j * n * s
        L22 = c

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


def _evaluate(n_vals, d_vals, lambdas, n_ambient, n_substrate):
    """Compute mean reflectance in percent."""
    R = _reflectance_spectrum(n_vals, d_vals, lambdas, n_ambient, n_substrate)
    return 100.0 * np.mean(R)


def _is_valid(d_vals, max_total_nm, max_layer_nm):
    """Check thickness constraints."""
    if len(d_vals) == 0:
        return True
    total = sum(d_vals)
    if total > max_total_nm + 1e-6:
        return False
    for t in d_vals:
        if t < -1e-6 or t > max_layer_nm + 1e-6:
            return False
    return True


def _n_from_materials(materials, n_L, n_H):
    """Convert material labels to refractive index array."""
    return np.array([n_L if m == 'L' else n_H for m in materials])


def design(problem):
    """
    Design a broadband antireflection coating using differential evolution
    and Nelder-Mead local refinement.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    lambdas = np.linspace(band[0], band[1], 401)
    center_lam = 0.5 * (band[0] + band[1])

    def time_ok():
        return time.time() < deadline - 3.0

    def eval_design(materials, thicknesses):
        """Evaluate a design, return mean reflectance in percent or inf if invalid."""
        if len(materials) > max_layers:
            return float('inf')
        d = np.array(thicknesses, dtype=float)
        if not _is_valid(d, max_total_nm, max_layer_nm):
            return float('inf')
        n_vals = _n_from_materials(materials, n_L, n_H)
        return _evaluate(n_vals, d, lambdas, n_ambient, n_substrate)

    # Generate initial candidate designs (quarter-wave and variants)
    candidates = []

    for n_layers in range(1, 13):
        for start_mat in ['L', 'H']:
            materials = []
            thicknesses = []
            for i in range(n_layers):
                mat = start_mat if (i % 2 == 0) else ('H' if start_mat == 'L' else 'L')
                materials.append(mat)
                n = n_L if mat == 'L' else n_H
                thicknesses.append(center_lam / (4.0 * n))
            candidates.append((materials[:], thicknesses[:]))

    # Evaluate initial candidates
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None

    for mats, thicks in candidates:
        if not time_ok():
            break
        score = eval_design(mats, thicks)
        if score < best_score:
            best_score = score
            best_materials = mats[:]
            best_thicknesses = thicks[:]

    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]
        best_score = eval_design(best_materials, best_thicknesses)

    # Phase 1: Differential evolution for fixed structures
    # Try different layer counts and material patterns
    structures_to_try = []

    # Quarter-wave alternating patterns
    for n_layers in range(1, 11):
        for start_mat in ['L', 'H']:
            materials = []
            for i in range(n_layers):
                materials.append(start_mat if (i % 2 == 0) else ('H' if start_mat == 'L' else 'L'))
            structures_to_try.append(materials)

    # Some fixed patterns with specific material sequences
    structures_to_try.append(['L', 'H', 'L'])
    structures_to_try.append(['H', 'L', 'H'])
    structures_to_try.append(['L', 'H', 'L', 'H'])
    structures_to_try.append(['H', 'L', 'H', 'L'])
    structures_to_try.append(['L', 'H', 'L', 'H', 'L'])
    structures_to_try.append(['H', 'L', 'H', 'L', 'H'])
    structures_to_try.append(['L', 'H', 'L', 'H', 'L', 'H'])

    # Remove duplicates while preserving order
    seen = set()
    unique_structures = []
    for s in structures_to_try:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_structures.append(s)

    # Run DE on each structure (limited time per structure)
    for materials in unique_structures:
        if not time_ok():
            break

        n_layers = len(materials)
        n_vals = _n_from_materials(materials, n_L, n_H)

        # Objective function for DE
        def objective(x):
            d = np.clip(x, 0, max_layer_nm)
            if np.sum(d) > max_total_nm + 1e-6:
                return 100.0  # Large penalty
            return _evaluate(n_vals, d, lambdas, n_ambient, n_substrate)

        # Bounds: each layer 0 to max_layer_nm
        bounds = [(0.0, max_layer_nm)] * n_layers

        # Initial guess: quarter-wave
        x0 = np.array([center_lam / (4.0 * n_vals[i]) for i in range(n_layers)])
        # Ensure x0 is within bounds
        x0 = np.clip(x0, 0, max_layer_nm)
        # Scale if total exceeds max
        if np.sum(x0) > max_total_nm:
            x0 = x0 * (max_total_nm / np.sum(x0))

        # Run DE with limited population and iterations
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                x0=x0,
                maxiter=30,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            if result.success or result.fun < best_score:
                d_opt = np.clip(result.x, 0, max_layer_nm)
                score = objective(d_opt)
                if score < best_score:
                    best_score = score
                    best_materials = materials[:]
                    best_thicknesses = d_opt.tolist()
        except Exception:
            pass

        # Nelder-Mead refinement on the DE result
        if time_ok():
            try:
                def nm_objective(x):
                    d = np.clip(x, 0, max_layer_nm)
                    if np.sum(d) > max_total_nm + 1e-6:
                        return 100.0
                    return _evaluate(n_vals, d, lambdas, n_ambient, n_substrate)

                result_nm = minimize(
                    nm_objective,
                    x0=np.clip(best_thicknesses, 0, max_layer_nm) if best_materials == materials else x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                d_nm = np.clip(result_nm.x, 0, max_layer_nm)
                score_nm = nm_objective(d_nm)
                if score_nm < best_score:
                    best_score = score_nm
                    best_materials = materials[:]
                    best_thicknesses = d_nm.tolist()
            except Exception:
                pass

    # Phase 2: Local search with adaptive steps on the best structure
    if time_ok():
        current_materials = best_materials[:]
        current_thicknesses = best_thicknesses[:]
        current_score = best_score

        step_sizes = [50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78, 0.39, 0.195]

        for step in step_sizes:
            if not time_ok():
                break

            improved = True
            while improved and time_ok():
                improved = False
                n_layers = len(current_materials)

                # Try perturbing each thickness
                for i in range(n_layers):
                    for delta in [-step, step]:
                        new_thicks = current_thicknesses[:]
                        new_thicks[i] = max(0.0, min(max_layer_nm, new_thicks[i] + delta))

                        if not _is_valid(new_thicks, max_total_nm, max_layer_nm):
                            continue

                        score = eval_design(current_materials, new_thicks)
                        if score < current_score - 1e-10:
                            current_thicknesses = new_thicks
                            current_score = score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_materials = current_materials[:]
                                best_thicknesses = current_thicknesses[:]

                # Try adding a layer
                if len(current_materials) < max_layers and time_ok():
                    for pos in range(len(current_materials) + 1):
                        for new_mat in ['L', 'H']:
                            new_mats = current_materials[:]
                            new_mats.insert(pos, new_mat)

                            for init_t in [30.0, 60.0, 100.0, 150.0]:
                                new_thicks = current_thicknesses[:]
                                new_thicks.insert(pos, init_t)

                                if not _is_valid(new_thicks, max_total_nm, max_layer_nm):
                                    continue

                                score = eval_design(new_mats, new_thicks)
                                if score < current_score - 1e-10:
                                    current_materials = new_mats
                                    current_thicknesses = new_thicks
                                    current_score = score
                                    improved = True
                                    if current_score < best_score:
                                        best_score = current_score
                                        best_materials = current_materials[:]
                                        best_thicknesses = current_thicknesses[:]
                                    break
                        if improved:
                            break
                    if improved:
                        break

                # Try removing a layer
                if len(current_materials) > 1 and time_ok():
                    for i in range(len(current_materials)):
                        new_mats = current_materials[:]
                        new_mats.pop(i)
                        new_thicks = current_thicknesses[:]
                        new_thicks.pop(i)

                        if not _is_valid(new_thicks, max_total_nm, max_layer_nm):
                            continue

                        score = eval_design(new_mats, new_thicks)
                        if score < current_score - 1e-10:
                            current_materials = new_mats
                            current_thicknesses = new_thicks
                            current_score = score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_materials = current_materials[:]
                                best_thicknesses = current_thicknesses[:]
                            break

        # Final Nelder-Mead polish
        if time_ok() and len(best_materials) > 0:
            try:
                n_vals = _n_from_materials(best_materials, n_L, n_H)

                def final_nm_objective(x):
                    d = np.clip(x, 0, max_layer_nm)
                    if np.sum(d) > max_total_nm + 1e-6:
                        return 100.0
                    return _evaluate(n_vals, d, lambdas, n_ambient, n_substrate)

                result_final = minimize(
                    final_nm_objective,
                    x0=np.array(best_thicknesses),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                d_final = np.clip(result_final.x, 0, max_layer_nm)
                score_final = final_nm_objective(d_final)
                if score_final < best_score:
                    best_score = score_final
                    best_thicknesses = d_final.tolist()
            except Exception:
                pass

    # Final validation and clamping
    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]

    # Clamp thicknesses
    best_thicknesses = [max(0.0, min(max_layer_nm, t)) for t in best_thicknesses]

    # Check total thickness and scale if needed
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]

    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
