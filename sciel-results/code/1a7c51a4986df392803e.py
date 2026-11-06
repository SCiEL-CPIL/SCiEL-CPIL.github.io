"""Broadband antireflection coating design.

Uses a differentiable surrogate based on the characteristic-matrix method
evaluated at the band center, combined with multi-start Nelder-Mead and
coordinate-descent polishing on the exact 401-point spectrum.
"""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _reflectance_spectrum(n_vals, d_vals, lambdas, n_ambient, n_substrate):
    """Vectorized reflectance (0..1) at each wavelength via char-matrix method."""
    W = len(lambdas)
    M11 = np.ones(W, dtype=complex)
    M12 = np.zeros(W, dtype=complex)
    M21 = np.zeros(W, dtype=complex)
    M22 = np.ones(W, dtype=complex)

    for j in range(len(n_vals)):
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
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22

    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    r = (n_ambient - Y) / (n_ambient + Y)
    return np.abs(r) ** 2


def _mean_reflectance(n_vals, d_vals, lambdas, n_ambient, n_substrate):
    return 100.0 * np.mean(_reflectance_spectrum(n_vals, d_vals, lambdas,
                                                 n_ambient, n_substrate))


def _is_valid(d_vals, max_total_nm, max_layer_nm):
    if d_vals.size == 0:
        return True
    if d_vals.sum() > max_total_nm + 1e-6:
        return False
    if d_vals.min() < -1e-6 or d_vals.max() > max_layer_nm + 1e-6:
        return False
    return True


def _n_array(materials, n_L, n_H):
    return np.array([n_L if m == 'L' else n_H for m in materials], dtype=float)


def _nelder_mead(f, x0, bounds, max_iter=200, tol=1e-8, time_budget=None, deadline=None):
    """Simple Nelder-Mead with bounds and time check."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    f0 = f(simplex[0])
    for i in range(n):
        xi = simplex[0].copy()
        step = 0.05 * (bounds[1][i] - bounds[0][i]) if bounds[1][i] > bounds[0][i] else 0.05
        xi[i] += step
        xi = np.clip(xi, bounds[0], bounds[1])
        simplex.append(xi)
        simplex.append(simplex[0].copy())  # placeholder, will be filled

    # Build proper initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = x0.copy()
        step = 0.05 * (bounds[1][i] - bounds[0][i]) if bounds[1][i] > bounds[0][i] else 0.05
        xi[i] += step
        simplex.append(np.clip(xi, bounds[0], bounds[1]))

    fs = [f(simplex[i]) for i in range(n + 1)]

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Sort
        order = np.argsort(fs)
        simplex = [simplex[i] for i in order]
        fs = [fs[i] for i in order]

        if abs(fs[0] - fs[-1]) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        xr = np.clip(xr, bounds[0], bounds[1])
        fr = f(xr)

        if fr < fs[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            xe = np.clip(xe, bounds[0], bounds[1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fs[-1] = fe
            else:
                simplex[-1] = xr
                fs[-1] = fr
        elif fr < fs[-2]:
            simplex[-1] = xr
            fs[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            xc = np.clip(xc, bounds[0], bounds[1])
            fc = f(xc)
            if fc < fs[-1]:
                simplex[-1] = xc
                fs[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = 0.5 * (simplex[0] + simplex[i])
                    fs[i] = f(simplex[i])

    best_idx = np.argmin(fs)
    return simplex[best_idx], fs[best_idx]


def _coordinate_descent(f, x0, bounds, time_budget=5.0, deadline=None):
    """Coordinate descent with adaptive step sizes."""
    x = np.array(x0, dtype=float)
    fx = f(x)
    start = time.time()

    for step in [100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125, 0.390625]:
        if deadline is not None and time.time() > deadline:
            break
        if time.time() - start > time_budget:
            break
        improved = True
        while improved and (deadline is None or time.time() < deadline):
            improved = False
            for i in range(len(x)):
                for delta in (-step, step):
                    nx = x.copy()
                    nx[i] = np.clip(nx[i] + delta, bounds[0][i], bounds[1][i])
                    nf = f(nx)
                    if nf < fx - 1e-10:
                        x, fx = nx, nf
                        improved = True
    return x, fx


def _surrogate_reflectance(n_vals, d_vals, lam_center, n_ambient, n_substrate):
    """Fast surrogate: reflectance at center wavelength only."""
    return _mean_reflectance(n_vals, d_vals, np.array([lam_center]), n_ambient, n_substrate)


def _generate_initial_solutions(n_L, n_H, n_ambient, n_substrate, center_lam,
                                 max_layers, max_total_nm, max_layer_nm, n_starts=10):
    """Generate diverse initial solutions."""
    starts = []

    # Single-layer index matching
    for mat, n in (('L', n_L), ('H', n_H)):
        d = center_lam / (4.0 * n)
        if d <= max_layer_nm:
            starts.append(([mat], [d]))

    # Quarter-wave stacks
    for n_layers in range(1, min(7, max_layers + 1)):
        for start_mat in ('L', 'H'):
            materials, thicknesses = [], []
            total = 0.0
            for i in range(n_layers):
                mat = start_mat if i % 2 == 0 else ('H' if start_mat == 'L' else 'L')
                n = n_L if mat == 'L' else n_H
                d = center_lam / (4.0 * n)
                if total + d > max_total_nm or d > max_layer_nm:
                    break
                materials.append(mat)
                thicknesses.append(d)
                total += d
            if len(materials) == n_layers and materials:
                starts.append((materials, thicknesses))

    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(n_starts):
        n_layers = rng.integers(1, min(6, max_layers + 1))
        materials = []
        thicknesses = []
        total = 0.0
        for i in range(n_layers):
            mat = 'L' if rng.random() < 0.5 else 'H'
            n = n_L if mat == 'L' else n_H
            d = rng.uniform(20.0, min(max_layer_nm, (max_total_nm - total) / (n_layers - i)))
            if d < 10.0 or total + d > max_total_nm:
                continue
            materials.append(mat)
            thicknesses.append(d)
            total += d
        if materials:
            starts.append((materials, thicknesses))

    return starts


def design(problem):
    """Design a broadband antireflection coating stack (air -> substrate order)."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin under 150s kill

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
        return time.time() < deadline - 5.0

    # --- Generate and evaluate initial solutions ---
    starts = _generate_initial_solutions(n_L, n_H, n_ambient, n_substrate,
                                         center_lam, max_layers, max_total_nm,
                                         max_layer_nm, n_starts=15)

    best_materials, best_thicknesses, best_score = None, None, float('inf')
    best_start_idx = -1

    for idx, (mats, thicks) in enumerate(starts):
        if not time_ok():
            break
        th = np.array(thicks, dtype=float)
        if not _is_valid(th, max_total_nm, max_layer_nm):
            continue
        nv = _n_array(mats, n_L, n_H)
        s = _mean_reflectance(nv, th, lambdas, n_ambient, n_substrate)
        if s < best_score:
            best_score = s
            best_materials, best_thicknesses = mats[:], thicks[:]
            best_start_idx = idx

    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]
        best_score = _mean_reflectance(_n_array(best_materials, n_L, n_H),
                                       np.array(best_thicknesses, dtype=float),
                                       lambdas, n_ambient, n_substrate)

    # --- Multi-start Nelder-Mead + coordinate descent ---
    # Try optimizing several of the best starting points
    # First, rank all valid starts by their exact score
    ranked_starts = []
    for idx, (mats, thicks) in enumerate(starts):
        th = np.array(thicks, dtype=float)
        if not _is_valid(th, max_total_nm, max_layer_nm):
            continue
        nv = _n_array(mats, n_L, n_H)
        s = _mean_reflectance(nv, th, lambdas, n_ambient, n_substrate)
        ranked_starts.append((s, idx, mats, thicks))

    ranked_starts.sort(key=lambda x: x[0])

    # Optimize top candidates
    n_optimize = min(5, len(ranked_starts))
    for rank in range(n_optimize):
        if not time_ok():
            break

        s_init, idx, mats, thicks = ranked_starts[rank]
        th = np.array(thicks, dtype=float)
        nv = _n_array(mats, n_L, n_H)
        n_layers = len(mats)

        # Define objective for Nelder-Mead
        def objective(x):
            x = np.clip(x, 0.0, max_layer_nm)
            if x.sum() > max_total_nm:
                return 100.0  # Penalty
            return _mean_reflectance(nv, x, lambdas, n_ambient, n_substrate)

        bounds = (np.zeros(n_layers), np.full(n_layers, max_layer_nm))

        # Nelder-Mead
        try:
            x_nm, f_nm = _nelder_mead(objective, th, bounds, max_iter=150,
                                       tol=1e-7, deadline=deadline - 5.0)
            x_nm = np.clip(x_nm, 0.0, max_layer_nm)
            if x_nm.sum() > max_total_nm:
                # Scale down
                scale = max_total_nm / x_nm.sum()
                x_nm *= scale
            f_nm = objective(x_nm)
        except Exception:
            x_nm, f_nm = th, s_init

        # Coordinate descent polish
        try:
            x_cd, f_cd = _coordinate_descent(objective, x_nm, bounds,
                                             time_budget=3.0, deadline=deadline - 5.0)
            x_cd = np.clip(x_cd, 0.0, max_layer_nm)
            if x_cd.sum() > max_total_nm:
                scale = max_total_nm / x_cd.sum()
                x_cd *= scale
            f_cd = objective(x_cd)
        except Exception:
            x_cd, f_cd = x_nm, f_nm

        if f_cd < best_score:
            best_score = f_cd
            best_materials = mats[:]
            best_thicknesses = x_cd.tolist()

    # --- Final structural search: try inserting/removing layers around best ---
    if time_ok():
        current_mats = list(best_materials)
        current_th = np.array(best_thicknesses, dtype=float)
        current_score = best_score

        for _ in range(50):
            if not time_ok():
                break

            n_layers = len(current_mats)
            best_move = None
            best_move_score = current_score

            # Try inserting a layer
            if n_layers < max_layers:
                for pos in range(n_layers + 1):
                    for mat in ('L', 'H'):
                        nm = current_mats[:pos] + [mat] + current_mats[pos:]
                        nt = np.concatenate([current_th[:pos], [100.0], current_th[pos:]])
                        if not _is_valid(nt, max_total_nm, max_layer_nm):
                            continue
                        nv = _n_array(nm, n_L, n_H)
                        s = _mean_reflectance(nv, nt, lambdas, n_ambient, n_substrate)
                        if s < best_move_score - 1e-6:
                            best_move = (nm, nt, s)
                            best_move_score = s

            # Try deleting a layer
            if n_layers > 1:
                for pos in range(n_layers):
                    nm = current_mats[:pos] + current_mats[pos + 1:]
                    nt = np.delete(current_th, pos)
                    if not _is_valid(nt, max_total_nm, max_layer_nm):
                        continue
                    nv = _n_array(nm, n_L, n_H)
                    s = _mean_reflectance(nv, nt, lambdas, n_ambient, n_substrate)
                    if s < best_move_score - 1e-6:
                        best_move = (nm, nt, s)
                        best_move_score = s

            # Try swapping material
            for pos in range(n_layers):
                nm = current_mats[:]
                nm[pos] = 'H' if nm[pos] == 'L' else 'L'
                nv = _n_array(nm, n_L, n_H)
                s = _mean_reflectance(nv, current_th, lambdas, n_ambient, n_substrate)
                if s < best_move_score - 1e-6:
                    best_move = (nm, current_th.copy(), s)
                    best_move_score = s

            if best_move is None:
                break

            current_mats, current_th, current_score = best_move
            if current_score < best_score:
                best_score = current_score
                best_materials, best_thicknesses = current_mats[:], current_th.tolist()

            # Quick polish
            if time_ok():
                nv = _n_array(current_mats, n_L, n_H)
                n_layers = len(current_mats)

                def objective2(x):
                    x = np.clip(x, 0.0, max_layer_nm)
                    if x.sum() > max_total_nm:
                        return 100.0
                    return _mean_reflectance(nv, x, lambdas, n_ambient, n_substrate)

                bounds = (np.zeros(n_layers), np.full(n_layers, max_layer_nm))
                try:
                    x_cd, f_cd = _coordinate_descent(objective2, current_th, bounds,
                                                     time_budget=2.0, deadline=deadline - 5.0)
                    x_cd = np.clip(x_cd, 0.0, max_layer_nm)
                    if x_cd.sum() > max_total_nm:
                        scale = max_total_nm / x_cd.sum()
                        x_cd *= scale
                    f_cd = objective2(x_cd)
                    if f_cd < current_score:
                        current_th = x_cd
                        current_score = f_cd
                        if f_cd < best_score:
                            best_score = f_cd
                            best_materials, best_thicknesses = current_mats[:], current_th.tolist()
                except Exception:
                    pass

    # --- Safety fallback ---
    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]

    # Clamp
    best_thicknesses = [min(max_layer_nm, max(0.0, float(t))) for t in best_thicknesses]

    # Final validation
    th = np.array(best_thicknesses, dtype=float)
    if not _is_valid(th, max_total_nm, max_layer_nm):
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]

    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
