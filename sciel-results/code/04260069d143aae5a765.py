"""Broadband antireflection coating design.

Combines a fully vectorized characteristic-matrix reflectance evaluator with a
time-budgeted multi-start optimization:
  1. Nelder-Mead simplex optimization over continuous thickness space for
     multiple layer counts and material patterns.
  2. Coordinate-descent refinement with shrinking steps.
  3. Structural moves (insertion, deletion, material swaps) followed by
     re-optimization.
Analytic starting points (quarter-wave stacks, index-matching single layers)
seed the search.
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
    if len(d_vals) == 0:
        return True
    if np.sum(d_vals) > max_total_nm + 1e-6:
        return False
    if np.min(d_vals) < -1e-6 or np.max(d_vals) > max_layer_nm + 1e-6:
        return False
    return True


def _n_array(materials, n_L, n_H):
    return np.array([n_L if m == 'L' else n_H for m in materials], dtype=float)


def _nelder_mead(objective, x0, bounds, max_iter=500, tol=1e-6, time_budget=None, start_time=None):
    """
    Nelder-Mead simplex optimization with bounds.
    objective: function of 1-D numpy array -> float
    x0: initial point
    bounds: (lower, upper) arrays of same length as x0
    Returns best x and best f.
    """
    n = len(x0)
    lower = np.array(bounds[0], dtype=float)
    upper = np.array(bounds[1], dtype=float)

    # Build initial simplex
    simplex = [np.array(x0, dtype=float)]
    f_vals = [objective(simplex[0])]
    for i in range(n):
        x_new = np.array(x0, dtype=float)
        step = 0.25 * (upper[i] - lower[i]) if upper[i] > lower[i] else 0.25
        x_new[i] += step
        x_new = np.clip(x_new, lower, upper)
        simplex.append(x_new)
        f_vals.append(objective(simplex[-1]))

    best_x = simplex[0].copy()
    best_f = f_vals[0]

    for it in range(max_iter):
        if time_budget is not None and start_time is not None:
            if time.time() - start_time > time_budget:
                break

        # Sort by objective value
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        if f_vals[0] < best_f:
            best_f = f_vals[0]
            best_x = simplex[0].copy()

        if f_vals[-1] - f_vals[0] < tol:
            break

        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, lower, upper)
        f_r = objective(x_r)

        if f_r < f_vals[0]:
            # Expansion
            gamma = 2.0
            x_e = centroid + gamma * (centroid - simplex[-1])
            x_e = np.clip(x_e, lower, upper)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
        elif f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
        else:
            # Contraction
            rho = 0.5
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, lower, upper)
            f_c = objective(x_c)
            if f_c < f_vals[-1]:
                simplex[-1] = x_c
                f_vals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lower, upper)
                    f_vals[i] = objective(simplex[i])

    return best_x, best_f


def _optimize_thicknesses(materials, thicknesses, lambdas, n_ambient, n_substrate,
                          n_L, n_H, max_total_nm, max_layer_nm, time_budget, start_time):
    """
    Optimize thicknesses using Nelder-Mead, then coordinate descent.
    Returns (thicknesses_list, score).
    """
    n_layers = len(materials)
    if n_layers == 0:
        return [], 0.0

    n_vals = _n_array(materials, n_L, n_H)
    lower = np.zeros(n_layers)
    upper = np.full(n_layers, max_layer_nm)

    def objective(th):
        th = np.clip(th, lower, upper)
        # Enforce total thickness constraint via penalty
        if np.sum(th) > max_total_nm:
            return 1000.0 + 10.0 * (np.sum(th) - max_total_nm)
        if np.min(th) < 0:
            return 1000.0
        return _mean_reflectance(n_vals, th, lambdas, n_ambient, n_substrate)

    x0 = np.array(thicknesses, dtype=float)
    x0 = np.clip(x0, lower, upper)
    if np.sum(x0) > max_total_nm:
        scale = max_total_nm / np.sum(x0)
        x0 = x0 * scale

    remaining = time_budget - (time.time() - start_time)
    if remaining < 1.0:
        return thicknesses, _mean_reflectance(n_vals, np.array(thicknesses), lambdas, n_ambient, n_substrate)

    x_best, f_best = _nelder_mead(objective, x0, (lower, upper),
                                  max_iter=300, tol=1e-5,
                                  time_budget=remaining * 0.6, start_time=start_time)

    # Coordinate descent refinement
    remaining = time_budget - (time.time() - start_time)
    if remaining > 1.0:
        t = x_best.copy()
        sc = f_best
        for step in [50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125]:
            if time.time() - start_time > time_budget - 0.5:
                break
            improved = True
            while improved and time.time() - start_time < time_budget - 0.5:
                improved = False
                for i in range(n_layers):
                    for delta in (-step, step):
                        nt = t.copy()
                        nt[i] = min(max_layer_nm, max(0.0, nt[i] + delta))
                        if np.sum(nt) > max_total_nm:
                            continue
                        s = _mean_reflectance(n_vals, nt, lambdas, n_ambient, n_substrate)
                        if s < sc - 1e-9:
                            t, sc = nt, s
                            improved = True
        x_best, f_best = t, sc

    return x_best.tolist(), f_best


def _structural_search(materials, thicknesses, lambdas, n_ambient, n_substrate,
                       n_L, n_H, max_layers, max_total_nm, max_layer_nm,
                       time_budget, start_time):
    """
    Try structural moves (insert, delete, swap) and re-optimize.
    Returns (best_materials, best_thicknesses, best_score).
    """
    mats = list(materials)
    th = list(thicknesses)
    n_vals = _n_array(mats, n_L, n_H)
    score = _mean_reflectance(n_vals, np.array(th), lambdas, n_ambient, n_substrate)
    best_mats, best_th, best_score = mats[:], th[:], score

    for _ in range(50):
        if time.time() - start_time > time_budget - 2.0:
            break

        n_layers = len(mats)
        moves = []

        # Insert a new layer
        if n_layers < max_layers:
            for pos in range(n_layers + 1):
                for mat in ('L', 'H'):
                    nm = mats[:pos] + [mat] + mats[pos:]
                    nt = th[:pos] + [150.0] + th[pos:]
                    if np.sum(nt) <= max_total_nm:
                        moves.append((nm, nt))

        # Delete a layer
        if n_layers > 1:
            for pos in range(n_layers):
                nm = mats[:pos] + mats[pos + 1:]
                nt = th[:pos] + th[pos + 1:]
                moves.append((nm, nt))

        # Swap material
        for pos in range(n_layers):
            nm = mats[:]
            nm[pos] = 'H' if mats[pos] == 'L' else 'L'
            nt = th[:]
            moves.append((nm, nt))

        best_move_score = best_score
        best_move = None

        for nm, nt in moves:
            if time.time() - start_time > time_budget - 2.0:
                break
            nv = _n_array(nm, n_L, n_H)
            s = _mean_reflectance(nv, np.array(nt), lambdas, n_ambient, n_substrate)
            if s < best_move_score - 1e-9:
                best_move_score = s
                best_move = (nm, nt)

        if best_move is None:
            break

        mats, th = best_move
        # Re-optimize thicknesses
        remaining = time_budget - (time.time() - start_time)
        if remaining > 2.0:
            th, score = _optimize_thicknesses(mats, th, lambdas, n_ambient, n_substrate,
                                              n_L, n_H, max_total_nm, max_layer_nm,
                                              remaining, start_time)
            if score < best_score:
                best_score = score
                best_mats, best_th = mats[:], th[:]

    return best_mats, best_th, best_score


def design(problem):
    """Design a broadband antireflection coating stack (air -> substrate order)."""
    start = time.time()
    deadline = start + 140.0  # safety margin under the 150 s wall-clock kill

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

    # --- Build analytic starting points -------------------------------------
    starts = []
    # Single-layer index-matching
    for mat, n in (('L', n_L), ('H', n_H)):
        starts.append(([mat], [center_lam / (4.0 * n)]))
    # Quarter-wave stacks of various lengths
    for n_layers in range(2, 13):
        for start_mat in ('L', 'H'):
            materials, thicknesses = [], []
            for i in range(n_layers):
                mat = start_mat if i % 2 == 0 else ('H' if start_mat == 'L' else 'L')
                n = n_L if mat == 'L' else n_H
                materials.append(mat)
                thicknesses.append(center_lam / (4.0 * n))
            starts.append((materials, thicknesses))

    # --- Evaluate starting points and pick the best -------------------------
    best_materials, best_thicknesses, best_score = None, None, float('inf')
    for mats, thicks in starts:
        if not time_ok():
            break
        if len(mats) > max_layers:
            continue
        nv = _n_array(mats, n_L, n_H)
        th = np.array(thicks, dtype=float)
        if not _is_valid(th, max_total_nm, max_layer_nm):
            continue
        s = _mean_reflectance(nv, th, lambdas, n_ambient, n_substrate)
        if s < best_score:
            best_score = s
            best_materials, best_thicknesses = mats[:], thicks[:]

    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]
        best_score = _mean_reflectance(_n_array(best_materials, n_L, n_H),
                                       np.array(best_thicknesses, dtype=float),
                                       lambdas, n_ambient, n_substrate)

    # --- Nelder-Mead + coordinate descent optimization ----------------------
    remaining = max(0.0, (deadline - 5.0) - time.time())
    if remaining > 5.0:
        thicks, score = _optimize_thicknesses(
            best_materials, best_thicknesses, lambdas, n_ambient, n_substrate,
            n_L, n_H, max_total_nm, max_layer_nm, remaining, start)
        if np.isfinite(score) and score < best_score:
            best_thicknesses, best_score = thicks, score

    # --- Structural search --------------------------------------------------
    remaining = max(0.0, (deadline - 3.0) - time.time())
    if remaining > 5.0:
        mats, thicks, score = _structural_search(
            best_materials, best_thicknesses, lambdas, n_ambient, n_substrate,
            n_L, n_H, max_layers, max_total_nm, max_layer_nm, remaining, start)
        if np.isfinite(score) and score < best_score:
            best_materials, best_thicknesses, best_score = mats, thicks, score

    # --- Safety fallback: guarantee a valid, in-bounds result ---------------
    if best_materials is None or not _is_valid(np.array(best_thicknesses, dtype=float),
                                               max_total_nm, max_layer_nm):
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]

    # Clamp any stray values defensively
    best_thicknesses = [min(max_layer_nm, max(0.0, float(t))) for t in best_thicknesses]
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
