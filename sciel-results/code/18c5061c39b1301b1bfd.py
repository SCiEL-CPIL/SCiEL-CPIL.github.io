"""Fresh design: quarter-wave-complement alternating L/H stack.
Constraint: for each pair, n_L*d_L + n_H*d_H = lambda0/4.
Search space: H thicknesses (one per pair) + lambda0.
"""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE_MARGIN = 12.0  # seconds to spare before 150s wall-clock kill


def _reflectance_stack(materials, thicknesses, lambdas, n0, ns):
    """Exact characteristic-matrix method, normal incidence.
    Returns reflectance in percent (scalar per wavelength)."""
    nL = materials["L"]
    nH = materials["H"]
    n = np.array([nL if m == "L" else nH for m in materials], dtype=float)
    d = np.array(thicknesses, dtype=float)
    lam = np.asarray(lambdas, dtype=float)
    k = 2.0 * np.pi / lam  # shape (M,)
    # Build 2x2 matrices per wavelength
    M = np.empty((lam.size, 2, 2), dtype=complex)
    for i in range(lam.size):
        Mi = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for j in range(len(n)):
            phi = k[i] * n[j] * d[j]
            cj = np.cos(phi)
            sj = np.sin(phi)
            # Layer matrix at normal incidence: [[cos, i sin/n],[i n sin, cos]]
            Lm = np.array([[cj, 1j * sj / n[j]], [1j * n[j] * sj, cj]], dtype=complex)
            Mi = Mi @ Lm
        M[i] = Mi
    # Substrate: transfer through substrate to infinity -> boundary condition
    # r = (M11 + M12 * ns - M21 - M22 * n0) / (M11 + M12 * ns + M21 + M22 * n0)
    num = M[:, 0, 0] + M[:, 0, 1] * ns - M[:, 1, 0] - M[:, 1, 1] * n0
    den = M[:, 0, 0] + M[:, 0, 1] * ns + M[:, 1, 0] + M[:, 1, 1] * n0
    r = num / den
    R = np.abs(r) ** 2
    return R * 100.0


def _build_stack(pairs, h_thick, lam0, nL, nH, max_layer_nm, max_total_nm):
    """Build alternating L/H material list and thicknesses from pair count and H thicknesses.
    Pair j: L layer then H layer, with nL*dL + nH*dH = lam0/4.
    Order air -> ... -> substrate: we place pairs from air side.
    """
    materials = []
    thicknesses = []
    total = 0.0
    for j in range(pairs):
        dH = h_thick[j]
        dL = (lam0 / 4.0 - nH * dH) / nL
        # clamp to valid range
        dH = min(max(dH, 0.0), max_layer_nm)
        dL = min(max(dL, 0.0), max_layer_nm)
        materials.append("L")
        materials.append("H")
        thicknesses.append(dL)
        thicknesses.append(dH)
        total += dL + dH
    if total > max_total_nm + 1e-6:
        return None
    return materials, thicknesses


def _objective(x, problem, lam_grid):
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layers = problem["max_layers"]
    pairs = int(x[0])
    if pairs < 1:
        pairs = 1
    if pairs * 2 > max_layers:
        pairs = max_layers // 2
    lam0 = x[1]
    h_thick = x[2:2 + pairs]
    built = _build_stack(pairs, h_thick, lam0, nL, nH, max_layer_nm, max_total_nm)
    if built is None:
        return 1e6
    materials, thicknesses = built
    R = _reflectance_stack(materials, thicknesses, lam_grid, n0, ns)
    return float(np.mean(R))


def design(problem):
    start = time.time()
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]

    # Evaluation grid: 401 equally spaced wavelengths
    lam_grid = np.linspace(band[0], band[1], 401)

    def elapsed():
        return time.time() - start

    def remaining():
        return 150.0 - elapsed() - _DEADLINE_MARGIN

    max_pairs = max_layers // 2  # each pair = 2 layers

    # Parameterize: x = [pairs, lam0, h_1, ..., h_pairs]
    # We'll search over a fixed number of pairs P (try several), optimizing
    # lam0 and the P H-thicknesses.
    best_val = 1e9
    best_result = None
    # Fallback: single quarter-wave L layer
    centre = 0.5 * (band[0] + band[1])
    fb_mat, fb_th = ["L"], [centre / (4.0 * nL)]
    fb_R = _reflectance_stack(fb_mat, fb_th, lam_grid, n0, ns)
    best_val = float(np.mean(fb_R))
    best_result = (fb_mat, fb_th)

    # Try a range of pair counts
    for P in range(1, max_pairs + 1):
        if remaining() < 8.0:
            break
        # Initial guess: lam0 = centre, each H thickness = quarter-wave H
        lam0_init = centre
        h_init = np.array([lam0_init / (4.0 * nH)] * P)
        # But ensure dL = (lam0/4 - nH*dH)/nL >= 0 and <= max_layer
        # With dH = lam0/(4 nH), dL = 0. So start with a split.
        # Let optical split: fraction f of optical thickness to H.
        f = 0.5
        h_init = np.array([(f * lam0_init) / nH] * P)
        x0 = np.concatenate([[P], [lam0_init], h_init])

        # Bounds: pairs is discrete (fixed here), lam0 in [300, 1000], h in [0, max_layer]
        bounds = [(300.0, 1000.0)] + [(0.0, max_layer_nm)] * P
        # x[0] is pairs (fixed), so we don't optimize it; pass through.
        # We'll optimize x[1:] with bounds.
        def obj_sub(z):
            x = np.concatenate([[P], z])
            return _objective(x, problem, lam_grid)

        z0 = x0[1:]
        try:
            res = minimize(obj_sub, z0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 200, "maxfun": 2000, "ftol": 1e-10, "gtol": 1e-8})
            if res.success or res.fun < best_val:
                val = float(res.fun)
                x = np.concatenate([[P], res.x])
                built = _build_stack(P, res.x[1:], res.x[0], nL, nH, max_layer_nm, max_total_nm)
                # careful: x[0] is pairs, x[1] is lam0, x[2:] is h
                built = _build_stack(P, res.x[2:2+P], res.x[1], nL, nH, max_layer_nm, max_total_nm)
                if built is not None:
                    materials, thicknesses = built
                    R = _reflectance_stack(materials, thicknesses, lam_grid, n0, ns)
                    mR = float(np.mean(R))
                    if mR < best_val:
                        best_val = mR
                        best_result = (materials, thicknesses)
        except Exception:
            pass

        # Also try a few random restarts for this P if time
        if remaining() < 8.0:
            break
        rng = np.random.default_rng(12345 + P)
        for _ in range(3):
            if remaining() < 5.0:
                break
            lam0_r = rng.uniform(450, 750)
            h_r = rng.uniform(0, max_layer_nm, size=P)
            # ensure feasible dL
            z0r = np.concatenate([[lam0_r], h_r])
            try:
                res = minimize(obj_sub, z0r, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 150, "maxfun": 1500, "ftol": 1e-10, "gtol": 1e-8})
                if res.fun < best_val:
                    x = np.concatenate([[P], res.x])
                    built = _build_stack(P, res.x[2:2+P], res.x[1], nL, nH, max_layer_nm, max_total_nm)
                    if built is not None:
                        materials, thicknesses = built
                        R = _reflectance_stack(materials, thicknesses, lam_grid, n0, ns)
                        mR = float(np.mean(R))
                        if mR < best_val:
                            best_val = mR
                            best_result = (materials, thicknesses)
            except Exception:
                pass

    # Final safety: validate best_result
    if best_result is None:
        return fb_mat, fb_th
    materials, thicknesses = best_result
    # Validate constraints
    if len(materials) > max_layers:
        return fb_mat, fb_th
    if any(t < -1e-9 or t > max_layer_nm + 1e-9 for t in thicknesses):
        return fb_mat, fb_th
    if sum(thicknesses) > max_total_nm + 1e-6:
        return fb_mat, fb_th
    # clamp tiny negatives
    thicknesses = [max(0.0, t) for t in thicknesses]
    return materials, thicknesses
