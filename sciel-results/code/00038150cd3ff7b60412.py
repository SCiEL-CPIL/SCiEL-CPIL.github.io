"""Antireflection coating design via vectorized thin-film matrices + global/local optimization."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START


def _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm):
    """Mean reflectance (percent) over 401 wavelengths, vectorized characteristic-matrix."""
    if len(labels) == 0:
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return r0 * r0 * 100.0

    n_L = materials['L']
    n_H = materials['H']
    n_layers = len(labels)
    d = np.asarray(thicknesses, dtype=float)

    # Per-layer optical quantities (vector over wavelengths)
    n_arr = np.empty(n_layers)
    for i, lab in enumerate(labels):
        n_arr[i] = n_L if lab == 'L' else n_H

    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)  # (401,)
    nd = n_arr[:, None] * d[:, None]                          # (n_layers, 401)
    delta = 2.0 * np.pi * nd / wavelengths[None, :]           # (n_layers, 401)
    c = np.cos(delta)
    s = np.sin(delta)
    inv_n = 1.0 / n_arr[:, None]
    A = c
    B = 1j * s * inv_n
    C = 1j * n_arr[:, None] * s
    D = c

    # Product of layer matrices, vectorized over wavelengths
    m11 = np.ones(401, dtype=complex)
    m12 = np.zeros(401, dtype=complex)
    m21 = np.zeros(401, dtype=complex)
    m22 = np.ones(401, dtype=complex)
    for j in range(n_layers):
        a = A[j]; b = B[j]; cc = C[j]; dd = D[j]
        n11 = m11 * a + m12 * cc
        n12 = m11 * b + m12 * dd
        n21 = m21 * a + m22 * cc
        n22 = m21 * b + m22 * dd
        m11, m12, m21, m22 = n11, n12, n21, n22

    ns = substrate_index
    Y = (m21 + m22 * ns) / (m11 + m12 * ns)
    r = (ambient_index - Y) / (ambient_index + Y)
    return float(np.mean(np.abs(r) ** 2) * 100.0)


def _safe_scale(labels, thicknesses, max_layer_nm, max_total_nm):
    t = np.asarray(thicknesses, dtype=float)
    t = np.clip(t, 0.0, max_layer_nm)
    tot = t.sum()
    if tot > max_total_nm and tot > 0:
        t = t * (max_total_nm / tot)
    return t.tolist()


def _fallback(problem):
    materials = problem["materials"]
    n_L = materials['L']
    n_H = materials['H']
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    d_L = centre / (4.0 * n_L)
    d_H = centre / (4.0 * n_H)
    candidates = [
        (["L"], [d_L]),
        (["L", "H"], [d_L, d_H]),
        (["H", "L", "H"], [d_H, d_L, d_H]),
        (["H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H]),
        (["H", "L", "H", "L", "H", "L"], [d_H, d_L, d_H, d_L, d_H, d_L]),
        (["H", "L", "H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H, d_L, d_H]),
    ]
    best = None
    best_score = float('inf')
    for labels, th in candidates:
        th = _safe_scale(labels, th, problem["max_layer_nm"], problem["max_total_nm"])
        if sum(th) > problem["max_total_nm"] + 1e-6:
            continue
        sc = _compute_reflectance(labels, th, materials, problem["ambient_index"],
                                  problem["substrate_index"], problem["band_nm"])
        if sc < best_score:
            best_score = sc
            best = (labels[:], th)
    return best


def design(problem):
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s wall-clock kill

    materials = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    n_L = materials['L']
    n_H = materials['H']
    centre = 0.5 * (band_nm[0] + band_nm[1])
    d_L = centre / (4.0 * n_L)
    d_H = centre / (4.0 * n_H)

    def remaining():
        return deadline - time.time()

    # ---- Safe fallback first (guarantees a valid result) ----
    best = _fallback(problem)
    best_labels, best_th = best
    best_score = _compute_reflectance(best_labels, best_th, materials, ambient_index,
                                      substrate_index, band_nm)

    # ---- Global search over a fixed number of layers, last layer forced to 'L' ----
    def _objective(x, labels):
        t = np.asarray(x, dtype=float)
        t = np.clip(t, 0.0, max_layer_nm)
        tot = t.sum()
        if tot > max_total_nm:
            t = t * (max_total_nm / tot)
        return _compute_reflectance(labels, t.tolist(), materials, ambient_index,
                                    substrate_index, band_nm)

    def _polish(labels, x0):
        def obj(x):
            t = np.asarray(x, dtype=float)
            t = np.clip(t, 0.0, max_layer_nm)
            tot = t.sum()
            if tot > max_total_nm:
                t = t * (max_total_nm / tot)
            return _compute_reflectance(labels, t.tolist(), materials, ambient_index,
                                        substrate_index, band_nm)
        try:
            res = minimize(obj, np.asarray(x0, dtype=float), method='L-BFGS-B',
                           bounds=[(0.0, max_layer_nm)] * len(labels),
                           options={'maxiter': 300, 'maxfun': 4000, 'ftol': 1e-10})
            if res.success or res.fun < obj(np.asarray(x0, dtype=float)):
                return res.x, res.fun
        except Exception:
            pass
        return np.asarray(x0, dtype=float), obj(np.asarray(x0, dtype=float))

    # Layer counts to explore (1..min(9, max_layers)); last layer forced to 'L'
    max_n = min(9, max_layers)
    # Prefer alternating patterns starting with 'H' (best broadband behavior), ending 'L'
    patterns = []
    for n in range(1, max_n + 1):
        # alternating starting with H
        lab = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        lab[-1] = 'L'
        patterns.append(lab)
        if n >= 2:
            # alternating starting with L
            lab2 = ['L' if i % 2 == 0 else 'H' for i in range(n)]
            lab2[-1] = 'L'
            if lab2 != lab:
                patterns.append(lab2)

    # Order patterns by expected quality (more layers, H-start first)
    def _pat_key(p):
        return (-(len(p)), 0 if p[0] == 'H' else 1)
    patterns.sort(key=_pat_key)

    for labels in patterns:
        if remaining() < 8.0:
            break
        n = len(labels)
        x0 = []
        for i, lab in enumerate(labels):
            x0.append(d_H if lab == 'H' else d_L)
        x0 = _safe_scale(labels, x0, max_layer_nm, max_total_nm)

        # Local polish from the quarter-wave starting point
        xp, fp = _polish(labels, x0)
        if fp < best_score:
            best_score = fp
            best_labels = labels[:]
            best_th = _safe_scale(labels, xp.tolist(), max_layer_nm, max_total_nm)
        if remaining() < 12.0:
            break

        # Global differential evolution for this pattern (skip very small stacks)
        if n < 3:
            continue
        if remaining() < 20.0:
            break
        try:
            res = differential_evolution(
                lambda x: _objective(x, labels),
                bounds=[(0.0, max_layer_nm)] * n,
                seed=7, maxiter=60, popsize=12, tol=1e-6,
                mutation=(0.4, 1.0), recombination=0.8,
                polish=False, init='latinhypercube',
            )
            if remaining() < 5.0:
                break
            if res.x is not None:
                xp, fp = _polish(labels, res.x)
                if fp < best_score:
                    best_score = fp
                    best_labels = labels[:]
                    best_th = _safe_scale(labels, xp.tolist(), max_layer_nm, max_total_nm)
        except Exception:
            pass

    # ---- Final validation and return ----
    best_th = _safe_scale(best_labels, best_th, max_layer_nm, max_total_nm)
    if sum(best_th) > max_total_nm + 1e-6:
        scale = max_total_nm / sum(best_th)
        best_th = [t * scale for t in best_th]
    best_th = [max(0.0, min(max_layer_nm, float(t))) for t in best_th]

    # Recompute to make sure it's valid and better than bare glass
    final_score = _compute_reflectance(best_labels, best_th, materials, ambient_index,
                                       substrate_index, band_nm)
    if final_score > 4.26 + 1e-6:
        # fall back to the guaranteed good quarter-wave design
        fb = _fallback(problem)
        best_labels, best_th = fb
        best_th = _safe_scale(best_labels, best_th, max_layer_nm, max_total_nm)

    return best_labels, best_th


# EVOLVE-BLOCK-END
