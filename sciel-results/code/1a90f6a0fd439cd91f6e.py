"""Broadband antireflection coating design: gradient stack + L-BFGS-B thickness optimization."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _reflectance(mat, thick, n0, ns, band_nm):
    """Exact normal-incidence reflectance (%) via the characteristic-matrix method.
    mat: array of refractive indices (layer 0 = air side, last = substrate).
    thick: array of physical thicknesses in nm (len = len(mat)-1)."""
    w = np.linspace(band_nm[0], band_nm[1], 401)
    k = 2.0 * np.pi / w
    n = mat.astype(float)
    t = thick.astype(float)
    d = k * n[1:] * t
    # characteristic matrix of the whole stack
    M = np.array([[np.cos(d[0]), 1j / n[1] * np.sin(d[0])],
                  [1j * n[1] * np.sin(d[0]), np.cos(d[0])]], dtype=complex)
    for i in range(1, len(t)):
        Mi = np.array([[np.cos(d[i]), 1j / n[i + 1] * np.sin(d[i])],
                       [1j * n[i + 1] * np.sin(d[i]), np.cos(d[i])]], dtype=complex)
        M = M @ Mi
    B0, C0 = M[0, 0] + (1j / ns) * M[0, 1], M[1, 0] + (1j / ns) * M[1, 1]
    Y = n[0] * B0 / C0
    R = (Y - n[0]) / (Y + n[0])
    return np.mean(np.abs(R) ** 2) * 100.0


def _obj(x, n0, ns, nL, nH, nlayer, band_nm):
    """Objective: mean reflectance (%) for a fixed H/L pattern with thicknesses x."""
    idx = [n0]
    th = list(x)
    for i in range(nlayer):
        idx.append(nH if i % 2 else nL)
    idx.append(ns)
    return _reflectance(np.array(idx), np.array(th), n0, ns, band_nm)


def _pattern(nlayer):
    """Layer labels (H/L) for a gradient starting low-index at the air side."""
    return ["H" if i % 2 else "L" for i in range(nlayer)]


def design(problem):
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    deadline = time.time() + 130.0   # explicit wall-clock guard (limit 150s)
    best = None
    best_val = float("inf")

    def record(mat, th):
        nonlocal best, best_val
        s = float(sum(th))
        if len(mat) > max_layers or s > max_total + 1e-6:
            return
        th = [min(max(float(t), 0.0), max_layer) for t in th]
        if sum(th) > max_total + 1e-6:
            th = [t * (max_total / sum(th)) for t in th]
        val = _reflectance(np.array(mat, dtype=float), np.array(th, dtype=float),
                           n0, ns, band)
        if val < best_val:
            best_val = val
            best = (mat, th)

    # ---- 1) Analytical gradient starting points (fast, no optimizer) ----
    for nl in range(1, max_layers + 1):
        lam = 0.5 * (band[0] + band[1])
        th = [lam / (4.0 * (nH if i % 2 else nL)) for i in range(nl)]
        if sum(th) > max_total:
            th = [t * (max_total / sum(th)) for t in th]
        mat = [n0] + [(nH if i % 2 else nL) for i in range(nl)] + [ns]
        record(mat, th)
        if time.time() > deadline:
            break

    # ---- 2) L-BFGS-B thickness refinement on the best patterns ----
    # candidate patterns: best gradient patterns by layer count (fewer = faster)
    candidates = []
    for nl in [4, 6, 8, 10, 12]:
        if nl > max_layers:
            continue
        lam = 0.5 * (band[0] + band[1])
        x0 = np.array([lam / (4.0 * (nH if i % 2 else nL)) for i in range(nl)], dtype=float)
        if x0.sum() > max_total:
            x0 = x0 * (max_total / x0.sum())
        candidates.append((nl, x0))

    for nl, x0 in candidates:
        if time.time() > deadline:
            break
        bounds = [(0.0, max_layer)] * nl
        try:
            res = minimize(_obj, x0, args=(n0, ns, nL, nH, nl, band),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-8})
        except Exception:
            continue
        xopt = np.clip(res.x, 0.0, max_layer)
        if xopt.sum() > max_total:
            xopt = xopt * (max_total / xopt.sum())
        mat = [n0] + [(nH if i % 2 else nL) for i in range(nl)] + [ns]
        record(mat, list(xopt))

    # ---- 3) Fallbacks ----
    if best is None:
        lam = 0.5 * (band[0] + band[1])
        return ["L"], [lam / (4.0 * nL)]

    mat, th = best
    labels = [ "H" if m == nH else "L" for m in mat[1:-1] ]
    return labels, [float(t) for t in th]

# EVOLVE-BLOCK-END
