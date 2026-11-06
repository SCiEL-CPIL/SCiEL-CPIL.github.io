"""Broadband antireflection coating design.

Structural approach: a fast vectorized thin-film characteristic-matrix
reflectance, combined with a hybrid optimizer that searches discrete
layer-sequence structures (deterministic + random) and refines each with a
Brent 1-D line search along the steepest-descent direction.  An explicit
deadline guard returns the best result with a safety margin.
"""

import math
import time
import random

import numpy as np

# Wavelength grid used for the objective (matches the evaluator's 401 points).
_WLS = np.linspace(400.0, 800.0, 401)


def _mean_reflectance(mats, th, n0, ns, wl):
    """Mean normal-incidence reflectance (fraction) over wavelength grid wl.

    Uses the thin-film characteristic-matrix method.  Vectorized over
    wavelength; mats is a sequence of complex indices, th the physical
    thicknesses (nm), n0 the ambient index and ns the substrate index.
    """
    if len(mats) == 0:
        r = (n0 - ns) / (n0 + ns)
        return float(r * r)

    n = np.asarray(mats, dtype=complex)
    t = np.asarray(th, dtype=float)
    k = 2.0 * np.pi / wl          # radians / nm
    y = n * k * t                 # optical phase per layer (radians)

    m11 = np.cos(y)
    m12 = 1j * np.sin(y) / n
    m21 = 1j * n * np.sin(y)
    m22 = np.cos(y)

    # Multiply all layer matrices (M_1 M_2 ... M_N).
    A = np.ones_like(m11)
    B = np.zeros_like(m11)
    C = np.zeros_like(m11)
    D = np.ones_like(m11)
    for j in range(len(mats)):
        a11, a12, a21, a22 = m11[j], m12[j], m21[j], m22[j]
        A, B, C, D = (A * a11 + B * a21,
                      A * a12 + B * a22,
                      C * a11 + D * a21,
                      C * a12 + D * a22)

    # Input admittance Y0 = (C + D*ns) / (A + B*ns); R = |(n0 - Y0)/(n0 + Y0)|^2
    Y0 = (C + D * ns) / (A + B * ns)
    r = (n0 - Y0) / (n0 + Y0)
    R = (r * np.conj(r)).real
    return float(np.mean(R))


def _objective(mats, th, n0, ns):
    """Objective: mean reflectance fraction over the fixed 401-point grid."""
    return _mean_reflectance(mats, th, n0, ns, _WLS)


def _is_valid(th, max_layer, max_total, max_layers, n_layers):
    if n_layers > max_layers:
        return False
    if th is None or len(th) != n_layers:
        return False
    if any(v < -1e-9 for v in th):
        return False
    if any(v > max_layer + 1e-9 for v in th):
        return False
    if sum(th) > max_total + 1e-9:
        return False
    return True


def _clip_th(th, max_layer, max_total):
    """Clamp thicknesses into [0, max_layer] and scale down to the total cap."""
    th = np.clip(np.asarray(th, dtype=float), 0.0, max_layer)
    s = th.sum()
    if s > max_total:
        th = th * (max_total / s)
        th = np.clip(th, 0.0, max_layer)
    return th


def _refine_line(labels, th, n0, ns, max_layer, max_total, deadline, iters=6):
    """Steepest-descent refinement via Brent 1-D line search.

    Each iteration computes a finite-difference gradient of the mean
    reflectance w.r.t. every thickness, takes the axis of largest negative
    gradient, and runs a scalar bounded Brent line search along it (with box
    + total-thickness feasibility handling).  Returns the best
    (score, thicknesses) encountered.
    """
    n = len(th)
    best_th = np.array(th, dtype=float)
    best_score = _objective(labels, best_th, n0, ns)
    eps = 2.0

    for _ in range(iters):
        if time.time() >= deadline:
            break
        grad = np.zeros(n)
        for i in range(n):
            hi = best_th.copy(); hi[i] = min(best_th[i] + eps, max_layer)
            lo = best_th.copy(); lo[i] = max(best_th[i] - eps, 0.0)
            if hi[i] == lo[i]:
                continue
            fh = _objective(labels, hi, n0, ns)
            fl = _objective(labels, lo, n0, ns)
            grad[i] = (fh - fl) / (hi[i] - lo[i])
        idx = int(np.argmax(-grad))
        if -grad[idx] <= 1e-9:
            break
        g = -grad[idx]
        x0 = best_th[idx]
        if g > 0:
            alpha_max = (max_layer - x0) / g
        else:
            alpha_max = (x0 - 0.0) / (-g)
        if alpha_max <= 1e-6:
            break

        def f_alpha(alpha):
            x = x0 + alpha * g
            if x < 0.0 or x > max_layer:
                return 1e9
            th2 = best_th.copy(); th2[idx] = x
            if th2.sum() > max_total:
                return 1e9
            return _objective(labels, th2, n0, ns)

        try:
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(f_alpha, bounds=(0.0, alpha_max),
                                  method='bounded',
                                  options={'xatol': 0.01, 'maxiter': 40})
            alpha = res.x
            x = x0 + alpha * g
            th2 = best_th.copy(); th2[idx] = x
            if th2.sum() > max_total:
                continue
            s2 = _objective(labels, th2, n0, ns)
            if s2 < best_score - 1e-12:
                best_score = s2
                best_th = th2
        except Exception:
            break
    return best_score, best_th


def design(problem):
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mats = problem["materials"]
    nl = mats["L"]
    nh = mats["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    start = time.time()
    # Safety margin: stop searching well before the 120 s evaluation limit.
    deadline = start + 100.0

    rng = random.Random(12345)
    centre = 0.5 * (band[0] + band[1])
    lam = centre  # reference wavelength (nm)

    best_labels = ["L"]
    best_th = [lam / (4.0 * nl)]
    best_score = _objective(best_labels, best_th, n0, ns)

    def consider(labels, th):
        nonlocal best_labels, best_th, best_score
        th = _clip_th(th, max_layer, max_total)
        if not _is_valid(th, max_layer, max_total, max_layers, len(labels)):
            return
        s = _objective(labels, th, n0, ns)
        if s < best_score - 1e-12:
            best_score = s
            best_labels = list(labels)
            best_th = [float(v) for v in th]

    # ---- Deterministic candidate structures ----
    for label, nmat in (("L", nl), ("H", nh)):
        consider([label], [lam / (4.0 * nmat)])

    for seq in (("L", "H"), ("H", "L")):
        th = [lam / (4.0 * (nl if lab == "L" else nh)) for lab in seq]
        consider(list(seq), th)

    consider(["L", "H", "L"],
             [lam / (4.0 * nl), lam / (4.0 * nh), lam / (4.0 * nl)])

    for nrep in (2, 3, 4):
        labels, th = [], []
        for kk in range(nrep):
            lab = "L" if kk % 2 == 0 else "H"
            nmat = nl if lab == "L" else nh
            labels.append(lab)
            th.append(lam / (4.0 * nmat))
        if len(labels) <= max_layers:
            consider(labels, th)

    # ---- Hybrid search: random discrete structures + line-search refine ----
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        n = rng.randint(2, min(8, max_layers))
        labels = [rng.choice("LH") for _ in range(n)]
        lam_r = rng.uniform(band[0], band[1])
        th = []
        for lab in labels:
            nmat = nl if lab == "L" else nh
            base = lam_r / (4.0 * nmat)
            th.append(rng.uniform(0.0, max_layer) * 0.5 + base * 0.5)
        consider(labels, th)

        if time.time() < deadline - 5.0:
            s2, th2 = _refine_line(best_labels, best_th, n0, ns,
                                   max_layer, max_total, deadline, iters=6)
            if s2 < best_score - 1e-12:
                best_score = s2
                best_th = [float(v) for v in th2]

        if attempts % 3 == 0 and time.time() < deadline - 5.0:
            s3, th3 = _refine_line(labels, th, n0, ns,
                                   max_layer, max_total, deadline, iters=4)
            if s3 < best_score - 1e-12:
                best_score = s3
                best_labels = list(labels)
                best_th = [float(v) for v in th3]

    # Final safety: ensure the returned design is valid and within caps.
    best_th = _clip_th(best_th, max_layer, max_total)
    if not _is_valid(best_th, max_layer, max_total, max_layers, len(best_labels)):
        best_labels = ["L"]
        best_th = [ (0.5 * (band[0] + band[1])) / (4.0 * nl) ]

    return best_labels, [float(v) for v in best_th]
