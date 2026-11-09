"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

GRADIENT-PROFILE + NELLER-MEAD strategy:
  1. Fixed alternating L/H structure with variable layer count (4-20).
  2. Exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  3. Multi-start Nelder-Mead local search in log-thickness space, initialized
     from quarter-wave analytical solutions and diverse random perturbations.
  4. Soft constraint on total thickness to avoid nonsmooth scaling.
  5. Explicit deadline guard: return the best feasible result found before the limit.
"""

import time
import math
import random
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _mean_reflectance(materials, thicknesses, problem):
    """Mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)

    n_ds = np.array([n * d for n, d in zip(ns, ds)])
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + i * B * n_j * sj
        B_new = i * A * sj / n_j + B * cj
        C_new = C * cj + i * D * n_j * sj
        D_new = i * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def _mean_reflectance_fast(ns, ds, wavs, n_0, n_s):
    """Fast version with precomputed wavelengths and index array."""
    n_ds = ns * ds
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(ns)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + i * B * n_j * sj
        B_new = i * A * sj / n_j + B * cj
        C_new = C * cj + i * D * n_j * sj
        D_new = i * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def _make_evaluator(materials, problem, max_total, max_layer):
    """Return (eval, lo, hi). eval(x) takes a log-scaled length vector x (length n)
    and returns the mean reflectance with a soft penalty for exceeding max_total."""
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    n = len(materials)
    lo = np.full(n, math.log(1.0))
    hi = np.full(n, math.log(max_layer))
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    # Soft penalty weight for total thickness constraint
    penalty_weight = 50.0

    def eval_x(x):
        t = np.exp(np.clip(x, lo, hi))
        total = t.sum()
        r = _mean_reflectance_fast(ns, t, wavs, n_0, n_s)
        if total > max_total:
            # Soft penalty: quadratic in excess
            excess = total - max_total
            r += penalty_weight * (excess / max_total) ** 2
        return r

    return eval_x, lo, hi


def _nm_fallback(f, x0, lo, hi, deadline):
    """Lightweight Nelder-Mead (constrained to box) as a scipy fallback."""
    n = len(x0)
    best = list(x0)
    best_val = f(best)
    simplex = [list(x0)]
    # Use adaptive initial step based on box size
    for i in range(n):
        pt = list(x0)
        step = 0.15 * (hi[i] - lo[i])
        pt[i] = min(hi[i], max(lo[i], x0[i] + step))
        simplex.append(pt)
    scores = [f(pt) for pt in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    it = 0
    max_iters = 600
    while it < max_iters and time.time() < deadline:
        it += 1
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        if scores[0] < best_val:
            best_val = scores[0]
            best = list(simplex[0])
        centroid = np.mean(simplex[:-1], axis=0)
        xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        fr = f(list(xr))
        if fr < scores[0]:
            xe = np.clip(centroid + gamma * (centroid - simplex[-1]), lo, hi)
            fe = f(list(xe))
            if fe < fr:
                simplex[-1], scores[-1] = list(xe), fe
            else:
                simplex[-1], scores[-1] = list(xr), fr
        elif fr < scores[-2]:
            simplex[-1], scores[-1] = list(xr), fr
        else:
            xc = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
            fc = f(list(xc))
            if fc <= fr:
                simplex[-1], scores[-1] = list(xc), fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = np.clip(
                        np.array(simplex[0]) + sigma * (np.array(simplex[i]) - np.array(simplex[0])),
                        lo, hi)
                    scores[i] = f(list(simplex[i]))
                if max(scores) - min(scores) < 1e-10:
                    break
    return best, best_val


def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before the 150 s kill

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    random.seed(2024)
    np.random.seed(2024)

    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    qw_L = wl_center / (4.0 * nL)
    qw_H = wl_center / (4.0 * nH)

    best_ths = None
    best_mats = None
    best_score = float("inf")

    def maybe_record(mats, ths):
        nonlocal best_ths, best_mats, best_score
        total = sum(ths)
        if total > max_total + 1e-6:
            ths = [t * (max_total / total) for t in ths]
        ths = [max(0.0, min(max_layer, t)) for t in ths]
        if any(t < 0 or t > max_layer for t in ths):
            return
        s = _mean_reflectance(mats, ths, problem)
        if s < best_score:
            best_score = s
            best_ths = list(ths)
            best_mats = list(mats)

    # Try multiple layer counts to find the optimal trade-off
    layer_counts = [20, 18, 16, 14, 12, 10, 8, 6]
    layer_counts = [lc for lc in layer_counts if lc <= max_layers]

    for n_layers in layer_counts:
        if time.time() >= deadline - 20:
            break
        
        materials = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        eval_x, lo, hi = _make_evaluator(materials, problem, max_total, max_layer)

        # Generate diverse starting points
        n_starts = 6
        starts = []
        
        # Start 1: quarter-wave analytical solution
        x0_qw = []
        for m in materials:
            qw = qw_L if m == 'L' else qw_H
            t = max(1.0, min(max_layer, qw))
            x0_qw.append(math.log(t))
        starts.append(np.array(x0_qw))
        
        # Starts 2-6: diverse random perturbations around quarter-wave
        for _ in range(n_starts - 1):
            x0 = []
            for m in materials:
                qw = qw_L if m == 'L' else qw_H
                t = qw * random.uniform(0.3, 2.0)
                t = max(1.0, min(max_layer, t))
                x0.append(math.log(t))
            starts.append(np.array(x0))

        # Run Nelder-Mead from each start
        for start_idx, x0 in enumerate(starts):
            if time.time() >= deadline - 15:
                break
            if _HAVE_SCIPY:
                res = _scipy_minimize(
                    eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                    options={"maxiter": 150, "maxfun": 2000, "maxls": 15},
                )
                xc = np.clip(res.x, lo, hi)
            else:
                xc, _ = _nm_fallback(eval_x, x0, lo, hi, deadline - 15)
            # Convert back to physical thicknesses.
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(materials, list(t))

    # Final refinement pass on the single best stack found, if time allows.
    if best_ths is not None and time.time() < deadline - 10:
        eval_x, lo, hi = _make_evaluator(best_mats, problem, max_total, max_layer)
        x0 = [math.log(max(1.0, t)) for t in best_ths]
        if _HAVE_SCIPY:
            res = _scipy_minimize(
                eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                options={"maxiter": 400, "maxfun": 8000, "maxls": 30},
            )
            xc = np.clip(res.x, lo, hi)
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(best_mats, list(t))
        else:
            xc, _ = _nm_fallback(eval_x, x0, lo, hi, deadline - 5)
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(best_mats, list(t))

    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center.
        return ['L'], [max(0.0, min(max_layer, qw_L))]

    return list(best_mats), list(best_ths)
