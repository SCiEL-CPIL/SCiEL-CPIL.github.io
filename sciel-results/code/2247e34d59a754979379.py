"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

IMPROVED STRATEGY:
  1. Variable layer count (8-20) with alternating L/H structure
  2. Graded-index initialization with smooth thickness profiles
  3. Penalty-based constraint handling for total thickness
  4. Multi-start L-BFGS-B with diverse initializations
  5. Final polish pass with tighter tolerances
  6. Explicit deadline guard with safety margin
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


def _make_evaluator(materials, problem, max_total, max_layer, penalty_weight=100.0):
    """Return (eval, lo, hi). eval(x) takes a log-scaled length vector x (length n)
    and returns the mean reflectance plus penalty for constraint violations."""
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    n = len(materials)
    lo = np.full(n, math.log(0.5))
    hi = np.full(n, math.log(max_layer))
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]

    def eval_x(x):
        t = np.exp(np.clip(x, lo, hi))
        # Penalty for exceeding total thickness
        total = t.sum()
        penalty = 0.0
        if total > max_total:
            penalty = penalty_weight * (total - max_total) / max_total
        val = _mean_reflectance_fast(ns, t, wavs, n_0, n_s)
        return val + penalty

    return eval_x, lo, hi


def _nm_fallback(f, x0, lo, hi, deadline):
    """Lightweight Nelder-Mead (constrained to box) as a scipy fallback."""
    n = len(x0)
    best = list(x0)
    best_val = f(best)
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        step = 0.3 * (hi[i] - lo[i])
        pt[i] = lo[i] + step
        simplex.append(pt)
    scores = [f(pt) for pt in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    it = 0
    while it < 600 and time.time() < deadline:
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
                if max(scores) - min(scores) < 1e-9:
                    break
    return best, best_val


def _generate_graded_init(n_layers, nL, nH, max_total, max_layer, wl_center, mode='linear'):
    """Generate a graded-index initial thickness profile."""
    materials = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    qw_L = wl_center / (4.0 * nL)
    qw_H = wl_center / (4.0 * nH)
    
    thicknesses = []
    for i in range(n_layers):
        frac = i / max(1, n_layers - 1)
        if mode == 'linear':
            # Linear interpolation between low and high quarter-wave
            t = qw_L * (1 - frac) + qw_H * frac
        elif mode == 'cosine':
            # Cosine taper
            t = 0.5 * (qw_L + qw_H) + 0.5 * (qw_H - qw_L) * math.cos(math.pi * frac)
        elif mode == 'smoothstep':
            # Smoothstep for smoother transitions
            s = frac * frac * (3 - 2 * frac)
            t = qw_L * (1 - s) + qw_H * s
        else:
            t = 0.5 * (qw_L + qw_H)
        thicknesses.append(max(1.0, min(max_layer, t)))
    
    # Scale to fit within max_total
    total = sum(thicknesses)
    if total > max_total:
        scale = max_total / total
        thicknesses = [t * scale for t in thicknesses]
    
    return materials, thicknesses


def design(problem):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    random.seed(42)
    np.random.seed(42)

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

    # Try multiple layer counts and initialization strategies
    layer_counts = [10, 12, 14, 16, 18, 20]
    init_modes = ['linear', 'cosine', 'smoothstep']
    
    # Collect all candidate starting points
    candidates = []
    
    for n_layers in layer_counts:
        for mode in init_modes:
            if time.time() >= deadline - 20:
                break
            materials, thicknesses = _generate_graded_init(
                n_layers, nL, nH, max_total, max_layer, wl_center, mode
            )
            x0 = np.array([math.log(max(0.5, t)) for t in thicknesses])
            candidates.append((materials, x0, n_layers))
    
    # Add pure quarter-wave starts
    for n_layers in [12, 16, 20]:
        if time.time() >= deadline - 20:
            break
        materials = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        thicknesses = []
        for m in materials:
            qw = qw_L if m == 'L' else qw_H
            t = max(1.0, min(max_layer, qw))
            thicknesses.append(t)
        # Scale to fit
        total = sum(thicknesses)
        if total > max_total:
            scale = max_total / total
            thicknesses = [t * scale for t in thicknesses]
        x0 = np.array([math.log(max(0.5, t)) for t in thicknesses])
        candidates.append((materials, x0, n_layers))
    
    # Add random perturbations around best graded init
    if candidates:
        base_mats, base_x0, base_n = candidates[0]
        for _ in range(6):
            if time.time() >= deadline - 20:
                break
            x0 = base_x0 + np.random.normal(0, 0.3, size=base_n)
            x0 = np.clip(x0, np.full(base_n, math.log(0.5)), np.full(base_n, math.log(max_layer)))
            candidates.append((base_mats, x0, base_n))

    # Optimize each candidate
    for idx, (materials, x0, n_layers) in enumerate(candidates):
        if time.time() >= deadline - 15:
            break
        eval_x, lo, hi = _make_evaluator(materials, problem, max_total, max_layer)
        
        if _HAVE_SCIPY:
            res = _scipy_minimize(
                eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                options={"maxiter": 300, "maxfun": 4000, "maxls": 25},
            )
            xc = np.clip(res.x, lo, hi)
            t = np.exp(xc)
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(materials, list(t))
        else:
            xc, val = _nm_fallback(eval_x, x0, lo, hi, deadline - 15)
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(materials, list(t))

    # Final polish pass on the best stack found
    if best_ths is not None and time.time() < deadline - 10:
        eval_x, lo, hi = _make_evaluator(best_mats, problem, max_total, max_layer)
        x0 = np.array([math.log(max(0.5, t)) for t in best_ths])
        
        if _HAVE_SCIPY:
            # Multiple polish attempts with different initial perturbations
            for attempt in range(3):
                if time.time() >= deadline - 5:
                    break
                if attempt == 0:
                    x_start = x0
                else:
                    x_start = x0 + np.random.normal(0, 0.1, size=len(x0))
                    x_start = np.clip(x_start, lo, hi)
                
                res = _scipy_minimize(
                    eval_x, x_start, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                    options={"maxiter": 500, "maxfun": 8000, "maxls": 30},
                )
                xc = np.clip(res.x, lo, hi)
                t = np.exp(xc)
                total = t.sum()
                if total > max_total:
                    t = t * (max_total / total)
                maybe_record(best_mats, list(t))
        else:
            xc, val = _nm_fallback(eval_x, x0, lo, hi, deadline - 5)
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(best_mats, list(t))

    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center
        return ['L'], [max(0.0, min(max_layer, qw_L))]

    return list(best_mats), list(best_ths)
