"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

MODEL-AND-OPTIMIZE strategy:
  1. Exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Unconstrained independent-thickness parameterization (each layer optimized freely),
     expressed in log-scale so the optimizer explores the full [0, max_layer] range.
  3. Fast L-BFGS-B local optimization (or a lightweight Nelder-Mead fallback) run from
     many diverse random starting points so the objective's many basins are explored.
  4. Explicit deadline guard: return the best feasible result found before the limit.
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


def _make_evaluator(materials, problem, max_total, max_layer):
    """Return (eval, lo, hi). eval(x) takes a log-scaled length vector x (length n)
    and returns the mean reflectance for the corresponding physical thicknesses,
    where t_j = clip(exp(x_j), 0, max_layer) and the whole stack is scaled to satisfy
    sum(t) <= max_total."""
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    n = len(materials)
    lo = np.full(n, math.log(1.0))
    hi = np.full(n, math.log(max_layer))

    def eval_x(x):
        t = np.exp(np.clip(x, lo, hi))
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
        return _mean_reflectance(materials, list(t), problem)

    return eval_x, lo, hi


def _nm_fallback(f, x0, lo, hi, deadline):
    """Lightweight Nelder-Mead (constrained to box + total) as a scipy fallback."""
    n = len(x0)
    best = list(x0)
    best_val = f(best)
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        # Use a larger initial step for better exploration
        pt[i] = hi[i] if pt[i] == lo[i] else lo[i] + 0.2 * (hi[i] - lo[i])
        simplex.append(pt)
    scores = [f(pt) for pt in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    it = 0
    # Increase iteration limit for more thorough local search
    while it < 800 and time.time() < deadline:
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


def design(problem):
    start_time = time.time()
    deadline = start_time + 120  # safety margin before the 150 s kill

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

    # Precompute a per-layer-count evaluator once; build stacks on the fly.
    def optimize_layer_count(n_layers):
        nonlocal best_ths, best_mats, best_score
        if time.time() >= deadline - 8:
            return
        materials = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        eval_x, lo, hi = _make_evaluator(materials, problem, max_total, max_layer)

        # A few diverse starting points for this layer count.
    # Use more starts for smaller layer counts, and seed from best solution when available.
        n_starts = 6 if n_layers <= 6 else (4 if n_layers <= 10 else 2)
        for start_idx in range(n_starts):
            if time.time() >= deadline - 8:
                return
            if start_idx == 0 and best_ths is not None and len(best_ths) == n_layers:
                # Seed from best solution found so far (rescaled if needed)
                x0 = []
                for t in best_ths:
                    t = max(1.0, min(max_layer, t))
                    x0.append(math.log(t))
            else:
                x0 = []
                for m in materials:
                    qw = qw_L if m == 'L' else qw_H
                    # More diverse range: 0.3 to 2.0 times quarter-wave
                    t = qw * random.uniform(0.3, 2.0)
                    t = max(1.0, min(max_layer, t))
                    x0.append(math.log(t))
            x0 = np.array(x0)
            if _HAVE_SCIPY:
                res = _scipy_minimize(
                    eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                    options={"maxiter": 300, "maxfun": 4000, "maxls": 30},
                )
                xc = np.clip(res.x, lo, hi)
                val = eval_x(list(xc))
            else:
                xc, val = _nm_fallback(eval_x, x0, lo, hi, deadline - 8)
            # Convert back to physical thicknesses.
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(materials, list(t))

    # Explore a spread of layer counts, denser where solutions are known to live.
    counts = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    counts = [c for c in counts if c <= max_layers]
    # Interleave small and large so we get a usable fallback early.
    order = []
    i, j = 0, len(counts) - 1
    while i <= j:
        order.append(counts[i])
        if i != j:
            order.append(counts[j])
        i += 1
        j -= 1

    for n_layers in order:
        if time.time() >= deadline - 25:
            break
        optimize_layer_count(n_layers)

    # Final refinement pass on the single best stack found, if time allows.
    # Use higher precision and more iterations for a tight local solve.
    if best_ths is not None and time.time() < deadline - 6:
        eval_x, lo, hi = _make_evaluator(best_mats, problem, max_total, max_layer)
        x0 = [math.log(max(1.0, t)) for t in best_ths]
        if _HAVE_SCIPY:
            # First pass: L-BFGS-B with tighter tolerance
            res = _scipy_minimize(
                eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                options={"maxiter": 500, "maxfun": 10000, "maxls": 50,
                         "ftol": 1e-10, "gtol": 1e-8},
            )
            xc = np.clip(res.x, lo, hi)
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(best_mats, list(t))
            
            # Second pass: try Nelder-Mead from the refined point for potential escape
            if time.time() < deadline - 4:
                x0_nm = [math.log(max(1.0, t)) for t in t]
                xc_nm, val_nm = _nm_fallback(eval_x, x0_nm, lo, hi, deadline - 4)
                t_nm = np.exp(np.clip(xc_nm, lo, hi))
                total_nm = t_nm.sum()
                if total_nm > max_total:
                    t_nm = t_nm * (max_total / total_nm)
                maybe_record(best_mats, list(t_nm))

    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center.
        return ['L'], [max(0.0, min(max_layer, qw_L))]

    return list(best_mats), list(best_ths)
