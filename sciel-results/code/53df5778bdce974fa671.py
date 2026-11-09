"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

COMBINED strategy:
  1. Exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Multi-start local optimization using both L-BFGS-B (scipy) and Nelder-Mead.
  3. Two parameterizations:
     a) Independent log-scale thicknesses (current program's approach)
     b) Gradient-parameterized stacks (inspiration's approach) with linear ramp
  4. Explore a range of layer counts with diverse starting points.
  5. Final high-precision polish on the best solution found.
  6. Explicit deadline guard: return the best feasible result before the limit.
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


def _constrain_thicknesses(ths, max_total, max_layer):
    """Constrain thicknesses to be within [0, max_layer] and sum <= max_total."""
    ths = [max(0.0, min(max_layer, t)) for t in ths]
    total = sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = [t * scale for t in ths]
        # Re-clamp after scaling
        ths = [max(0.0, min(max_layer, t)) for t in ths]
    return ths


def _make_independent_evaluator(materials, problem, max_total, max_layer):
    """Return (eval, lo, hi) for independent log-scale thicknesses."""
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


def _gradient_thicknesses(num_layers, t0, t1, max_total, max_layer):
    """Generate a linear-gradient stack of alternating L/H layers."""
    materials = []
    thicknesses = []
    if num_layers == 1:
        materials = ['L']
        thicknesses = [max(0.0, min(max_layer, t0))]
    else:
        for i in range(num_layers):
            frac = i / (num_layers - 1)
            t = t0 + (t1 - t0) * frac
            t = max(0.0, min(max_layer, t))
            materials.append('L' if i % 2 == 0 else 'H')
            thicknesses.append(t)
    
    # Enforce total thickness constraint
    total = sum(thicknesses)
    if total > max_total:
        scale = max_total / total
        thicknesses = [t * scale for t in thicknesses]
        thicknesses = [max(0.0, min(max_layer, t)) for t in thicknesses]
        total2 = sum(thicknesses)
        if total2 > max_total:
            scale = max_total / total2
            thicknesses = [t * scale for t in thicknesses]
    
    return materials, thicknesses


def _make_gradient_evaluator(num_layers, problem, max_total, max_layer):
    """Return (eval, lo, hi) for gradient-parameterized stacks.
    
    Parameters are (t0, t1) - first and last layer thicknesses.
    """
    lo = np.array([5.0, 5.0])
    hi = np.array([max_layer, max_layer])

    def eval_x(x):
        t0, t1 = x
        mats, ths = _gradient_thicknesses(num_layers, t0, t1, max_total, max_layer)
        return _mean_reflectance(mats, ths, problem)

    return eval_x, lo, hi


def _nelder_mead(f, x0, lo, hi, deadline, max_iters=500):
    """Nelder-Mead local optimization with box constraints."""
    n = len(x0)
    best = list(x0)
    best_val = f(best)
    
    simplex = [list(x0)]
    step = 0.1 * (hi - lo)
    for i in range(n):
        pt = list(x0)
        pt[i] = min(hi[i], pt[i] + step[i])
        simplex.append(pt)
    
    scores = [f(pt) for pt in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    it = 0
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
                if max(scores) - min(scores) < 1e-9:
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
        ths = _constrain_thicknesses(ths, max_total, max_layer)
        if any(t < 0 or t > max_layer for t in ths):
            return
        s = _mean_reflectance(mats, ths, problem)
        if s < best_score:
            best_score = s
            best_ths = list(ths)
            best_mats = list(mats)

    def _optimize_independent(n_layers):
        """Optimize independent thicknesses for a given layer count."""
        nonlocal best_ths, best_mats, best_score
        if time.time() >= deadline - 10:
            return
        materials = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        eval_x, lo, hi = _make_independent_evaluator(materials, problem, max_total, max_layer)

        # Multiple diverse starting points
        n_starts = 6 if n_layers <= 10 else 4
        for start_i in range(n_starts):
            if time.time() >= deadline - 10:
                return
            
            # Generate starting point
            x0 = []
            for m in materials:
                qw = qw_L if m == 'L' else qw_H
                # Vary the starting thickness around quarter-wave
                factor = random.uniform(0.4, 1.8)
                t = qw * factor
                t = max(1.0, min(max_layer, t))
                x0.append(math.log(t))
            x0 = np.array(x0)
            
            if _HAVE_SCIPY:
                res = _scipy_minimize(
                    eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                    options={"maxiter": 200, "maxfun": 3000, "maxls": 30},
                )
                xc = np.clip(res.x, lo, hi)
                val = eval_x(list(xc))
            else:
                xc, val = _nelder_mead(eval_x, x0, lo, hi, deadline - 10, max_iters=300)
            
            # Convert back to physical thicknesses
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(materials, list(t))

    def _optimize_gradient(n_layers):
        """Optimize gradient-parameterized stack for a given layer count."""
        nonlocal best_ths, best_mats, best_score
        if time.time() >= deadline - 10:
            return
        eval_x, lo, hi = _make_gradient_evaluator(n_layers, problem, max_total, max_layer)

        # Multiple diverse starting points in (t0, t1) space
        n_starts = 8 if n_layers <= 12 else 5
        for start_i in range(n_starts):
            if time.time() >= deadline - 10:
                return
            
            # Generate starting point: t0 and t1 around quarter-wave values
            t0 = random.uniform(0.3 * qw_L, 1.5 * qw_L)
            t1 = random.uniform(0.3 * qw_H, 1.5 * qw_H)
            x0 = np.array([t0, t1])
            
            if _HAVE_SCIPY:
                res = _scipy_minimize(
                    eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                    options={"maxiter": 200, "maxfun": 3000, "maxls": 30},
                )
                xc = np.clip(res.x, lo, hi)
                val = eval_x(list(xc))
            else:
                xc, val = _nelder_mead(eval_x, x0, lo, hi, deadline - 10, max_iters=300)
            
            # Get the resulting thicknesses
            mats, ths = _gradient_thicknesses(n_layers, xc[0], xc[1], max_total, max_layer)
            maybe_record(mats, ths)

    # Explore layer counts with both parameterizations
    # Interleave small and large counts for early fallback
    counts = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    counts = [c for c in counts if c <= max_layers]
    
    # Create an interleaved order
    order = []
    i, j = 0, len(counts) - 1
    while i <= j:
        order.append(counts[i])
        if i != j:
            order.append(counts[j])
        i += 1
        j -= 1

    # Alternate between independent and gradient optimization
    for idx, n_layers in enumerate(order):
        if time.time() >= deadline - 25:
            break
        
        # Use independent for smaller counts, gradient for larger
        if n_layers <= 10 or idx % 2 == 0:
            _optimize_independent(n_layers)
        else:
            _optimize_gradient(n_layers)

    # Additional focused optimization on promising layer counts
    if time.time() < deadline - 20:
        for n_layers in [6, 8, 10, 12, 14, 16]:
            if n_layers > max_layers or time.time() >= deadline - 15:
                continue
            _optimize_gradient(n_layers)

    # Final high-precision polish on the best solution found
    if best_ths is not None and time.time() < deadline - 8:
        eval_x, lo, hi = _make_independent_evaluator(best_mats, problem, max_total, max_layer)
        x0 = [math.log(max(1.0, t)) for t in best_ths]
        
        if _HAVE_SCIPY:
            # Tighter tolerance for final polish
            res = _scipy_minimize(
                eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                options={"maxiter": 300, "maxfun": 5000, "maxls": 50,
                         "ftol": 1e-12, "gtol": 1e-10},
            )
            xc = np.clip(res.x, lo, hi)
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(best_mats, list(t))
        
        # Also try Nelder-Mead polish for robustness
        if time.time() < deadline - 4:
            eval_x2, lo2, hi2 = _make_independent_evaluator(best_mats, problem, max_total, max_layer)
            xc2, val2 = _nelder_mead(eval_x2, x0, lo2, hi2, deadline - 2, max_iters=400)
            t2 = np.exp(np.clip(xc2, lo2, hi2))
            total2 = t2.sum()
            if total2 > max_total:
                t2 = t2 * (max_total / total2)
            maybe_record(best_mats, list(t2))

    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center
        return ['L'], [max(0.0, min(max_layer, qw_L))]

    return list(best_mats), list(best_ths)
