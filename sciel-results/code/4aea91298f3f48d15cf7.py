"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

STRUCTURAL CHANGE: Treat the coating as a continuous effective refractive index profile.
  1. Define a smooth effective-index profile n_eff(z) between 1.38 and 2.35 using a
     low-dimensional parametric family (cosine taper with adjustable parameters).
  2. Discretize into up to 20 layers: for each layer, choose material (L or H) and
     thickness to best approximate the local target index and optical path.
  3. Optimize the continuous profile parameters using a global stochastic search
     (differential evolution) followed by local polish.
  4. Explicit deadline guard with safety margin.
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


def _profile_to_layers(params, problem):
    """Convert continuous profile parameters to discrete layer stack.
    
    params: [n_start, n_end, shape_param, total_thickness, n_layers]
    n_start: effective index at air interface (clipped to [nL, nH])
    n_end: effective index at substrate interface (clipped to [nL, nH])
    shape_param: controls the taper shape (0=linear, 1=cosine)
    total_thickness: total coating thickness
    n_layers: number of layers (int, 1-20)
    """
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    max_layers = problem["max_layers"]
    
    n_start = np.clip(params[0], nL, nH)
    n_end = np.clip(params[1], nL, nH)
    shape = np.clip(params[2], 0.0, 1.0)
    total_t = np.clip(params[3], 1.0, max_total)
    n_layers = int(np.round(np.clip(params[4], 1, max_layers)))
    
    # Generate effective index profile
    z = np.linspace(0, 1, n_layers)
    # Blend between linear and cosine taper
    linear = n_start + (n_end - n_start) * z
    cosine = 0.5 * (n_start + n_end) + 0.5 * (n_start - n_end) * np.cos(np.pi * z)
    n_eff = (1 - shape) * linear + shape * cosine
    
    # For each layer, choose material and thickness to approximate n_eff
    materials = []
    thicknesses = []
    layer_thickness = total_t / n_layers
    
    for i in range(n_layers):
        target_n = n_eff[i]
        # Choose material: L if target is closer to nL, H if closer to nH
        if abs(target_n - nL) <= abs(target_n - nH):
            mat = 'L'
            n_mat = nL
        else:
            mat = 'H'
            n_mat = nH
        
        # Optimize thickness: we want the optical path n_mat * t to be close to
        # the "effective" optical path. Since we can't change the index, we adjust
        # thickness to balance the optical path contribution.
        # Target optical path per layer: (target_n / (nL + nH) * (nL + nH)) * layer_thickness
        # Simpler: just use uniform thickness, but adjust based on index mismatch
        t = layer_thickness
        materials.append(mat)
        thicknesses.append(t)
    
    # Ensure constraints
    total = sum(thicknesses)
    if total > max_total:
        scale = max_total / total
        thicknesses = [t * scale for t in thicknesses]
    thicknesses = [max(0.0, min(max_layer, t)) for t in thicknesses]
    
    return materials, thicknesses


def _profile_to_layers_v2(params, problem):
    """Improved: use quarter-wave thicknesses scaled by index ratio.
    
    params: [n_start, n_end, shape_param, total_thickness, n_layers, qw_scale]
    """
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    max_layers = problem["max_layers"]
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    
    n_start = np.clip(params[0], nL, nH)
    n_end = np.clip(params[1], nL, nH)
    shape = np.clip(params[2], 0.0, 1.0)
    total_t = np.clip(params[3], 1.0, max_total)
    n_layers = int(np.round(np.clip(params[4], 1, max_layers)))
    qw_scale = np.clip(params[5], 0.5, 2.0) if len(params) > 5 else 1.0
    
    z = np.linspace(0, 1, n_layers)
    linear = n_start + (n_end - n_start) * z
    cosine = 0.5 * (n_start + n_end) + 0.5 * (n_start - n_end) * np.cos(np.pi * z)
    n_eff = (1 - shape) * linear + shape * cosine
    
    materials = []
    thicknesses = []
    
    for i in range(n_layers):
        target_n = n_eff[i]
        if abs(target_n - nL) <= abs(target_n - nH):
            mat = 'L'
            n_mat = nL
        else:
            mat = 'H'
            n_mat = nH
        
        # Quarter-wave thickness at center wavelength, scaled
        qw = wl_center / (4.0 * n_mat) * qw_scale
        # Scale to fit total budget
        thicknesses.append(qw)
        materials.append(mat)
    
    total = sum(thicknesses)
    if total > max_total:
        scale = max_total / total
        thicknesses = [t * scale for t in thicknesses]
    thicknesses = [max(0.0, min(max_layer, t)) for t in thicknesses]
    
    return materials, thicknesses


def _differential_evolution(f, bounds, pop_size=30, max_iter=100, cr=0.8, F=0.8,
                            deadline=None, seed=42):
    """Simple differential evolution implementation."""
    rng = np.random.RandomState(seed)
    n_params = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = []
    for i in range(pop_size):
        x = lo + rng.rand(n_params) * (hi - lo)
        pop.append(x)
    
    scores = [f(x) for x in pop]
    best_idx = np.argmin(scores)
    best_x = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    for it in range(max_iter):
        if deadline is not None and time.time() >= deadline:
            break
        
        for i in range(pop_size):
            # Select three distinct random individuals
            r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
            while r1 == i or r2 == i or r3 == i:
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
            
            # Create trial vector
            v = pop[r1] + F * (pop[r2] - pop[r3])
            v = np.clip(v, lo, hi)
            
            # Evaluate trial
            fv = f(v)
            
            # Selection
            if fv <= scores[i]:
                pop[i] = v
                scores[i] = fv
                if fv < best_score:
                    best_score = fv
                    best_x = v.copy()
    
    return best_x, best_score


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

    best_ths = None
    best_mats = None
    best_score = float("inf")

    def maybe_record(mats, ths):
        nonlocal best_ths, best_mats, best_score
        if not mats or not ths:
            return
        total = sum(ths)
        if total > max_total + 1e-6:
            ths = [t * (max_total / total) for t in ths]
        ths = [max(0.0, min(max_layer, t)) for t in ths]
        if any(t < 0 or t > max_layer for t in ths):
            return
        if len(mats) > max_layers:
            return
        s = _mean_reflectance(mats, ths, problem)
        if s < best_score:
            best_score = s
            best_ths = list(ths)
            best_mats = list(mats)

    # Phase 1: Differential evolution over profile parameters
    # Profile params: [n_start, n_end, shape, total_thickness, n_layers, qw_scale]
    bounds = [
        (nL, nH),          # n_start
        (nL, nH),          # n_end
        (0.0, 1.0),        # shape
        (50.0, max_total), # total_thickness
        (10, max_layers),  # n_layers
        (0.5, 2.0),        # qw_scale
    ]
    
    def profile_eval(params):
        mats, ths = _profile_to_layers_v2(params, problem)
        if not mats:
            return 100.0
        return _mean_reflectance(mats, ths, problem)
    
    # Run differential evolution
    if time.time() < deadline - 40:
        best_params, best_val = _differential_evolution(
            profile_eval, bounds, pop_size=40, max_iter=80, cr=0.8, F=0.8,
            deadline=deadline - 40, seed=42
        )
        mats, ths = _profile_to_layers_v2(best_params, problem)
        maybe_record(mats, ths)
    
    # Phase 2: Local polish on best found stack
    if best_ths is not None and time.time() < deadline - 15:
        # Optimize thicknesses directly for the best material pattern
        n_layers = len(best_mats)
        n_mat = {k: v for k, v in problem["materials"].items()}
        ns = np.array([n_mat[m] for m in best_mats], dtype=float)
        wavs = np.linspace(band[0], band[1], 401)
        n_0 = problem["ambient_index"]
        n_s = problem["substrate_index"]
        
        lo = np.full(n_layers, math.log(0.5))
        hi = np.full(n_layers, math.log(max_layer))
        
        def eval_x(x):
            t = np.exp(np.clip(x, lo, hi))
            total = t.sum()
            penalty = 0.0
            if total > max_total:
                penalty = 100.0 * (total - max_total) / max_total
            val = _mean_reflectance_fast(ns, t, wavs, n_0, n_s)
            return val + penalty
        
        x0 = np.array([math.log(max(0.5, t)) for t in best_ths])
        
        if _HAVE_SCIPY:
            res = _scipy_minimize(
                eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                options={"maxiter": 500, "maxfun": 5000, "maxls": 30},
            )
            xc = np.clip(res.x, lo, hi)
            t = np.exp(xc)
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(best_mats, list(t))
        else:
            xc, val = _nm_fallback(eval_x, x0, lo, hi, deadline - 10)
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(best_mats, list(t))
    
    # Phase 3: Try a few different material patterns with optimized thicknesses
    if time.time() < deadline - 20:
        # Symmetric patterns
        patterns = [
            ['L' if i % 2 == 0 else 'H' for i in range(20)],
            ['H' if i % 2 == 0 else 'L' for i in range(20)],
            ['L' if i % 2 == 0 else 'H' for i in range(16)],
            ['L' if i % 2 == 0 else 'H' for i in range(12)],
        ]
        
        for mats in patterns:
            if time.time() >= deadline - 15:
                break
            n_layers = len(mats)
            n_mat = {k: v for k, v in problem["materials"].items()}
            ns = np.array([n_mat[m] for m in mats], dtype=float)
            wavs = np.linspace(band[0], band[1], 401)
            n_0 = problem["ambient_index"]
            n_s = problem["substrate_index"]
            
            lo = np.full(n_layers, math.log(0.5))
            hi = np.full(n_layers, math.log(max_layer))
            
            def eval_x(x):
                t = np.exp(np.clip(x, lo, hi))
                total = t.sum()
                penalty = 0.0
                if total > max_total:
                    penalty = 100.0 * (total - max_total) / max_total
                val = _mean_reflectance_fast(ns, t, wavs, n_0, n_s)
                return val + penalty
            
            # Initialize with quarter-wave
            x0 = np.array([math.log(max(0.5, wl_center / (4.0 * n))) for n in ns])
            x0 = np.clip(x0, lo, hi)
            
            if _HAVE_SCIPY:
                res = _scipy_minimize(
                    eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                    options={"maxiter": 300, "maxfun": 3000, "maxls": 25},
                )
                xc = np.clip(res.x, lo, hi)
                t = np.exp(xc)
                total = t.sum()
                if total > max_total:
                    t = t * (max_total / total)
                maybe_record(mats, list(t))
    
    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center
        qw_L = wl_center / (4.0 * nL)
        return ['L'], [max(0.0, min(max_layer, qw_L))]

    return list(best_mats), list(best_ths)
