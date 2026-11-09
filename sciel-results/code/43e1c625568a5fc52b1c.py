"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

DUAL STRATEGY:
  1. Exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Fixed 20-layer alternating L/H structure (graded-index approximation) to remove
     combinatorial search and focus optimization on continuous thicknesses.
  3. Differential evolution (population-based global optimizer) for robust global search,
     followed by L-BFGS-B local refinement.
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


def _differential_evolution(f, lo, hi, pop_size=20, max_iter=50, deadline=None):
    """Simple differential evolution for global optimization."""
    n = len(lo)
    lo = np.array(lo)
    hi = np.array(hi)
    
    # Initialize population
    pop = np.random.uniform(lo, hi, size=(pop_size, n))
    scores = [f(p) for p in pop]
    
    # Track best
    best_idx = np.argmin(scores)
    best_x = pop[best_idx].copy()
    best_val = scores[best_idx]
    
    it = 0
    while it < max_iter:
        if deadline is not None and time.time() >= deadline:
            break
        it += 1
        
        # Shuffle population indices
        perm = np.random.permutation(pop_size)
        
        for i in range(pop_size):
            if deadline is not None and time.time() >= deadline:
                break
            
            # Select three distinct individuals
            a, b, c = perm[(i + 1) % pop_size], perm[(i + 2) % pop_size], perm[(i + 3) % pop_size]
            while a == b or a == c or b == c:
                a = np.random.randint(pop_size)
                b = np.random.randint(pop_size)
                c = np.random.randint(pop_size)
            
            # Differential mutation
            F = 0.8
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, lo, hi)
            
            # Crossover
            CR = 0.9
            j_rand = np.random.randint(n)
            u = np.random.uniform(size=n)
            trial = np.where(u < CR, v, pop[i])
            trial[j_rand] = v[j_rand]
            
            # Selection
            trial_score = f(trial)
            if trial_score < scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score < best_val:
                    best_val = trial_score
                    best_x = trial.copy()
    
    return best_x, best_val


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

    # Strategy: Focus on 20-layer structure with differential evolution for global search
    # This removes combinatorial search and allows deeper optimization of thicknesses
    
    n_layers_20 = min(20, max_layers)
    materials_20 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers_20)]
    eval_x_20, lo_20, hi_20 = _make_evaluator(materials_20, problem, max_total, max_layer)
    
    # Phase 1: Differential evolution for global search (use ~70% of time budget)
    if time.time() < deadline - 40:
        de_deadline = deadline - 40
        # Use more population and iterations for better global coverage
        best_x_de, best_val_de = _differential_evolution(
            eval_x_20, lo_20, hi_20, 
            pop_size=30, max_iter=100, 
            deadline=de_deadline
        )
        # Convert back to physical thicknesses
        t_de = np.exp(np.clip(best_x_de, lo_20, hi_20))
        total_de = t_de.sum()
        if total_de > max_total:
            t_de = t_de * (max_total / total_de)
        maybe_record(materials_20, list(t_de))
    
    # Phase 2: Local refinement with L-BFGS-B from multiple starting points
    if best_ths is not None and time.time() < deadline - 15:
        eval_x, lo, hi = _make_evaluator(best_mats, problem, max_total, max_layer)
        
        # Try L-BFGS-B from the DE result and a few other diverse starts
        starts = []
        
        # Start 1: From best solution found so far
        if best_ths is not None and len(best_ths) == n_layers_20:
            x0_1 = [math.log(max(1.0, min(max_layer, t))) for t in best_ths]
            starts.append(np.array(x0_1))
        
        # Start 2-4: Quarter-wave based diverse starts
        for k in range(3):
            x0 = []
            for m in materials_20:
                qw = qw_L if m == 'L' else qw_H
                # Vary the scaling factor for diversity
                scale = random.uniform(0.5, 1.5)
                t = qw * scale
                t = max(1.0, min(max_layer, t))
                x0.append(math.log(t))
            starts.append(np.array(x0))
        
        for x0 in starts:
            if time.time() >= deadline - 10:
                break
            if _HAVE_SCIPY:
                res = _scipy_minimize(
                    eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                    options={"maxiter": 500, "maxfun": 8000, "maxls": 40,
                             "ftol": 1e-10, "gtol": 1e-8},
                )
                xc = np.clip(res.x, lo, hi)
                t = np.exp(np.clip(xc, lo, hi))
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
    
    # Phase 3: Final tight local refinement if time allows
    if best_ths is not None and time.time() < deadline - 5:
        eval_x, lo, hi = _make_evaluator(best_mats, problem, max_total, max_layer)
        x0 = [math.log(max(1.0, min(max_layer, t))) for t in best_ths]
        if _HAVE_SCIPY:
            res = _scipy_minimize(
                eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                options={"maxiter": 1000, "maxfun": 15000, "maxls": 60,
                         "ftol": 1e-12, "gtol": 1e-10},
            )
            xc = np.clip(res.x, lo, hi)
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(best_mats, list(t))
    
    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center.
        return ['L'], [max(0.0, min(max_layer, qw_L))]

    return list(best_mats), list(best_ths)
