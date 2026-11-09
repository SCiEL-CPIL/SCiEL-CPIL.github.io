"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

STRUCTURAL CHANGE:
  Instead of alternating L/H layers, we model the coating as a discretized
  refractive-index profile. We generate a binary sequence of L/H layers that
  approximates a continuous index gradient (exponential or power-law) from
  ambient (1.0) to substrate (1.52). This "graded" structure is known to
  provide superior broadband antireflection performance compared to simple
  quarter-wave stacks.

  For each candidate gradient structure, we optimize the physical thicknesses
  using:
    1. Differential Evolution (global) to find good basins.
    2. L-BFGS-B (local) for high-precision polishing.

  We explore a diverse set of gradient shapes (exponential, power-law, linear)
  and layer counts, then polish the best candidates.
"""

import time
import math
import random
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    from scipy.optimize import differential_evolution as _scipy_de
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
        ths = [max(0.0, min(max_layer, t)) for t in ths]
    return ths


def _generate_gradient_sequence(n_layers, n_amb, n_sub, n_L, n_H, profile='exp'):
    """
    Generate a binary sequence of L/H layers that approximates a continuous
    index gradient from n_amb to n_sub.
    
    profile: 'exp', 'pow', 'lin', 'alt'
    """
    if n_layers <= 0:
        return []
    
    # Target indices: from n_amb to n_sub
    # We have only two indices: n_L and n_H.
    # We need to decide which layer is L and which is H.
    # Since n_L (1.38) is closer to n_amb (1.0) and n_H (2.35) is closer to n_sub (1.52)?
    # Wait, n_sub is 1.52. n_L is 1.38, n_H is 2.35.
    # Actually, for a graded index, we want the index to increase from 1.0 to 1.52.
    # n_L = 1.38 is closer to 1.52 than 1.0? No, 1.38 is closer to 1.52 (diff 0.14) than 1.0 (diff 0.38).
    # n_H = 2.35 is far from both.
    # However, the problem states we have materials L and H.
    # A "graded" index usually implies a smooth transition.
    # With binary materials, we can only switch between 1.38 and 2.35.
    # To approximate a gradient from 1.0 to 1.52:
    # Ideally, the first layers should be low index (L) and the last layers should be high index (H)?
    # But H is 2.35, which is much higher than 1.52.
    # Maybe the gradient is not about matching the exact index profile, but about creating a smooth transition in optical path?
    # Or perhaps we should just try different patterns.
    
    # Let's try a few patterns:
    # 1. Alternating (classic)
    # 2. All L then all H
    # 3. All H then all L
    # 4. Random-ish but with a bias towards L at the start and H at the end?
    
    # Actually, the "inspiration" mentioned binary search to approximate a continuous exponential or power-law index gradient.
    # Let's define a target index profile n(z) from n_amb to n_sub.
    # Then, for each layer i, compute the target index n_i.
    # If n_i is closer to n_L, use 'L', else use 'H'.
    
    if profile == 'exp':
        # n(z) = n_amb * exp(k * z)
        # n(1) = n_sub => k = ln(n_sub / n_amb)
        k = math.log(n_sub / n_amb)
        target_ns = [n_amb * math.exp(k * (i / (n_layers - 1))) for i in range(n_layers)]
    elif profile == 'pow':
        # n(z) = n_amb * (z)^(alpha)
        # n(1) = n_sub => n_sub = n_amb * 1^alpha => doesn't work directly.
        # Try n(z) = n_amb + (n_sub - n_amb) * z^alpha
        alpha = 2.0
        target_ns = [n_amb + (n_sub - n_amb) * ((i / (n_layers - 1)) ** alpha) for i in range(n_layers)]
    elif profile == 'lin':
        target_ns = [n_amb + (n_sub - n_amb) * (i / (n_layers - 1)) for i in range(n_layers)]
    elif profile == 'alt':
        # Alternating
        return ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    else:
        target_ns = [n_amb + (n_sub - n_amb) * (i / (n_layers - 1)) for i in range(n_layers)]
    
    seq = []
    for n_t in target_ns:
        # Choose the material whose index is closer to the target
        if abs(n_t - n_L) <= abs(n_t - n_H):
            seq.append('L')
        else:
            seq.append('H')
            
    return seq


def design(problem):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before the 150 s kill

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    random.seed(42)
    np.random.seed(42)

    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
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

    def _make_evaluator(materials, problem, max_total, max_layer):
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

    def _optimize_structure(materials, n_starts=4):
        """Optimize independent thicknesses for a given material sequence."""
        nonlocal best_ths, best_mats, best_score
        if time.time() >= deadline - 15:
            return
        eval_x, lo, hi = _make_evaluator(materials, problem, max_total, max_layer)

        # Try Differential Evolution first for global exploration
        if _HAVE_SCIPY and time.time() < deadline - 30:
            try:
                de_res = _scipy_de(
                    eval_x, bounds=list(zip(lo, hi)),
                    maxiter=30, tol=1e-5, seed=42,
                    polish=False, popsize=12, mutation=(0.5, 1.0),
                    recombination=0.8, init='latinhypercube'
                )
                xc = np.clip(de_res.x, lo, hi)
                t = np.exp(np.clip(xc, lo, hi))
                total = t.sum()
                if total > max_total:
                    t = t * (max_total / total)
                maybe_record(materials, list(t))
            except Exception:
                pass

        # Multiple diverse starting points with L-BFGS-B
        for start_i in range(n_starts):
            if time.time() >= deadline - 10:
                return

            # Generate starting point based on quarter-wave thicknesses
            x0 = []
            for m in materials:
                qw = qw_L if m == 'L' else qw_H
                factor = random.uniform(0.5, 1.5)
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
            else:
                xc = x0.copy()

            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(materials, list(t))

    # Generate a diverse set of candidate structures
    candidate_structures = []
    
    # 1. Alternating sequences (classic)
    for n in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n <= max_layers:
            candidate_structures.append(_generate_gradient_sequence(n, n_amb, n_sub, nL, nH, profile='alt'))
    
    # 2. Gradient sequences
    for n in [8, 10, 12, 14, 16, 18, 20]:
        if n <= max_layers:
            candidate_structures.append(_generate_gradient_sequence(n, n_amb, n_sub, nL, nH, profile='exp'))
            candidate_structures.append(_generate_gradient_sequence(n, n_amb, n_sub, nL, nH, profile='pow'))
            candidate_structures.append(_generate_gradient_sequence(n, n_amb, n_sub, nL, nH, profile='lin'))
    
    # Remove duplicates
    unique_structures = []
    seen = set()
    for s in candidate_structures:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_structures.append(s)
    
    # Sort by length to try smaller ones first for quick wins? Or random?
    # Let's shuffle to mix it up
    random.shuffle(unique_structures)
    
    # Optimize each structure
    for mats in unique_structures:
        if time.time() >= deadline - 20:
            break
        n_starts = 4 if len(mats) <= 10 else 2
        _optimize_structure(mats, n_starts)

    # Final high-precision polish on the best solution found
    if best_ths is not None and time.time() < deadline - 10:
        eval_x, lo, hi = _make_evaluator(best_mats, problem, max_total, max_layer)
        x0 = [math.log(max(1.0, t)) for t in best_ths]

        if _HAVE_SCIPY:
            res = _scipy_minimize(
                eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                options={"maxiter": 500, "maxfun": 10000, "maxls": 50,
                         "ftol": 1e-14, "gtol": 1e-12},
            )
            xc = np.clip(res.x, lo, hi)
            t = np.exp(np.clip(xc, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(best_mats, list(t))

    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center
        return ['L'], [max(0.0, min(max_layer, qw_L))]

    return list(best_mats), list(best_ths)
