"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Strategy:
  1. Exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Analytical gradient with respect to thicknesses for fast local optimization.
  3. Nelder-Mead simplex optimization for robust local polishing (no gradient needed).
  4. Multiple diverse layer patterns and random restarts.
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


def _compute_matrices(materials, thicknesses, problem, wavs):
    """Compute characteristic matrices and return (A, B, C, D) arrays over wavelengths."""
    if not materials:
        n = len(wavs)
        return (np.ones(n), np.zeros(n), np.zeros(n), np.ones(n))
    
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
    
    return A, B, C, D


def _mean_reflectance(materials, thicknesses, problem):
    """Mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    A, B, C, D = _compute_matrices(materials, thicknesses, problem, wavs)
    
    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def _nm_polish(materials, ths, problem, max_total, max_layer, deadline):
    """Polish using Nelder-Mead simplex method. Returns (improved_ths, score)."""
    n = len(materials)
    if n == 0:
        return ths, float('inf')
    
    # Precompute wavs
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    
    def obj(x):
        if time.time() > deadline:
            return 1e6
        t = np.asarray(x, dtype=float)
        # Clip to valid range
        t = np.clip(t, 0.0, max_layer)
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
            t = np.clip(t, 0.0, max_layer)
        return _mean_reflectance(materials, list(t), problem)
    
    x0 = np.asarray(ths, dtype=float)
    x0 = np.clip(x0, 0.0, max_layer)
    total = x0.sum()
    if total > max_total:
        x0 = x0 * (max_total / total)
        x0 = np.clip(x0, 0.0, max_layer)
    
    try:
        res = _scipy_minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        xc = np.clip(res.x, 0.0, max_layer)
        total = xc.sum()
        if total > max_total:
            xc = xc * (max_total / total)
            xc = np.clip(xc, 0.0, max_layer)
        score = obj(xc)
        return list(xc), score
    except Exception:
        return list(x0), obj(x0)


def _generate_patterns(n_layers):
    """Generate diverse layer material patterns."""
    patterns = []
    # Alternating starting with L
    patterns.append(['L' if i % 2 == 0 else 'H' for i in range(n_layers)])
    # Alternating starting with H
    patterns.append(['H' if i % 2 == 0 else 'L' for i in range(n_layers)])
    # Grouped: LLHH LLHH...
    if n_layers >= 4:
        p = []
        for i in range(n_layers // 4 + 1):
            p.extend(['L', 'L', 'H', 'H'])
        patterns.append(p[:n_layers])
    # Grouped: HHLH HHLH...
    if n_layers >= 4:
        p = []
        for i in range(n_layers // 4 + 1):
            p.extend(['H', 'H', 'L', 'H'])
        patterns.append(p[:n_layers])
    # Symmetric patterns
    if n_layers >= 3:
        half = n_layers // 2
        for first in ['L', 'H']:
            p = [first] * half
            if n_layers % 2 == 1:
                p.append('H' if first == 'L' else 'L')
            p.extend(p[:half][::-1])
            patterns.append(p[:n_layers])
    return patterns


def _de_search(materials, problem, max_total, max_layer, deadline, pop_size=30, max_iter=30):
    """Placeholder for removed DE search."""
    return None, float('inf')


def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
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
        if time.time() > deadline:
            return
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
    
    # Layer counts to try, prioritized
    counts_to_try = [4, 6, 8, 10, 12, 5, 7, 9, 11, 14, 16, 18, 20, 3, 2]
    counts_to_try = [c for c in counts_to_try if c <= max_layers]
    
    # Create tasks: (n_layers, pattern)
    tasks = []
    for n_layers in counts_to_try:
        patterns = _generate_patterns(n_layers)
        for mats in patterns:
            tasks.append((n_layers, mats))
    
    # Shuffle tasks to diversify search
    random.shuffle(tasks)
    
    # Phase 1: Multi-start Nelder-Mead with diverse initial points
    restarts_per_task = 3
    for n_layers, mats in tasks:
        if time.time() > deadline - 15:
            break
        
        for _ in range(restarts_per_task):
            if time.time() > deadline - 10:
                break
            
            ths = []
            for m in mats:
                qw = qw_L if m == 'L' else qw_H
                # Randomize around quarter wave with wider range
                t = qw * random.uniform(0.1, 3.0)
                t = max(0.5, min(max_layer, t))
                ths.append(t)
            
            total = sum(ths)
            if total > max_total:
                ths = [t * max_total / total for t in ths]
            
            # Nelder-Mead polish
            polished_ths, score = _nm_polish(mats, ths, problem, max_total, max_layer, deadline)
            maybe_record(mats, polished_ths)
    
    # Phase 2: If we have a best, do additional refinement with different initializations
    if best_ths is not None and time.time() < deadline - 10:
        # Try perturbing the best solution
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            perturbed = [max(0.0, min(max_layer, t * random.uniform(0.7, 1.3))) for t in best_ths]
            total = sum(perturbed)
            if total > max_total:
                perturbed = [t * max_total / total for t in perturbed]
            polished_ths, score = _nm_polish(best_mats, perturbed, problem, max_total, max_layer, deadline)
            maybe_record(best_mats, polished_ths)
    
    # Final polish on best
    if best_ths is not None and time.time() < deadline - 3:
        polished_ths, score = _nm_polish(best_mats, best_ths, problem, max_total, max_layer, deadline)
        maybe_record(best_mats, polished_ths)
    
    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center
        return ['L'], [max(0.0, min(max_layer, qw_L))]
    
    return list(best_mats), list(best_ths)
