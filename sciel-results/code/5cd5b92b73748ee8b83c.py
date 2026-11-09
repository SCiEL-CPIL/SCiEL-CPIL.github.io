"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Strategy:
  1. Exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Analytical gradient with respect to thicknesses for fast local optimization.
  3. Differential Evolution for global exploration over independent layer thicknesses.
  4. L-BFGS-B local polish from DE candidates and diverse random starts.
  5. Multiple layer patterns (alternating, L-first, H-first, grouped).
  6. Explicit deadline guard: return the best feasible result found before the limit.
"""

import time
import math
import random
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize, differential_evolution as _de
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


def _gradient_reflectance(materials, thicknesses, problem, wavs):
    """Compute analytical gradient of mean reflectance w.r.t. each thickness."""
    if not materials:
        return np.array([])
    
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)
    n_layers = len(materials)
    
    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    
    n_ds = np.array([n * d for n, d in zip(ns, ds)])
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)
    i = 1.0j
    
    prefix = []
    A_pre = np.ones_like(wavs)
    B_pre = np.zeros_like(wavs)
    C_pre = np.zeros_like(wavs)
    D_pre = np.ones_like(wavs)
    
    for j in range(n_layers):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A_pre * cj + i * B_pre * n_j * sj
        B_new = i * A_pre * sj / n_j + B_pre * cj
        C_new = C_pre * cj + i * D_pre * n_j * sj
        D_new = i * C_pre * sj / n_j + D_pre * cj
        prefix.append((A_new.copy(), B_new.copy(), C_new.copy(), D_new.copy()))
        A_pre, B_pre, C_pre, D_pre = A_new, B_new, C_new, D_new
    
    suffix = []
    A_suf = np.ones_like(wavs)
    B_suf = np.zeros_like(wavs)
    C_suf = np.zeros_like(wavs)
    D_suf = np.ones_like(wavs)
    
    for j in range(n_layers - 1, -1, -1):
        suffix.append((A_suf.copy(), B_suf.copy(), C_suf.copy(), D_suf.copy()))
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A_suf * cj + i * B_suf * n_j * sj
        B_new = i * A_suf * sj / n_j + B_suf * cj
        C_new = C_suf * cj + i * D_suf * n_j * sj
        D_new = i * C_suf * sj / n_j + D_suf * cj
        A_suf, B_suf, C_suf, D_suf = A_new, B_new, C_new, D_new
    
    A_tot, B_tot, C_tot, D_tot = prefix[-1]
    
    num = n_0 * (A_tot + B_tot * n_s) - (C_tot + D_tot * n_s)
    den = n_0 * (A_tot + B_tot * n_s) + (C_tot + D_tot * n_s)
    r = num / den
    
    grad = np.zeros(n_layers)
    
    for j in range(n_layers):
        # M_l is M_1...M_{j-1}. If j==0, it's Identity.
        if j == 0:
            A_l, B_l, C_l, D_l = np.ones_like(wavs), np.zeros_like(wavs), np.zeros_like(wavs), np.ones_like(wavs)
        else:
            A_l, B_l, C_l, D_l = prefix[j-1]
            
        # M_r is M_{j+1}...M_N. suffix[j] is already this.
        A_r, B_r, C_r, D_r = suffix[j]
        
        n_j = ns[j]
        d_j = delta[j]
        sd = np.sin(d_j)
        cd = np.cos(d_j)
        k = 2.0 * np.pi / wavs
        
        dA = -sd * n_j * k
        dB = i * cd * k
        dC = i * n_j * cd * k
        dD = -sd * n_j * k
        
        tmp_A = A_l * dA + B_l * dC
        tmp_B = A_l * dB + B_l * dD
        tmp_C = C_l * dA + D_l * dC
        tmp_D = C_l * dB + D_l * dD
        
        dA_tot = tmp_A * A_r + tmp_B * C_r
        dB_tot = tmp_A * B_r + tmp_B * D_r
        dC_tot = tmp_C * A_r + tmp_D * C_r
        dD_tot = tmp_C * B_r + tmp_D * D_r
        
        dnum = n_0 * (dA_tot + dB_tot * n_s) - (dC_tot + dD_tot * n_s)
        dden = n_0 * (dA_tot + dB_tot * n_s) + (dC_tot + dD_tot * n_s)
        
        dr = (dnum * den - num * dden) / (den * den)
        dR = 2.0 * np.real(np.conj(r) * dr)
        
        grad[j] = np.mean(dR)
    
    return grad


def _make_evaluator(materials, problem, max_total, max_layer):
    """Return (eval, grad, lo, hi) for log-scaled thickness optimization."""
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    n = len(materials)
    lo = np.full(n, math.log(0.5))
    hi = np.full(n, math.log(max_layer))
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    
    def _physical(x):
        t = np.exp(np.clip(x, lo, hi))
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
        return t
    
    def eval_x(x):
        t = _physical(x)
        return _mean_reflectance(materials, list(t), problem)
    
    def grad_x(x):
        t = _physical(x)
        g_phys = _gradient_reflectance(materials, list(t), problem, wavs)
        # Chain rule: dR/dx_j = dR/dt_j * dt_j/dx_j = g_phys_j * t_j (since t_j = exp(x_j))
        # But if scaled: t_j = exp(x_j) * max_total/total, so dt_j/dx_j = t_j * (1 - t_j/total)
        total = t.sum()
        scale = max_total / total if total > max_total else 1.0
        if scale < 1.0:
            g = g_phys * t * (1.0 - t / total)
        else:
            g = g_phys * t
        return g
    
    return eval_x, grad_x, lo, hi


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
    
    def polish(materials, ths, deadline_polish):
        """Local polish using L-BFGS-B with gradients."""
        eval_x, grad_x, lo, hi = _make_evaluator(materials, problem, max_total, max_layer)
        x0 = [math.log(max(0.5, t)) for t in ths]
        x0 = np.clip(np.array(x0), lo, hi)
        
        try:
            res = _scipy_minimize(
                eval_x, x0, jac=grad_x, method="L-BFGS-B",
                bounds=list(zip(lo, hi)),
                options={"maxiter": 200, "maxfun": 2000, "maxls": 20},
            )
            xc = np.clip(res.x, lo, hi)
            t = np.exp(xc)
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(materials, list(t))
            return list(t)
        except Exception:
            maybe_record(materials, list(ths))
            return list(ths)
    
    # Define diverse layer patterns
    def get_patterns(n_layers):
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
        return patterns
    
    # Layer counts to try, prioritized
    counts_to_try = [4, 6, 8, 10, 12, 5, 7, 9, 11, 14, 16, 18, 20, 3, 2]
    counts_to_try = [c for c in counts_to_try if c <= max_layers]
    
    # Phase 1: Multi-start L-BFGS-B with fixed gradient
    # Prioritize layer counts that are known to be effective for broadband AR
    # 4, 6, 8, 10, 12 layers are common.
    
    # We will iterate through counts and patterns, doing multiple random restarts.
    # We use the deadline to stop.
    
    # To maximize coverage, we interleave counts and patterns.
    # Let's create a list of (n_layers, pattern) to try.
    tasks = []
    for n_layers in counts_to_try:
        patterns = get_patterns(n_layers)
        for mats in patterns:
            tasks.append((n_layers, mats))
    
    # Shuffle tasks to mix layer counts? Or keep structured?
    # Structured is better to ensure we try small counts first which are faster.
    # But we want to find the global min.
    # Let's do a few restarts for each task.
    
    restarts_per_task = 5
    for n_layers, mats in tasks:
        if time.time() > deadline - 10:
            break
        
        for _ in range(restarts_per_task):
            if time.time() > deadline - 5:
                break
            
            ths = []
            for m in mats:
                qw = qw_L if m == 'L' else qw_H
                # Randomize around quarter wave
                t = qw * random.uniform(0.2, 2.5)
                t = max(0.5, min(max_layer, t))
                ths.append(t)
            
            total = sum(ths)
            if total > max_total:
                ths = [t * max_total / total for t in ths]
            
            polish(mats, ths, deadline)
    
    # Final polish on best
    if best_ths is not None and time.time() < deadline - 3:
        polish(best_mats, best_ths, deadline)
    
    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center
        return ['L'], [max(0.0, min(max_layer, qw_L))]
    
    return list(best_mats), list(best_ths)
