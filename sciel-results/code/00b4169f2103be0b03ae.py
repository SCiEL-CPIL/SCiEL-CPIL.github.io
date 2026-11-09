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
        if j == 0:
            A_l, B_l, C_l, D_l = np.ones_like(wavs), np.zeros_like(wavs), np.zeros_like(wavs), np.ones_like(wavs)
        else:
            A_l, B_l, C_l, D_l = prefix[j-1]
            
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


def design(problem):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    random.seed(2024)
    np.random.seed(2024)
    
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    
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
    
    def polish_softmax(materials, t_init, problem, T_target):
        """Polish using L-BFGS-B with softmax parameterization to enforce sum=T_target."""
        if time.time() > deadline:
            return None
            
        n = len(materials)
        # Convert t_init to x_init
        # t_i = exp(x_i) / sum(exp(x_j)) * T_target
        # x_i = log(t_i) - log(T_target) + log(sum(t_j))  (approx)
        # Actually, just use log(t_i) as initial guess, it will be normalized
        t_safe = np.maximum(t_init, 1e-6)
        x0 = np.log(t_safe)
        
        band = problem["band_nm"]
        wavs = np.linspace(band[0], band[1], 401)
        n_mat = {k: v for k, v in problem["materials"].items()}
        ns = [n_mat[m] for m in materials]
        
        def obj_x(x):
            ex = np.exp(x - np.max(x)) # for stability
            S = np.sum(ex)
            t = ex / S * T_target
            # Check bounds
            if np.any(t < 0) or np.any(t > max_layer):
                # Soft penalty or clip? Clip for now, gradient might be off but L-BFGS handles bounds?
                # Actually, softmax doesn't guarantee t < max_layer.
                # We should add a penalty if any t > max_layer
                if np.any(t > max_layer):
                    return _mean_reflectance(materials, list(t), problem) + 100 * (np.max(t) - max_layer)
            return _mean_reflectance(materials, list(t), problem)
            
        def grad_x(x):
            ex = np.exp(x - np.max(x))
            S = np.sum(ex)
            t = ex / S * T_target
            
            g_phys = _gradient_reflectance(materials, list(t), problem, wavs)
            
            # Chain rule: dR/dx_k = sum_i dR/dt_i * dt_i/dx_k
            # dt_i/dx_k = t_i * (delta_ik - t_k/T_target)
            grad = np.zeros(n)
            for k in range(n):
                # dR/dx_k = g_phys_k * t_k * (1 - t_k/T) - sum_{i != k} g_phys_i * t_i * t_k/T
                # = t_k * (g_phys_k - sum_i g_phys_i * t_i / T)
                sum_gt = np.sum(g_phys * t) / T_target
                grad[k] = t[k] * (g_phys[k] - sum_gt)
                
            # Penalty gradient if t > max_layer
            if np.any(t > max_layer):
                # This is a heuristic, might not be perfect gradient but helps
                for k in range(n):
                    if t[k] > max_layer:
                        grad[k] += 100
            return grad
        
        try:
            res = _scipy_minimize(
                obj_x, x0, jac=grad_x, method="L-BFGS-B",
                options={"maxiter": 100, "maxfun": 1000},
            )
            ex = np.exp(res.x - np.max(res.x))
            S = np.sum(ex)
            t = ex / S * T_target
            t = np.clip(t, 0, max_layer)
            # Renormalize if clipping changed sum significantly?
            # If we clip, sum might be < T_target.
            # If sum < T_target, it's still feasible (sum <= max_total).
            maybe_record(materials, list(t))
            return list(t)
        except Exception:
            return None

    def run_de(materials, problem, deadline):
        n = len(materials)
        bounds = [(1.0, max_layer) for _ in range(n)]
        
        def objective(x):
            t = list(x)
            total = sum(t)
            if total > max_total:
                # Penalty
                return _mean_reflectance(materials, t, problem) + 10 * (total - max_total)
            return _mean_reflectance(materials, t, problem)
        
        try:
            res = _de(
                objective, bounds,
                popsize=15,
                maxiter=30,
                seed=2024,
                polish=False, # We do our own polish
                init='latinhypercube'
            )
            if res.success or res.fun < float('inf'):
                t = list(res.x)
                total = sum(t)
                if total > max_total:
                    t = [ti * max_total / total for ti in t]
                maybe_record(materials, t)
                return list(t)
        except Exception:
            pass
        return None

    # Define patterns
    patterns = []
    # Alternating L/H
    for N in [4, 6, 8]:
        if N <= max_layers:
            mats = ['L' if i % 2 == 0 else 'H' for i in range(N)]
            patterns.append((N, mats))
            
    # Try H first
    for N in [4, 6, 8]:
        if N <= max_layers:
            mats = ['H' if i % 2 == 0 else 'L' for i in range(N)]
            patterns.append((N, mats))

    # Run DE for each pattern
    candidates = []
    for N, mats in patterns:
        if time.time() > deadline - 30:
            break
        t_best = run_de(mats, problem, deadline)
        if t_best:
            candidates.append((mats, t_best))

    # Polish candidates with different total thickness targets
    T_targets = [1000, 800, 600, 400]
    for mats, t_init in candidates:
        for T in T_targets:
            if time.time() > deadline - 5:
                break
            polish_softmax(mats, t_init, problem, T)
            
    # If no DE candidates, do random restarts with L-BFGS-B (fallback)
    if not candidates:
        # Simple random restarts
        for N in [4, 6, 8]:
            if time.time() > deadline - 5:
                break
            mats = ['L' if i % 2 == 0 else 'H' for i in range(N)]
            for _ in range(5):
                t = [random.uniform(50, 200) for _ in range(N)]
                total = sum(t)
                if total > max_total:
                    t = [ti * max_total / total for ti in t]
                polish_softmax(mats, t, problem, 1000)

    if best_ths is None:
        # Fallback: single quarter-wave L layer
        qw_L = 600 / (4.0 * nL)
        return ['L'], [max(0.0, min(max_layer, qw_L))]
        
    return list(best_mats), list(best_ths)
