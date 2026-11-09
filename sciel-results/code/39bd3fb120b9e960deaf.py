"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

STRUCTURAL CHANGE: Optimize layer SEQUENCE (L/H assignment) via continuous relaxation
  1. Exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Fixed 20 layers with uniform thickness 45 nm (total 900 nm).
  3. Continuous relaxation: each layer has a soft assignment z_j in [0,1]
     (0 = L, 1 = H). Effective index n_j = n_L + z_j * (n_H - n_L).
  4. Gradient descent on the smooth surrogate to find good binary sequences.
  5. Local refinement of thicknesses around the best binary sequence.
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


def _reflectance_with_indices(ns, ds, wavs, n_0, n_s):
    """Compute mean reflectance given index array ns and thickness array ds."""
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


def _surrogate_objective(z, ds, wavs, n_0, n_s, n_L, n_H):
    """
    Smooth surrogate: z_j in [0,1] gives effective index n_j = n_L + z_j*(n_H - n_L).
    Returns mean reflectance for the continuous-index stack.
    """
    ns = n_L + z * (n_H - n_L)
    return _reflectance_with_indices(ns, ds, wavs, n_0, n_s)


def _gradient_descent_sequence(z0, ds, wavs, n_0, n_s, n_L, n_H, 
                                max_iter=2000, lr_init=0.1, lr_decay=0.999):
    """
    Gradient descent on continuous relaxation z to find good binary sequences.
    Uses numerical gradient (central differences) since analytic grad is complex.
    Returns best z and its binary assignment.
    """
    n = len(z)
    z = np.clip(z0.copy(), 0.0, 1.0)
    lr = lr_init
    best_z = z.copy()
    best_val = _surrogate_objective(z, ds, wavs, n_0, n_s, n_L, n_H)
    
    for it in range(max_iter):
        # Numerical gradient via central differences
        eps = 1e-4
        grad = np.zeros(n)
        for j in range(n):
            z_plus = z.copy()
            z_plus[j] = min(1.0, z[j] + eps)
            z_minus = z.copy()
            z_minus[j] = max(0.0, z[j] - eps)
            fp = _surrogate_objective(z_plus, ds, wavs, n_0, n_s, n_L, n_H)
            fm = _surrogate_objective(z_minus, ds, wavs, n_0, n_s, n_L, n_H)
            grad[j] = (fp - fm) / (2.0 * eps)
        
        # Gradient step with clipping
        z_new = np.clip(z - lr * grad, 0.0, 1.0)
        val_new = _surrogate_objective(z_new, ds, wavs, n_0, n_s, n_L, n_H)
        
        if val_new < best_val:
            best_val = val_new
            best_z = z_new.copy()
        
        z = z_new
        lr *= lr_decay
        
        # Check convergence
        if it > 50 and lr < 1e-6:
            break
    
    # Binary assignment: round to nearest
    z_binary = (best_z > 0.5).astype(float)
    return best_z, z_binary, best_val


def _evaluate_binary_sequence(binary, ds, wavs, n_0, n_s, n_L, n_H, problem):
    """Convert binary sequence to L/H materials and compute exact reflectance."""
    materials = ['H' if b > 0.5 else 'L' for b in binary]
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    return _reflectance_with_indices(ns, ds, wavs, n_0, n_s), materials


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
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    wavs = np.linspace(band[0], band[1], 401)

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

    n_layers = min(20, max_layers)
    ds_uniform = np.full(n_layers, max_total / n_layers)  # 50 nm each for 20 layers
    
    # Phase 1: Optimize layer sequence via continuous relaxation
    # Try multiple random initializations of z
    n_starts = 12
    best_binary = None
    best_binary_val = float("inf")
    
    for start_i in range(n_starts):
        if time.time() >= deadline - 40:
            break
        
        if start_i == 0:
            # Start with alternating pattern
            z0 = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n_layers)])
        elif start_i == 1:
            # Start with graded index (smooth transition)
            z0 = np.linspace(0.0, 1.0, n_layers)
        elif start_i == 2:
            # Start with inverse graded
            z0 = np.linspace(1.0, 0.0, n_layers)
        else:
            # Random start
            z0 = np.random.uniform(0.0, 1.0, n_layers)
        
        # Run gradient descent with reduced iterations to save time
        max_iter_gd = 500 if start_i == 0 else 300
        lr_init = 0.05
        z_best, z_bin, val = _gradient_descent_sequence(
            z0, ds_uniform, wavs, n_0, n_s, nL, nH,
            max_iter=max_iter_gd, lr_init=lr_init
        )
        
        if val < best_binary_val:
            best_binary_val = val
            best_binary = z_bin.copy()
        
        # Also try the rounded version with uniform thickness
        mats_bin = ['H' if b > 0.5 else 'L' for b in z_bin]
        n_mat = {k: v for k, v in problem["materials"].items()}
        ns_bin = np.array([n_mat[m] for m in mats_bin], dtype=float)
        val_exact = _reflectance_with_indices(ns_bin, ds_uniform, wavs, n_0, n_s)
        maybe_record(mats_bin, list(ds_uniform))
    
    # Phase 2: Local refinement of thicknesses for the best binary sequence
    if best_binary is not None and time.time() < deadline - 15:
        mats_best = ['H' if b > 0.5 else 'L' for b in best_binary]
        n_mat = {k: v for k, v in problem["materials"].items()}
        ns_best = np.array([n_mat[m] for m in mats_best], dtype=float)
        
        # Optimize thicknesses using L-BFGS-B in log-space
        n = n_layers
        lo = np.full(n, math.log(1.0))
        hi = np.full(n, math.log(max_layer))
        
        def eval_thick(x):
            t = np.exp(np.clip(x, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            return _reflectance_with_indices(ns_best, t, wavs, n_0, n_s)
        
        # Start from uniform thickness
        x0 = np.full(n, math.log(ds_uniform[0]))
        
        if _HAVE_SCIPY:
            bounds = list(zip(lo, hi))
            res = _scipy_minimize(
                eval_thick, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 500, "maxfun": 8000, "maxls": 50,
                         "ftol": 1e-12, "gtol": 1e-10},
            )
            xc = np.clip(res.x, lo, hi)
            t = np.exp(xc)
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(mats_best, list(t))
        
        # Try a few other thickness starting points
        for trial in range(3):
            if time.time() >= deadline - 5:
                break
            x0_trial = np.array([random.uniform(lo[i], hi[i]) for i in range(n)])
            if _HAVE_SCIPY:
                bounds = list(zip(lo, hi))
                res = _scipy_minimize(
                    eval_thick, x0_trial, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 300, "maxfun": 5000, "maxls": 40},
                )
                xc = np.clip(res.x, lo, hi)
                t = np.exp(xc)
                total = t.sum()
                if total > max_total:
                    t = t * (max_total / total)
                maybe_record(mats_best, list(t))
    
    # Phase 3: Also try the classic alternating L/H structure with optimized thicknesses
    if time.time() < deadline - 10:
        materials_alt = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        n_mat = {k: v for k, v in problem["materials"].items()}
        ns_alt = np.array([n_mat[m] for m in materials_alt], dtype=float)
        
        n = n_layers
        lo = np.full(n, math.log(1.0))
        hi = np.full(n, math.log(max_layer))
        
        def eval_alt(x):
            t = np.exp(np.clip(x, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            return _reflectance_with_indices(ns_alt, t, wavs, n_0, n_s)
        
        # Quarter-wave reference
        qw_L = wl_center / (4.0 * nL)
        qw_H = wl_center / (4.0 * nH)
        qw_ref = np.array([qw_L if m == 'L' else qw_H for m in materials_alt])
        x0_qw = np.log(np.clip(qw_ref, 1.0, max_layer))
        
        if _HAVE_SCIPY:
            bounds = list(zip(lo, hi))
            res = _scipy_minimize(
                eval_alt, x0_qw, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 500, "maxfun": 8000, "maxls": 50,
                         "ftol": 1e-12, "gtol": 1e-10},
            )
            xc = np.clip(res.x, lo, hi)
            t = np.exp(xc)
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            maybe_record(materials_alt, list(t))
    
    # Phase 4: Final polish with Nelder-Mead if time allows
    if best_ths is not None and time.time() < deadline - 4:
        n = len(best_mats)
        lo = np.full(n, math.log(1.0))
        hi = np.full(n, math.log(max_layer))
        
        def eval_final(x):
            t = np.exp(np.clip(x, lo, hi))
            total = t.sum()
            if total > max_total:
                t = t * (max_total / total)
            mats = best_mats
            n_mat = {k: v for k, v in problem["materials"].items()}
            ns = np.array([n_mat[m] for m in mats], dtype=float)
            return _reflectance_with_indices(ns, t, wavs, n_0, n_s)
        
        x0_nm = [math.log(max(1.0, t)) for t in best_ths]
        
        # Simple Nelder-Mead
        simplex = [x0_nm[:]]
        for i in range(n):
            pt = list(x0_nm)
            pt[i] = hi[i] if pt[i] == lo[i] else lo[i] + 0.2 * (hi[i] - lo[i])
            simplex.append(pt)
        scores = [eval_final(pt) for pt in simplex]
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        it = 0
        while it < 400 and time.time() < deadline - 2:
            it += 1
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
            fr = eval_final(list(xr))
            if fr < scores[0]:
                xe = np.clip(centroid + gamma * (centroid - simplex[-1]), lo, hi)
                fe = eval_final(list(xe))
                if fe < fr:
                    simplex[-1], scores[-1] = list(xe), fe
                else:
                    simplex[-1], scores[-1] = list(xr), fr
            elif fr < scores[-2]:
                simplex[-1], scores[-1] = list(xr), fr
            else:
                xc = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
                fc = eval_final(list(xc))
                if fc <= fr:
                    simplex[-1], scores[-1] = list(xc), fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = np.clip(
                            np.array(simplex[0]) + sigma * (np.array(simplex[i]) - np.array(simplex[0])),
                            lo, hi)
                        scores[i] = eval_final(list(simplex[i]))
                    if max(scores) - min(scores) < 1e-9:
                        break
        
        # Record best from NM
        best_idx = np.argmin(scores)
        t_nm = np.exp(np.clip(simplex[best_idx], lo, hi))
        total_nm = t_nm.sum()
        if total_nm > max_total:
            t_nm = t_nm * (max_total / total_nm)
        maybe_record(best_mats, list(t_nm))

    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center.
        qw_L = wl_center / (4.0 * nL)
        return ['L'], [max(0.0, min(max_layer, qw_L))]

    return list(best_mats), list(best_ths)
