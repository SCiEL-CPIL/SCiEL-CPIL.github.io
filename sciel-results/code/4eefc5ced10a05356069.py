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


def _reflectance_from_half(half_materials, half_thicknesses, problem):
    """Calculate reflectance for a symmetric stack given the first half."""
    # Construct full stack
    materials = half_materials + half_materials[::-1]
    thicknesses = half_thicknesses + half_thicknesses[::-1]
    return _mean_reflectance(materials, thicknesses, problem)


def design(problem):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    random.seed(2024)
    np.random.seed(2024)

    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    
    # We will use a symmetric 20-layer stack.
    # This means we optimize 10 layers and mirror them.
    # Total layers = 20.
    # Max total thickness = 1000 nm.
    # So sum of first 10 layers <= 500 nm.
    
    # Define the material sequence for the first 10 layers.
    # Alternating L/H is a good starting point.
    half_n = 10
    half_materials = ['L' if i % 2 == 0 else 'H' for i in range(half_n)]
    
    # Quarter-wave thicknesses at center wavelength
    qw_L = wl_center / (4.0 * nL)
    qw_H = wl_center / (4.0 * nH)
    
    def objective(x):
        # x is log-thicknesses of the first 10 layers
        # Convert to physical thicknesses
        t_half = np.exp(np.clip(x, math.log(1.0), math.log(max_layer)))
        
        # Check total thickness constraint
        # Sum of first 10 layers must be <= max_total / 2
        sum_half = np.sum(t_half)
        if sum_half > max_total / 2.0:
            # Scale down to satisfy constraint
            t_half = t_half * (max_total / 2.0) / sum_half
        
        # Calculate reflectance
        return _reflectance_from_half(half_materials, t_half, problem)

    best_score = float("inf")
    best_ths_half = None
    
    # Multi-start optimization
    n_starts = 10
    for i in range(n_starts):
        if time.time() >= deadline:
            break
            
        # Initialize x (log-thicknesses)
        x0 = np.zeros(half_n)
        for j in range(half_n):
            if half_materials[j] == 'L':
                t_init = qw_L
            else:
                t_init = qw_H
            # Add random perturbation
            t_init *= random.uniform(0.5, 1.5)
            t_init = max(1.0, min(max_layer, t_init))
            x0[j] = math.log(t_init)
            
        if _HAVE_SCIPY:
            res = _scipy_minimize(
                objective, x0, method="L-BFGS-B", 
                bounds=[(math.log(1.0), math.log(max_layer))] * half_n,
                options={"maxiter": 1000, "maxfun": 10000, "ftol": 1e-10, "gtol": 1e-8}
            )
            x_opt = res.x
        else:
            # Fallback: simple Nelder-Mead
            x_opt = x0
            # Simple local search
            step = 0.1
            for _ in range(100):
                improved = False
                for j in range(half_n):
                    for delta in [step, -step]:
                        x_try = x_opt.copy()
                        x_try[j] += delta
                        if objective(x_try) < objective(x_opt):
                            x_opt = x_try
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
        
        # Evaluate final result
        t_half = np.exp(np.clip(x_opt, math.log(1.0), math.log(max_layer)))
        sum_half = np.sum(t_half)
        if sum_half > max_total / 2.0:
            t_half = t_half * (max_total / 2.0) / sum_half
        
        score = _reflectance_from_half(half_materials, t_half, problem)
        
        if score < best_score:
            best_score = score
            best_ths_half = t_half.copy()
            
    # Construct final thicknesses
    if best_ths_half is None:
        # Fallback: single layer
        return ['L'], [qw_L]
    
    full_ths = list(best_ths_half) + list(best_ths_half[::-1])
    full_mats = half_materials + half_materials[::-1]
    
    # Final check and scaling if necessary (should be already satisfied)
    total = sum(full_ths)
    if total > max_total:
        scale = max_total / total
        full_ths = [t * scale for t in full_ths]
        
    # Clip individual layers
    full_ths = [max(0.0, min(max_layer, t)) for t in full_ths]
    
    return full_mats, full_ths
