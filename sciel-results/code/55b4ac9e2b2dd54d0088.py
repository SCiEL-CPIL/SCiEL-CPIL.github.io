"""Broadband antireflection coating: 20-layer symmetric graded structure with simulated annealing."""
import time
import math
import random
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)

    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    K = wavs.shape[0]
    A = np.ones(K)
    B = np.zeros(K)
    C = np.zeros(K)
    D = np.ones(K)
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
    den = n_0 * (A + B * n_s) + (C + n_s)
    r = num / den
    R = np.abs(r) ** 2
    return float(100.0 * np.mean(R))

def design(problem):
    start_time = time.time()
    deadline = start_time + 135  # 150s kill, 15s safety margin

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]

    # Build symmetric 20-layer graded pattern: L L L L L L L L L L H H H H H H H H H H
    # Actually for graded index from air (1.0) to substrate (1.52), we want increasing index
    # Symmetric: first 10 layers L, last 10 layers H? No, that's a step.
    # Better: symmetric pattern that creates a smooth-ish transition.
    # Use: L L L L L H H H H H H H H H H H L L L L L -> not symmetric.
    # True symmetric: mirror the first 10. 
    # Pattern: L L L L L L H H H H H H L L L L L L? No, 20 layers.
    # Let's use a pattern that is palindromic:
    # L L L L L H H H H H H H H H H L L L L L is 20 layers? 
    # 5 L, 10 H, 5 L = 20. This is symmetric.
    # Or: L L L L H H H H H H H H H H H H L L L L (4 L, 12 H, 4 L)
    # Or alternating? No, graded means monotonic.
    # Let's use: L L L L L L L L L L H H H H H H H H H H (10 L, 10 H) - this is not symmetric in the sense of palindromic, but it is a step.
    # A "graded" structure usually implies a smooth change. With only two materials, we can't make a smooth gradient, but we can make a symmetric stack.
    # Symmetric stack: L L L L L L H H H H H H H H H H L L L L L L is 22 layers.
    # 20 layers: L L L L L H H H H H H H H H H L L L L L (5 L, 10 H, 5 L)
    # This is symmetric. The index profile is Low-Mid(High)-Low. This is a "box" filter in index space.
    # Alternatively: L L L L H H H H H H H H H H H H L L L L (4 L, 12 H, 4 L)
    # Let's try: L L L L L L L L H H H H H H H H L L L L L L (8 L, 8 H, 4 L? No, 8+8+4=20, not symmetric)
    # Symmetric 20 layers: first 10 determine the second 10 (reversed).
    # Let's use a pattern that starts with L and ends with L, with H in the middle.
    # Pattern: L L L L L H H H H H H H H H H L L L L L
    # Indices: 1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1
    # This is symmetric.
    
    # Actually, for broadband AR, a symmetric stack is often good.
    # Let's define the first 10 layers and mirror them.
    # To get a graded effect, we should alternate or group.
    # Let's use a fixed pattern: 
    # L L L L L H H H H H (first half)
    # Mirror: H H H H H L L L L L (second half)
    # Full: L L L L L H H H H H H H H H H L L L L L
    # This is 20 layers.
    
    pattern = ["L"] * 5 + ["H"] * 10 + ["L"] * 5
    n_layers = len(pattern)
    
    # Initial thicknesses: quarter-wave at center wavelength
    # d = lambda / (4 * n)
    init_ths = np.array([wl_center / (4.0 * materials_dict[p]) for p in pattern])
    
    def clamp_ths(ths):
        ths = np.clip(ths, 0.0, max_layer)
        tot = ths.sum()
        if tot > max_total:
            ths = ths * (max_total / tot)
        return ths
    
    init_ths = clamp_ths(init_ths)
    
    def objective(ths):
        ths = clamp_ths(ths)
        return _mean_reflectance(pattern, list(ths), problem)
    
    best_score = objective(init_ths)
    best_ths = init_ths.copy()
    
    # Simulated Annealing
    random.seed(42)
    np.random.seed(42)
    
    T = 1.0  # Initial temperature
    T_min = 0.001
    cooling = 0.9995
    current_ths = init_ths.copy()
    current_score = best_score
    
    step_size = 20.0  # nm
    
    while time.time() < deadline - 5:
        # Generate neighbor
        new_ths = current_ths.copy()
        # Randomly perturb a few layers
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            idx = random.randint(0, n_layers - 1)
            new_ths[idx] += random.gauss(0, step_size)
        
        new_ths = clamp_ths(new_ths)
        new_score = objective(new_ths)
        
        # Acceptance criterion
        if new_score < current_score:
            current_ths = new_ths
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_ths = new_ths.copy()
        else:
            delta = new_score - current_score
            if T > T_min:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_ths = new_ths
                    current_score = new_score
        
        T *= cooling
        
        # Occasionally reheat or adjust step size
        if random.random() < 0.01:
            step_size = max(5.0, step_size * 0.9)
    
    # Final local polish with Nelder-Mead if time permits
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            def lm_obj(x):
                x = clamp_ths(x)
                return _mean_reflectance(pattern, list(x), problem)
            
            res = minimize(
                lm_obj, best_ths,
                method="Nelder-Mead",
                options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-6, "adaptive": True},
            )
            if res.fun < best_score:
                cand = clamp_ths(res.x)
                s = lm_obj(cand)
                if s < best_score:
                    best_score = s
                    best_ths = cand
        except Exception:
            pass
    
    final_ths = clamp_ths(best_ths)
    if sum(final_ths) > max_total:
        final_ths = final_ths * (max_total / sum(final_ths))
    for i in range(len(final_ths)):
        final_ths[i] = max(0.0, min(max_layer, final_ths[i]))
    
    return list(pattern), list(final_ths)
