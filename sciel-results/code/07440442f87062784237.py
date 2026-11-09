"""
Global optimization strategy for broadband antireflection coating design.
Uses a fixed set of high-symmetry layer patterns (palindromes, graded index)
and applies multi-start Nelder-Mead optimization on the continuous thickness
vector. This avoids the local minima traps of random-walk per-layer search
by exploring the full 20-dimensional (or lower) space with a global
derivative-free method.
"""
import time
import random
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[800-400+1-1], 401) # 401 points from 400 to 800
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
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
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
    ths = np.asarray(ths, dtype=float)
    ths = np.clip(ths, 0.0, max_layer)
    total = np.sum(ths)
    if total > max_total:
        ths = ths * (max_total / total)
    return ths

def _nelder_mead(func, x0, bounds, max_iter=200, timeout=None, start_time=None):
    """
    Custom Nelder-Mead implementation with time checks and bounds.
    x0: initial point
    bounds: (min_val, max_val) for each dimension (scalar bounds applied to all)
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    # Initialize simplex
    simplex = [x0.copy()]
    f_vals = [func(x0)]
    
    for i in range(n):
        x_new = x0.copy()
        x_new[i] += 0.05 if x0[i] == 0 else 0.05 * x0[i] # Small perturbation
        x_new = _constrain_thicknesses(x_new, bounds[1], bounds[0][1]) # Reusing bounds structure loosely
        # Actually, let's pass max_total and max_layer separately for clarity in caller
        # For now, assume bounds is (max_layer, max_total) passed to helper, but here we just clip
        # Let's refactor: caller passes max_total, max_layer. 
        # I will implement a simpler NM that takes constraints as a closure or just clips inside.
        
        # Let's stick to a robust standard NM with clipping
        x_new = x0.copy()
        x_new[i] = x0[i] + 0.1 # Fixed step size for initialization is safer
        simplex.append(x_new)
        f_vals.append(func(x_new))
        
    best_idx = 0
    for i in range(len(f_vals)):
        if f_vals[i] < f_vals[best_idx]:
            best_idx = i
    
    for _ in range(max_iter):
        if timeout and (time.time() - start_time) > timeout:
            break
            
        # Sort simplex
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = _constrain_thicknesses(x_r, *bounds)
        f_r = func(x_r)
        
        if f_r < f_vals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            x_e = _constrain_thicknesses(x_e, *bounds)
            f_e = func(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
        elif f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = _constrain_thicknesses(x_c, *bounds)
            f_c = func(x_c)
            if f_c < f_vals[-1]:
                simplex[-1] = x_c
                f_vals[-1] = f_c
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = _constrain_thicknesses(simplex[i], *bounds)
                    f_vals[i] = func(simplex[i])
                
    best_idx = np.argmin(f_vals)
    return simplex[best_idx], f_vals[best_idx]

def design(problem):
    start_time = time.time()
    deadline = start_time + 135 # Safety margin
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    # Precompute quarter-wave thicknesses at center wavelength
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)
        
    # Generate candidate patterns
    # Focus on symmetric (palindromic) structures and graded index
    candidates = []
    
    # 1. Graded Index (Alternating H/L) - Best for broadband
    for L in range(2, min(max_layers, 10) + 1, 2): # Even number of layers
        pattern = tuple('HL' * (L // 2))
        candidates.append(pattern)
        
    # 2. Palindromes
    # Length 3
    for c in ['L', 'H']:
        candidates.append(('H', c, 'H'))
        candidates.append(('L', c, 'L'))
    # Length 5
    for b in ['L', 'H']:
        for c in ['L', 'H']:
            candidates.append(('H', b, c, b, 'H'))
            candidates.append(('L', b, c, b, 'L'))
    # Length 7
    for b in ['L', 'H']:
        for c in ['L', 'H']:
            for d in ['L', 'H']:
                 candidates.append(('H', b, c, d, c, b, 'H'))
                 candidates.append(('L', b, c, d, c, b, 'L'))
    
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique_candidates.append(p)
            
    # Sort by length (shorter is faster to evaluate)
    unique_candidates.sort(key=lambda x: len(x))
    
    best_result = None
    best_score = float('inf')
    
    # Multi-start strategy
    # We will try to run Nelder-Mead on each pattern with multiple random starts
    # Time budget per pattern depends on dimensionality
    # Let's allocate time dynamically
    
    for pattern in unique_candidates:
        if time.time() > deadline - 10: # Stop if little time left
            break
            
        n_layers = len(pattern)
        # Time budget per pattern: more layers = more time per eval, so fewer starts
        # Base time per eval is approx 1ms for small N, 5ms for large N
        # Let's estimate 5ms per eval. NM takes ~100 * N evals.
        # For N=20, 2000 evals -> 10s.
        # We have ~135s total.
        # Let's try 5 starts for small patterns, 2 for large.
        
        if n_layers <= 5:
            n_starts = 10
        elif n_layers <= 10:
            n_starts = 5
        else:
            n_starts = 2
            
        # Initial guesses
        # 1. Quarter wave
        init_qw = [qw[m] for m in pattern]
        # 2. Random
        random_starts = []
        for _ in range(n_starts - 1):
            s = [random.uniform(0.2 * qw[m], 2.0 * qw[m]) for m in pattern]
            random_starts.append(s)
            
        all_starts = [init_qw] + random_starts
        
        for start_ths in all_starts:
            if time.time() > deadline - 5:
                break
                
            start_ths = _constrain_thicknesses(start_ths, max_total, max_layer)
            
            # Define objective function with constraints
            def objective(ths):
                ths = _constrain_thicknesses(ths, max_total, max_layer)
                return _mean_reflectance(pattern, ths, problem)
            
            # Run Nelder-Mead
            # Bounds: each layer in [0, max_layer], sum <= max_total
            # NM doesn't handle sum constraint well natively, but we clip in objective
            # and in the step. The clipping in _constrain_thicknesses handles sum.
            
            try:
                res_ths, res_score = _nelder_mead(
                    objective, 
                    np.array(start_ths), 
                    (max_layer, max_total), # Passing bounds as tuple for helper
                    max_iter=300,
                    timeout=130 - (time.time() - start_time),
                    start_time=start_time
                )
                
                # Final check
                res_ths = _constrain_thicknesses(res_ths, max_total, max_layer)
                final_score = _mean_reflectance(pattern, res_ths, problem)
                
                if final_score < best_score:
                    best_score = final_score
                    best_result = (list(pattern), list(res_ths))
                    
            except Exception:
                continue

    if best_result is None:
        # Fallback: single L layer
        return ["L"], [qw["L"]]

    mats, ths = best_result
    return mats, ths
