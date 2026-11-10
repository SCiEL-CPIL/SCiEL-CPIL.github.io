# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def _exact_1d_volume(x):
    """
    Compute the exact integral of the length of the union of sheared intervals
    [x_i + c*i/N, x_i + 1/N + c*i/N] for c in [0, 1].
    """
    N = 8
    x = np.asarray(x)
    s = np.arange(N) / N
    
    criticals = [0.0, 1.0]
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for off_i in [0.0, 1.0/N]:
                for off_j in [0.0, 1.0/N]:
                    ds = s[i] - s[j]
                    if abs(ds) < 1e-12:
                        continue
                    c_val = (x[j] + off_j - (x[i] + off_i)) / ds
                    if 0.0 < c_val < 1.0:
                        criticals.append(c_val)
    
    criticals = sorted(set([round(c, 15) for c in criticals]))
    
    total_vol = 0.0
    
    def union_length_at_c(c_val):
        starts = x + s * c_val
        ends = starts + 1.0/N
        idx = np.argsort(starts)
        starts_sorted = starts[idx]
        ends_sorted = ends[idx]
        
        current_max_end = -np.inf
        length = 0.0
        for i in range(N):
            s_i = starts_sorted[i]
            e_i = ends_sorted[i]
            if s_i > current_max_end:
                length += e_i - s_i
                current_max_end = e_i
            else:
                if e_i > current_max_end:
                    length += e_i - current_max_end
                    current_max_end = e_i
        return length

    for k in range(len(criticals) - 1):
        c_start = criticals[k]
        c_end = criticals[k + 1]
        if c_end - c_start < 1e-12:
            continue
        
        L_start = union_length_at_c(c_start)
        L_end = union_length_at_c(c_end)
        
        total_vol += 0.5 * (L_start + L_end) * (c_end - c_start)
        
    return total_vol

def _objective(params):
    x = params[:8]
    y = params[8:16]
    vx = _exact_1d_volume(x)
    vy = _exact_1d_volume(y)
    return vx * vy

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 580
    
    # Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Generate multiple initial guesses
    rng = np.random.default_rng(42)
    initial_guesses = [keich.copy()]
    
    # Add perturbed versions of keich
    for _ in range(5):
        perturbed = keich + rng.normal(0, 0.1, N)
        initial_guesses.append(perturbed)
    
    # Add random initializations
    for _ in range(3):
        random_init = rng.uniform(-0.5, 0.5, N)
        initial_guesses.append(random_init)
    
    # Also try sorted and anti-sorted versions
    initial_guesses.append(np.sort(keich))
    initial_guesses.append(-np.sort(keich))
    initial_guesses.append(keich[::-1])
    
    bounds = [(-1.0, 1.0)] * 16
    
    best_params = None
    best_score = np.inf
    
    # Phase 1: Local optimization from each initial guess
    for guess in initial_guesses:
        if time.time() > deadline - 30:
            break
        params0 = np.concatenate([guess, guess])
        try:
            res = scipy.optimize.minimize(
                _objective, 
                params0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Phase 2: Differential Evolution for global search
    if time.time() < deadline - 60:
        try:
            res_de = scipy.optimize.differential_evolution(
                _objective, 
                bounds, 
                maxiter=100, 
                popsize=8, 
                seed=42, 
                tol=1e-8,
                polish=True
            )
            if res_de.fun < best_score:
                best_score = res_de.fun
                best_params = res_de.x
        except Exception:
            pass
    
    # Phase 3: Final local refinement
    if best_params is not None and time.time() < deadline - 10:
        try:
            res_final = scipy.optimize.minimize(
                _objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_params = res_final.x
        except Exception:
            pass
    
    # Fallback to keich if nothing worked
    if best_params is None:
        x_opt = keich
        y_opt = keich
    else:
        x_opt = best_params[:8]
        y_opt = best_params[8:16]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
