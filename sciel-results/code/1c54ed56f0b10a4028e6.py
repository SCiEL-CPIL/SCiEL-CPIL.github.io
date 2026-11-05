import time
import math
import numpy as np
import random

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)
    
    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    # beta shape: (n_layers, n_wls)
    # Use np.outer for speed: beta[i, j] = 2*pi*n_i*d_i / wls[j]
    # n_d = [n_mat[m] * thicknesses[i] for i, m in enumerate(materials)]
    # beta = 2 * pi * np.outer(n_d, 1.0 / wls)
    
    n_d = np.array([n_mat[m] * d for m, d in zip(materials, thicknesses)])
    beta = 2.0 * math.pi * np.outer(n_d, 1.0 / wls)

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        # Matrix elements for layer i
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        # Matrix multiplication: M_new = M_old * M_i
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    # Admittance Y
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    
    # Reflectance r
    r = (n_ambient - Y) / (n_ambient + Y)
    
    # Reflectance R = |r|^2
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent.
    Returns penalty if constraints are violated.
    """
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6
        
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6
            
    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6 + (total_th - max_total) * 10.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _nelder_mead(objective, x0, bounds, deadline, max_iter=500, tol=1e-6):
    """
    Nelder-Mead optimization with bounds handling.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lo, hi = bounds
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        range_i = hi[i] - lo[i]
        step = 0.1 * range_i if range_i > 1e-6 else 1.0
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))
        
    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        if abs(f_vals[0] - f_vals[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue
            
        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue
            
        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue
            
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])
            
    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]

def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]
    
    best_val = float('inf')
    best_mats = None
    best_ths = None
    
    # Try different numbers of layers
    # Prioritize higher layer counts as they allow flatter response
    # But start with lower to find a good baseline quickly
    layer_counts = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    for N in layer_counts:
        if N > max_layers:
            continue
        if time.time() > deadline:
            break
            
        lo = np.zeros(N)
        hi = np.full(N, max_layer)
        
        # Generate candidate sequences
        sequences = []
        
        # Alternating LHLH...
        seq1 = ['L' if i % 2 == 0 else 'H' for i in range(N)]
        sequences.append(seq1)
        
        # Alternating HLHL...
        seq2 = ['H' if i % 2 == 0 else 'L' for i in range(N)]
        sequences.append(seq2)
        
        # Add some random sequences if N is large enough
        if N >= 6:
            for _ in range(2):
                seq = []
                for i in range(N):
                    # Bias towards alternating
                    if i == 0:
                        seq.append(random.choice(['L', 'H']))
                    else:
                        # 70% chance to alternate, 30% chance to repeat
                        if random.random() < 0.7:
                            seq.append('H' if seq[-1] == 'L' else 'L')
                        else:
                            seq.append(seq[-1])
                sequences.append(seq)
        
        for mats in sequences:
            if time.time() > deadline:
                break
                
            # Generate initial guesses
            initial_guesses = []
            
            # Quarter-wave at center wavelength
            cw = (band_start + band_end) / 2.0
            ths_qw = []
            for m in mats:
                n = n_mat[m]
                q = cw / (4.0 * n)
                ths_qw.append(q)
            initial_guesses.append(ths_qw)
            
            # Quarter-wave with random perturbation
            ths_qp = []
            for m in mats:
                n = n_mat[m]
                q = cw / (4.0 * n)
                q *= random.uniform(0.8, 1.2)
                ths_qp.append(q)
            initial_guesses.append(ths_qp)
            
            # Random thicknesses
            ths_r = [random.uniform(10, 200) for _ in range(N)]
            initial_guesses.append(ths_r)
            
            # Tapered thicknesses (thinner on air side)
            ths_t = []
            for i, m in enumerate(mats):
                n = n_mat[m]
                base = cw / (4.0 * n)
                # Linear taper from 0.5x to 1.5x
                factor = 0.5 + (i / (N - 1)) * 1.0 if N > 1 else 1.0
                ths_t.append(base * factor)
            initial_guesses.append(ths_t)
            
            for x0 in initial_guesses:
                if time.time() > deadline:
                    break
                    
                x0 = np.clip(x0, lo, hi)
                s = sum(x0)
                if s > max_total:
                    x0 = x0 * (max_total / s)
                    x0 = np.clip(x0, 0.0, max_layer)
                
                def objective(th, m=mats):
                    return _mean_reflectance_percent(m, list(th), problem, wls, n_mat)
                
                x_opt, f_opt = _nelder_mead(objective, x0, (lo, hi), deadline, max_iter=300)
                
                x_opt = np.clip(x_opt, 0.0, max_layer)
                s = sum(x_opt)
                if s > max_total:
                    x_opt = x_opt * (max_total / s)
                    x_opt = np.clip(x_opt, 0.0, max_layer)
                    
                val = _mean_reflectance_percent(mats, list(x_opt), problem, wls, n_mat)
                
                if val < best_val:
                    best_val = val
                    best_mats = mats
                    best_ths = list(x_opt)
                    
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
        
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
