import time
import math
import numpy as np
import random
from scipy.optimize import minimize

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
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d != 0:
            beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    # M = M_1 * M_2 * ... * M_N
    # Each M_i = [[cos(beta_i), i*sin(beta_i)/n_i], [i*n_i*sin(beta_i), cos(beta_i)]]
    
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        # Matrix elements for layer i
        # A = cos(beta), B = i*sin(beta)/n, C = i*n*sin(beta), D = cos(beta)
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        # Matrix multiplication: M_new = M_old * M_i
        # M_old is [[M11, M12], [M21, M22]]
        
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    # Admittance Y
    # Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    
    # Reflectance r
    # r = (n_ambient - Y) / (n_ambient + Y)
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
    
    # Check individual thickness bounds
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6
            
    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        # Heavy penalty for total thickness violation
        return 1e6 + (total_th - max_total) * 100.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _optimize_thicknesses(materials, x0, problem, wls, n_mat, deadline, max_iter=200):
    """
    Powell optimization for thicknesses of a fixed material sequence.
    Uses scipy.optimize.minimize with method='Powell'.
    """
    n = len(x0)
    lo = np.zeros(n)
    hi = np.full(n, problem["max_layer_nm"])
    max_total = problem["max_total_nm"]
    
    # Ensure initial guess is valid
    x = np.array(x0, dtype=float)
    x = np.clip(x, lo, hi)
    total = np.sum(x)
    if total > max_total:
        x = x * (max_total / total)
        x = np.clip(x, lo, hi)
        
    def objective(x):
        # Check total thickness constraint strictly during optimization
        if np.sum(x) > max_total:
            # Penalize proportionally to help the optimizer back off
            return 1e6 + (np.sum(x) - max_total) * 10.0
        return _mean_reflectance_percent(materials, list(x), problem, wls, n_mat)

    # Powell's method
    # maxfev is the maximum number of function evaluations.
    # We estimate time per eval and set a limit.
    # A safe guess: 1000 evals might take 1-2 seconds for 10 layers.
    # Let's set maxfev based on time remaining.
    
    # To be safe, we limit maxfev.
    # If we have 30 seconds left, and 1 eval takes ~5ms, we can do 6000 evals.
    # But Powell can be slow. Let's cap it.
    maxfev = 2000
    
    try:
        res = minimize(objective, x, method='Powell', bounds=list(zip(lo, hi)), 
                       options={'maxiter': max_iter, 'maxfev': maxfev, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.success or res.fun < objective(x):
             x_opt = res.x
             f_opt = res.fun
        else:
             x_opt = x
             f_opt = objective(x)
    except Exception:
        x_opt = x
        f_opt = objective(x)
        
    return np.clip(x_opt, lo, hi), f_opt

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    # 150s wall-clock limit. Keep a safety margin of 10s.
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
    
    # Generate candidate material sequences
    sequences = []
    
    # 1. Alternating sequences
    for length in range(2, max_layers + 1, 2):
        # Start with L
        seq_l = ['L' if i % 2 == 0 else 'H' for i in range(length)]
        sequences.append(seq_l)
        # Start with H
        seq_h = ['H' if i % 2 == 0 else 'L' for i in range(length)]
        sequences.append(seq_h)
        
    # 2. Random sequences with constraints (no more than 2 consecutive)
    random.seed(42)
    for length in range(4, max_layers + 1, 2): # Focus on even lengths for symmetry
        for _ in range(5): # 5 random tries per length
            seq = []
            for i in range(length):
                if i == 0:
                    seq.append(random.choice(['L', 'H']))
                else:
                    # Avoid 3 consecutive
                    if len(seq) >= 2 and seq[-1] == seq[-2]:
                        seq.append('L' if seq[-1] == 'H' else 'H')
                    else:
                        seq.append(random.choice(['L', 'H']))
            # Check if it's already in sequences
            if seq not in sequences:
                sequences.append(seq)

    # Optimization parameters
    max_iter_per_seq = 100 # Reduced since Powell is efficient
    
    for mats in sequences:
        if time.time() > deadline - 20.0: # Leave time for final polish
            break
            
        n_layers = len(mats)
        
        # Generate initial guesses
        # 1. Quarter-wave at geometric mean of band
        lambda_gm = math.sqrt(band_start * band_end)
        x0_gm = []
        for m in mats:
            n = n_mat[m]
            d = lambda_gm / (4.0 * n)
            x0_gm.append(d)
            
        # 2. Quarter-wave at arithmetic mean
        lambda_am = (band_start + band_end) / 2.0
        x0_am = []
        for m in mats:
            n = n_mat[m]
            d = lambda_am / (4.0 * n)
            x0_am.append(d)
            
        # 3. Random perturbation around GM
        np.random.seed(42 + n_layers) # Different seed per layer count
        x0_rand = []
        for m in mats:
            n = n_mat[m]
            d_base = lambda_gm / (4.0 * n)
            d = d_base * np.random.uniform(0.8, 1.2)
            x0_rand.append(d)
            
        initial_guesses = [x0_gm, x0_am, x0_rand]
        
        for x0 in initial_guesses:
            if time.time() > deadline - 10.0:
                break
                
            # Scale to fit total thickness if needed
            total = sum(x0)
            if total > max_total:
                factor = max_total / total
                x0 = [d * factor for d in x0]
            
            # Run optimization
            x_opt, f_opt = _optimize_thicknesses(mats, x0, problem, wls, n_mat, deadline, max_iter=max_iter_per_seq)
            
            # Post-process: Ensure constraints
            x_opt = np.clip(x_opt, 0.0, max_layer)
            total_opt = sum(x_opt)
            if total_opt > max_total:
                factor = max_total / total_opt
                x_opt = x_opt * factor
                x_opt = np.clip(x_opt, 0.0, max_layer)
            
            # Calculate final score
            val = _mean_reflectance_percent(mats, list(x_opt), problem, wls, n_mat)
            
            if val < best_val:
                best_val = val
                best_mats = mats
                best_ths = list(x_opt)
                
    # Final high-precision polish pass if time permits
    if best_mats is not None and time.time() < deadline - 5.0:
        n_layers = len(best_mats)
        
        # Run a final polish with more iterations
        x_polish, f_polish = _optimize_thicknesses(best_mats, best_ths, problem, wls, n_mat, deadline, max_iter=300)
        
        # Post-process
        x_polish = np.clip(x_polish, 0.0, max_layer)
        total_polish = sum(x_polish)
        if total_polish > max_total:
            factor = max_total / total_polish
            x_polish = x_polish * factor
            x_polish = np.clip(x_polish, 0.0, max_layer)
            
        val_polish = _mean_reflectance_percent(best_mats, list(x_polish), problem, wls, n_mat)
        
        if val_polish < best_val:
            best_val = val_polish
            best_ths = list(x_polish)
            
    if best_mats is None:
        # Fallback: single layer quarter-wave at center
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
        
    # Final constraint check and scaling
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
