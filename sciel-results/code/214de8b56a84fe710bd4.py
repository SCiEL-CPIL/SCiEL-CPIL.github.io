"""
Broadband antireflection coating design: Fixed symmetric graded structure with L-BFGS-B optimization.

Strategy:
  1. Fix the layer sequence to a 20-layer symmetric alternating pattern (L H L H ... L H H L ... H L) 
     to maximize index contrast and symmetry, which is physically favorable for broadband AR.
     Wait, 20 layers alternating L/H starting with L:
     L H L H L H L H L H H L H L H L H L H L ? No, symmetric.
     Let's use a specific high-performing structure:
     Layer indices: 1..20
     Symmetric: i and 21-i have same material.
     Alternating: L H L H L H L H L H (first 10)
     Second 10: H L H L H L H L H L (to be symmetric? No, if 1=L, 20=L. 2=H, 19=H. etc.)
     Pattern: L H L H L H L H L H H L H L H L H L H L
     Let's check:
     1:L, 2:H, 3:L, 4:H, 5:L, 6:H, 7:L, 8:H, 9:L, 10:H
     11:H, 12:L, 13:H, 14:L, 15:H, 16:L, 17:H, 18:L, 19:H, 20:L
     This is symmetric: 1=L, 20=L. 2=H, 19=H. 3=L, 18=L. 4=H, 17=H. 5=L, 16=L. 6=H, 15=H. 7=L, 14=L. 8=H, 13=H. 9=L, 12=L. 10=H, 11=H.
     This is a valid symmetric alternating pattern.
     
  2. Use scipy.optimize.minimize with L-BFGS-B to optimize the 20 thicknesses.
     L-BFGS-B is efficient for medium-dimensional continuous problems and handles bounds well.
  3. Multi-start: Initialize with quarter-wave thicknesses, then add random perturbations.
  4. Deadline guard: Return best result found before time limit.
"""

import time
import numpy as np
from scipy.optimize import minimize

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
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

def design(problem):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    # Define the fixed 20-layer symmetric alternating pattern
    # L H L H L H L H L H H L H L H L H L H L
    pattern = ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 
               'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L']
    
    # Verify symmetry
    # 0:L, 19:L
    # 1:H, 18:H
    # 2:L, 17:L
    # 3:H, 16:H
    # 4:L, 15:L
    # 5:H, 14:H
    # 6:L, 13:L
    # 7:H, 12:H
    # 8:L, 11:L
    # 9:H, 10:H
    # Correct.

    n_layers = len(pattern)
    ns = [materials_dict[m] for m in pattern]
    
    # Objective function for scipy
    def objective(ths):
        if time.time() > deadline:
            return 100.0 # Penalize if over time
        return _mean_reflectance(pattern, ths, problem)
    
    # Bounds for L-BFGS-B
    # Individual layer: [0, max_layer]
    # Total sum: <= max_total (handled by scaling or constraint? L-BFGS-B handles bounds well, 
    # but sum constraint is linear. We can use a penalty or just rely on bounds if we scale.
    # Better: Use bounds [0, max_layer] and check sum. If sum > max_total, we might violate.
    # Let's add a penalty for sum violation in the objective to guide it, or use SLSQP.
    # L-BFGS-B is faster. Let's stick to bounds and hope it stays under, or scale.
    # Actually, the optimal solution for AR coatings usually uses significant thickness.
    # Let's use bounds [0, max_layer] and a penalty term for sum > max_total.
    
    def objective_with_penalty(ths):
        if time.time() > deadline:
            return 1000.0
        val = _mean_reflectance(pattern, ths, problem)
        total = sum(ths)
        if total > max_total:
            val += (total - max_total) * 10.0 # Penalty
        return val
    
    bounds = [(0.0, max_layer) for _ in range(n_layers)]
    
    best_score = float('inf')
    best_ths = None
    
    # Multi-start initialization
    # Start 1: Quarter-wave at center wavelength
    qw = [wl_center / (4.0 * n) for n in ns]
    
    # Start 2: Quarter-wave at min wavelength
    qw_min = [band[0] / (4.0 * n) for n in ns]
    
    # Start 3: Quarter-wave at max wavelength
    qw_max = [band[1] / (4.0 * n) for n in ns]
    
    # Start 4-10: Random perturbations around quarter-wave
    np.random.seed(42)
    initial_guesses = [qw, qw_min, qw_max]
    for _ in range(7):
        perturb = np.random.uniform(-20, 20, n_layers)
        guess = [max(0, min(max_layer, q + p)) for q, p in zip(qw, perturb)]
        initial_guesses.append(guess)
    
    # Optimize each start
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 5:
            break
            
        try:
            # L-BFGS-B is good for this size
            res = minimize(objective_with_penalty, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if res.success or res.fun < best_score:
                ths = res.x
                # Check constraints
                if sum(ths) <= max_total + 1e-6 and all(0 <= t <= max_layer for t in ths):
                    score = _mean_reflectance(pattern, ths, problem)
                    if score < best_score:
                        best_score = score
                        best_ths = ths
        except Exception:
            continue
            
    # If L-BFGS-B didn't find a valid solution or score is still inf, try a simple grid or fallback
    if best_ths is None:
        # Fallback: Quarter-wave stack
        best_ths = qw
        best_score = _mean_reflectance(pattern, best_ths, problem)
        
    return pattern, list(best_ths)

