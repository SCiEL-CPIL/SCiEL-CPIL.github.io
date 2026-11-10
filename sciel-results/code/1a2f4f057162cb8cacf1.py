"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from scipy.optimize import minimize_scalar, minimize

# EVOLVE-BLOCK-START
import sys
import math

def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * z + c
    return result

def _eval_all(coeffs, angles):
    """Evaluate polynomial at all angles, return magnitudes."""
    zs = np.exp(1j * angles)
    vals = np.zeros(len(angles), dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    return np.abs(vals)

def _sup_norm_refined(coeffs, num_samples=8192):
    """
    Compute sup norm with golden-section refinement of top peaks.
    This is used for the final score to match evaluator accuracy.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    mag = _eval_all(coeffs, angles)
    
    best_val = float(np.max(mag))
    top_k = 8
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    window = 2 * np.pi / num_samples
    
    for idx in peak_indices:
        center = angles[idx] % (2 * np.pi)
        
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(_eval_poly(coeffs, z))
        
        lower = center - window
        upper = center + window
        
        try:
            res = minimize_scalar(obj, bounds=(lower, upper), 
                                  method='bounded', options={'xatol': 1e-12})
            if res.success:
                peak_mag = -res.fun
                if peak_mag > best_val:
                    best_val = peak_mag
        except Exception:
            pass
            
    return best_val

def _score_fast(coeffs, num_samples=2048):
    """Fast score for local search."""
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    mag = _eval_all(coeffs, angles)
    return float(np.max(mag)) / np.sqrt(len(coeffs))

def _soft_max_score(coeffs, num_samples=1024, beta=15.0):
    """Smooth surrogate using log-sum-exp."""
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    mag = _eval_all(coeffs, angles)
    mag = np.clip(mag, 1e-10, None)
    log_mag = np.log(mag)
    max_log = np.max(log_mag)
    shifted = log_mag - max_log
    sum_exp = np.sum(np.exp(shifted))
    soft_max = max_log + np.log(sum_exp)
    return float(soft_max) / np.sqrt(len(coeffs))

def _convex_relaxation(num_coeffs=24, num_angles=512, time_limit=30.0):
    """
    Solve continuous relaxation: minimize t subject to |P(e^{iθ})| <= t for all θ,
    with ||c||_1 = n as surrogate for ±1 constraint.
    Returns continuous solution and its projection to ±1.
    """
    start_time = time.time()
    
    # Precompute evaluation matrix: M[i,j] = e^{i*j*θ_i}
    angles = 2 * np.pi * np.arange(num_angles) / num_angles
    zs = np.exp(1j * angles)
    
    # Build Vandermonde-like matrix: row i is [1, z_i, z_i^2, ..., z_i^{n-1}]
    M = np.vander(zs, N=num_coeffs, increasing=True).T  # shape (num_coeffs, num_angles)
    
    def objective(x):
        """x = [c_0, ..., c_{n-1}, t]"""
        c = x[:num_coeffs]
        t = x[num_coeffs]
        # |P(z)| for all angles
        vals = M @ c  # shape (num_angles,)
        mags = np.abs(vals)
        # Penalize violations: sum of max(0, mag - t)^2
        violations = np.maximum(0, mags - t)
        penalty = np.sum(violations ** 2)
        # Regularize toward ±1: minimize sum((|c_i|^2 - 1)^2)
        reg = np.sum((np.abs(c) ** 2 - 1) ** 2)
        return t + 0.1 * penalty + 0.01 * reg
    
    # Initial guess: random ±1 with t = sqrt(n) * 1.3
    c0 = np.random.choice([-1, 1], size=num_coeffs).astype(float)
    t0 = np.sqrt(num_coeffs) * 1.3
    x0 = np.concatenate([c0, [t0]])
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Multi-start with different initial conditions
    num_starts = 5
    for s in range(num_starts):
        if time.time() - start_time > time_limit - 5:
            break
        
        if s == 0:
            x_init = x0
        else:
            # Perturbed start
            c_init = np.random.choice([-1, 1], size=num_coeffs).astype(float)
            t_init = np.sqrt(num_coeffs) * (1.0 + 0.5 * np.random.random())
            x_init = np.concatenate([c_init, [t_init]])
        
        try:
            res = minimize(objective, x_init, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Project to ±1
    c_cont = best_x[:num_coeffs]
    c_proj = np.sign(c_cont).astype(int)
    # Ensure no zeros
    c_proj[c_proj == 0] = 1
    
    return c_proj.tolist(), best_x

def _local_search(coeffs, time_limit=5.0, use_smooth=False):
    """Greedy local search: flip coefficients that improve score."""
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    
    if use_smooth:
        current_score = _soft_max_score(current)
    else:
        current_score = _score_fast(current)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            old_val = current[i]
            current[i] = -old_val
            if use_smooth:
                new_score = _soft_max_score(current)
            else:
                new_score = _score_fast(current)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
                
    return current, current_score

def _pair_flip_search(coeffs, time_limit=10.0):
    """Try flipping pairs of coefficients for better escape from local minima."""
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    current_score = _score_fast(current)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit - 0.5:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                old_i, old_j = current[i], current[j]
                current[i] = -old_i
                current[j] = -old_j
                new_score = _score_fast(current)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old_i
                    current[j] = old_j
                    
    return current, current_score

def _grs_sequence(length=24):
    """Generate Golay-Rudin-Shapiro sequence truncated to given length."""
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    n = 24
    
    # Phase 1: Convex relaxation to get good initial candidate
    best_coeffs = None
    best_score = float('inf')
    
    try:
        convex_coeffs, _ = _convex_relaxation(num_coeffs=n, num_angles=512, time_limit=25.0)
        if time.time() < deadline - 60:
            # Refine the convex solution with local search
            refined, score = _local_search(convex_coeffs, time_limit=5.0, use_smooth=True)
            hard_score = _score_fast(refined)
            if hard_score < best_score:
                best_score = hard_score
                best_coeffs = refined
    except Exception:
        pass
    
    # Phase 2: If convex relaxation didn't give good result, use GRS
    if best_coeffs is None or time.time() < deadline - 50:
        grs = _grs_sequence(n)
        grs_refined, grs_score = _local_search(grs, time_limit=5.0, use_smooth=True)
        grs_hard = _score_fast(grs_refined)
        if grs_hard < best_score:
            best_score = grs_hard
            best_coeffs = grs_refined
    
    # Phase 3: Multi-start with random initializations
    remaining = deadline - time.time()
    if remaining > 20:
        num_starts = min(10, int(remaining // 8))
        for s in range(num_starts):
            if time.time() > deadline - 15:
                break
            random_coeffs = [random.choice([-1, 1]) for _ in range(n)]
            # Quick smooth local search
            refined, _ = _local_search(random_coeffs, time_limit=3.0, use_smooth=True)
            hard_score = _score_fast(refined)
            if hard_score < best_score:
                best_score = hard_score
                best_coeffs = refined
    
    # Phase 4: Pair-flip refinement on best candidate
    if best_coeffs is not None and time.time() < deadline - 10:
        pair_refined, pair_score = _pair_flip_search(best_coeffs, time_limit=min(15.0, deadline - time.time() - 5))
        if pair_score < best_score:
            best_score = pair_score
            best_coeffs = pair_refined
    
    # Phase 5: Final hard local search polish
    if best_coeffs is not None and time.time() < deadline - 5:
        final_refined, final_score = _local_search(best_coeffs, time_limit=min(10.0, deadline - time.time() - 2), use_smooth=False)
        if final_score < best_score:
            best_score = final_score
            best_coeffs = final_refined
    
    # Ensure we have a valid result
    if best_coeffs is None:
        best_coeffs = _grs_sequence(n)
    
    return best_coeffs

# EVOLVE-BLOCK-END
