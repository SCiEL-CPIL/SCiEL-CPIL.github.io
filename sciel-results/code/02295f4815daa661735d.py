import time
import numpy as np
from scipy.optimize import minimize_scalar
import random

# Constants
N_COEFFS = 24
N_SAMPLES_SA = 2048
N_SAMPLES_REFINE = 30000
TIME_LIMIT = 120.0

# Precomputed data
_POWERS_SA = None
_POWERS_REFINE = None
_ANGLES_SA = None
_ANGLES_REFINE = None

def _precompute():
    global _POWERS_SA, _POWERS_REFINE, _ANGLES_SA, _ANGLES_REFINE
    
    # Precompute for SA loop (faster, lower res)
    _ANGLES_SA = 2 * np.pi * np.arange(N_SAMPLES_SA) / N_SAMPLES_SA
    _POWERS_SA = np.ones((N_COEFFS, N_SAMPLES_SA), dtype=complex)
    for k in range(1, N_COEFFS):
        _POWERS_SA[k] = _POWERS_SA[k-1] * np.exp(1j * _ANGLES_SA)
        
    # Precompute for Refinement (slower, higher res)
    _ANGLES_REFINE = 2 * np.pi * np.arange(N_SAMPLES_REFINE) / N_SAMPLES_REFINE
    _POWERS_REFINE = np.ones((N_COEFFS, N_SAMPLES_REFINE), dtype=complex)
    for k in range(1, N_COEFFS):
        _POWERS_REFINE[k] = _POWERS_REFINE[k-1] * np.exp(1j * _ANGLES_REFINE)

def _eval_poly_magnitude(coeffs, powers):
    """
    Evaluate max |P(z)| for given coeffs and precomputed powers.
    coeffs: array of shape (N_COEFFS,) with values in {-1, 1}
    powers: array of shape (N_COEFFS, N_SAMPLES)
    """
    # P(z) = sum c_k z^k
    # We have powers[k, j] = z_j^k
    # vals[j] = sum_k c_k powers[k, j]
    vals = powers.T @ coeffs
    return float(np.max(np.abs(vals)))

def _high_precision_sup_norm(coeffs):
    """
    Compute sup norm with high precision using dense sampling and refinement.
    """
    # 1. Dense sampling
    mag = _eval_poly_magnitude(coeffs, _POWERS_REFINE)
    
    # 2. Identify peaks
    # Get indices of top 20 peaks
    top_k = 20
    # We need the actual values to find peaks, so we recompute vals
    vals = _POWERS_REFINE.T @ coeffs
    mags = np.abs(vals)
    peak_indices = np.argsort(mags)[-top_k:][::-1]
    
    best_val = float(np.max(mags))
    window = 2 * np.pi / N_SAMPLES_REFINE
    
    # Helper to evaluate polynomial at a specific angle
    def eval_poly_at_angle(theta):
        z = np.exp(1j * theta)
        # Horner's method for scalar
        res = 0.0
        for c in reversed(coeffs):
            res = res * z + c
        return abs(res)
        
    for idx in peak_indices:
        center = _ANGLES_REFINE[idx]
        
        def obj(theta):
            return -eval_poly_at_angle(theta)
            
        try:
            res = minimize_scalar(obj, bounds=(center - window, center + window),
                                  method='bounded', options={'xatol': 1e-14})
            if res.success:
                peak_mag = -res.fun
                if peak_mag > best_val:
                    best_val = peak_mag
        except:
            pass
            
    return best_val

def _simulated_annealing(time_budget):
    """
    Run Simulated Annealing to find a good candidate.
    """
    start_time = time.time()
    
    # Initialize with a random sequence or GRS
    # GRS is a good starting point
    p, q = [1], [1]
    while len(p) < N_COEFFS:
        p, q = p + q, p + [-c for c in q]
    current = np.array(p[:N_COEFFS], dtype=float)
    
    current_energy = _eval_poly_magnitude(current, _POWERS_SA)
    best = current.copy()
    best_energy = current_energy
    
    T = 1.0
    cooling = 0.999
    min_T = 1e-4
    
    steps = 0
    while time.time() - start_time < time_budget:
        # Flip a random coefficient
        idx = random.randint(0, N_COEFFS - 1)
        
        # Calculate delta energy
        # E_new = E_old - 2 * c_idx * P_idx
        # But we need max, so we can't just use delta. We must recompute max.
        # Optimization: We can update the vector of values directly?
        # vals = powers.T @ coeffs
        # If we flip coeffs[idx], vals changes by -2 * coeffs[idx] * powers[idx]
        
        old_c = current[idx]
        current[idx] = -old_c
        
        # Update vals efficiently?
        # We don't have vals stored globally in the SA loop to save memory? 
        # Actually, we can store vals.
        # Let's restructure to store vals.
        
        # For now, simple recompute is fine for 2048 points.
        new_energy = _eval_poly_magnitude(current, _POWERS_SA)
        
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best = current.copy()
        else:
            current[idx] = old_c
            
        T *= cooling
        steps += 1
        
        # Check time more frequently
        if steps % 1000 == 0:
            if time.time() - start_time > time_budget:
                break
                
    return best, best_energy

def _local_search_polish(coeffs, time_budget):
    """
    Perform high-precision local search on a candidate.
    """
    start_time = time.time()
    current = list(coeffs)
    current_score = _high_precision_sup_norm(current)
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_budget:
            break
            
        # Single flips
        for i in range(N_COEFFS):
            if time.time() - start_time > time_budget:
                break
            old_val = current[i]
            current[i] = -old_val
            new_score = _high_precision_sup_norm(current)
            if new_score < current_score - 1e-9:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
                
        # Pair flips
        if time.time() - start_time > time_budget:
            break
        for i in range(N_COEFFS):
            if time.time() - start_time > time_budget:
                break
            for j in range(i + 1, N_COEFFS):
                if time.time() - start_time > time_budget:
                    break
                vi, vj = current[i], current[j]
                current[i] = -vi
                current[j] = -vj
                new_score = _high_precision_sup_norm(current)
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = vi
                    current[j] = vj
                    
    return current, current_score

def construct():
    deadline = time.time() + TIME_LIMIT
    
    # Precompute
    _precompute()
    
    # Phase 1: Simulated Annealing
    sa_time = 60.0
    best_coeffs, best_energy = _simulated_annealing(sa_time)
    
    # Phase 2: Polish the best candidate from SA
    # Also try a few other good candidates?
    # For simplicity, polish the best one found.
    # To be safer, we can run SA multiple times or keep a hall of fame.
    # Given time constraints, we'll polish the single best SA result,
    # and also polish a GRS sequence just in case.
    
    candidates = [best_coeffs]
    
    # Add GRS as a backup candidate
    p, q = [1], [1]
    while len(p) < N_COEFFS:
        p, q = p + q, p + [-c for c in q]
    grs = np.array(p[:N_COEFFS], dtype=float)
    if not np.array_equal(grs, best_coeffs):
        candidates.append(grs)
        
    # Add a random candidate
    rand_cand = np.random.choice([-1, 1], size=N_COEFFS).astype(float)
    candidates.append(rand_cand)
    
    best_final_coeffs = None
    best_final_score = float('inf')
    
    polish_time_each = 10.0
    
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        polished, score = _local_search_polish(cand, polish_time_each)
        if score < best_final_score:
            best_final_score = score
            best_final_coeffs = polished
            
    # Final high-precision verification and last-ditch effort
    if best_final_coeffs is not None:
        # One more polish with even tighter time if available?
        # The local search above already uses high precision.
        pass
        
    # Return as list of ints
    return [int(c) for c in best_final_coeffs]
