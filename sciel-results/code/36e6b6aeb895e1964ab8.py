"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
import sys

# Precompute DFT matrices for the decomposition approach
_N = 24
_HALF = 12

# For P(z) = A(z) + z^12 * B(z) where A and B are degree-11 polynomials
# |P(z)|^2 = |A(z)|^2 + |B(z)|^2 + 2*Re(A(z) * conj(B(z)) * z^12)
# We can precompute DFT values for all 2^12 possible A and B polynomials
# But 2^12 = 4096, and we need to evaluate at many points.
# Instead, we precompute the DFT matrix for 12-coefficient polynomials at N points.

def _precompute_dft_12():
    """Precompute DFT for 12-coefficient polynomials at 24 equally spaced points."""
    # We need to evaluate polynomials of degree 11 at 24 points on the unit circle
    # DFT_matrix[k, j] = exp(2*pi*i*k*j/24) for k=0..23, j=0..11
    # Then for a 12-coeff vector c, P(z_k) = sum_j c_j * DFT_matrix[k, j]
    k = np.arange(24)
    j = np.arange(12)
    DFT = np.exp(2j * np.pi * np.outer(k, j) / 24)
    return DFT

_DFT_12 = _precompute_dft_12()

def _eval_12_at_24(coeffs_12):
    """Evaluate a 12-coefficient polynomial at 24 equally spaced points on unit circle."""
    return _DFT_12 @ np.array(coeffs_12, dtype=complex)

def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * z + c
    return result

def _score_decomposed(A, B, num_samples=24):
    """
    Compute sup norm for P(z) = A(z) + z^12 * B(z) using precomputed DFT.
    A and B are lists of 12 coefficients in {+1, -1}.
    """
    # Evaluate A and B at 24 equally spaced points
    A_vals = _eval_12_at_24(A)
    B_vals = _eval_12_at_24(B)
    
    # P(z_k) = A(z_k) + z_k^12 * B(z_k)
    # z_k^12 = exp(2*pi*i*k*12/24) = exp(pi*i*k) = (-1)^k
    z12 = np.array([(-1.0)**k for k in range(24)])
    P_vals = A_vals + z12 * B_vals
    
    mag = np.abs(P_vals)
    return float(np.max(mag))

def _score_decomposed_refined(A, B, num_samples=4096):
    """
    Compute sup norm with golden-section refinement for decomposed polynomial.
    """
    # First get the 24-point estimate to find peak regions
    A_vals_24 = _eval_12_at_24(A)
    B_vals_24 = _eval_12_at_24(B)
    z12_24 = np.array([(-1.0)**k for k in range(24)])
    P_vals_24 = A_vals_24 + z12_24 * B_vals_24
    mag_24 = np.abs(P_vals_24)
    
    # Also do dense sampling
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Evaluate A(z) and B(z) at dense points
    # A(z) = sum_{j=0}^{11} A[j] * z^j
    # B(z) = sum_{j=0}^{11} B[j] * z^j
    # P(z) = A(z) + z^12 * B(z)
    
    A_dense = np.zeros(num_samples, dtype=complex)
    for j in range(12):
        A_dense += A[j] * zs**j
    
    B_dense = np.zeros(num_samples, dtype=complex)
    for j in range(12):
        B_dense += B[j] * zs**j
    
    P_dense = A_dense + zs**12 * B_dense
    mag_dense = np.abs(P_dense)
    
    best_val = float(np.max(mag_dense))
    
    # Refine top peaks
    top_k = 5
    peak_indices = np.argsort(mag_dense)[-top_k:][::-1]
    window = 2 * np.pi / num_samples
    
    for idx in peak_indices:
        center = angles[idx] % (2 * np.pi)
        
        def obj(theta):
            z = np.exp(1j * theta)
            # Evaluate A(z) and B(z)
            az = 0.0
            for j in range(12):
                az = az * z + A[11-j]  # Horner from highest to lowest
            bz = 0.0
            for j in range(12):
                bz = bz * z + B[11-j]
            pz = az + z**12 * bz
            return -abs(pz)
        
        lower = center - window
        upper = center + window
        try:
            res = minimize_scalar(obj, bounds=(lower, upper), 
                                  method='bounded', options={'xatol': 1e-10})
            if res.success:
                peak_mag = -res.fun
                if peak_mag > best_val:
                    best_val = peak_mag
        except Exception:
            pass
            
    return best_val

def _sup_norm_fast(coeffs, num_samples=1024):
    """
    Compute sup norm of P(z) over |z|=1 via dense sampling.
    This is used for the local search steps for speed.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Vectorized evaluation using Horner's method
    # coeffs is a list of real numbers, zs is a complex array
    # We want P(zs) for all zs.
    # P(z) = c0 + c1*z + ... + c_{n-1}*z^{n-1}
    # We can compute this by iterating coefficients from high to low.
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
        
    mag = np.abs(vals)
    return float(np.max(mag))

def _local_search_decomposed(A, B, time_limit=2.0):
    """
    Local search on the decomposed representation.
    Try flipping single bits in A and B, and pairs (one in A, one in B).
    """
    start_time = time.time()
    A = list(A)
    B = list(B)
    current_score = _score_decomposed(A, B)
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
            
        # Try flipping single bits in A
        for i in range(12):
            if time.time() - start_time > time_limit:
                break
            A[i] = -A[i]
            new_score = _score_decomposed(A, B)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                A[i] = -A[i]
                
        # Try flipping single bits in B
        for i in range(12):
            if time.time() - start_time > time_limit:
                break
            B[i] = -B[i]
            new_score = _score_decomposed(A, B)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                B[i] = -B[i]
                
        # Try flipping pairs (one in A, one in B) - this can escape local minima
        if time.time() - start_time > time_limit:
            break
        for i in range(12):
            if time.time() - start_time > time_limit:
                break
            for j in range(12):
                if time.time() - start_time > time_limit:
                    break
                A[i] = -A[i]
                B[j] = -B[j]
                new_score = _score_decomposed(A, B)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    A[i] = -A[i]
                    B[j] = -B[j]
                    
    return A, B, current_score

def _sup_norm_refined(coeffs, num_samples=4096):
    """
    Compute sup norm with golden-section refinement of top peaks.
    This is used for the final score to match evaluator accuracy.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
        
    mag = np.abs(vals)
    
    # Find top peaks for refinement
    top_k = 5
    if num_samples > top_k:
        peak_indices = np.argsort(mag)[-top_k:][::-1]
    else:
        peak_indices = np.arange(num_samples)
        
    best_val = float(np.max(mag))
    window = 2 * np.pi / num_samples
    
    for idx in peak_indices:
        center = angles[idx]
        # Ensure center is within [0, 2pi)
        center = center % (2 * np.pi)
        
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(_eval_poly(coeffs, z))
        
        # Bounded minimization (maximization of -abs)
        # We search in a small window around the peak
        lower = center - window
        upper = center + window
        
        # Handle wrapping around 2pi
        # If lower < 0 or upper > 2pi, we might need to be careful, 
        # but minimize_scalar handles bounds. 
        # However, the function is periodic. 
        # For simplicity, we just search in the local window.
        try:
            res = minimize_scalar(obj, bounds=(lower, upper), 
                                  method='bounded', options={'xatol': 1e-10})
            if res.success:
                peak_mag = -res.fun
                if peak_mag > best_val:
                    best_val = peak_mag
        except Exception:
            pass
            
    return best_val

def _score(coeffs, num_samples=1024, refine=False):
    """Score: max |P(z)| / sqrt(n)"""
    n = len(coeffs)
    if refine:
        return _sup_norm_refined(coeffs, num_samples=4096) / np.sqrt(n)
    else:
        return _sup_norm_fast(coeffs, num_samples=num_samples) / np.sqrt(n)

def _local_search(coeffs, time_limit=5.0):
    """
    Greedy local search: try flipping single coefficients.
    Returns the best coeffs found and its fast score.
    """
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    current_score = _score(current, num_samples=1024)
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
            
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            old_val = current[i]
            current[i] = -old_val
            new_score = _score(current, num_samples=1024)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
                
    return current, current_score

def _grs_sequence(length=24):
    """Generate Golay-Rudin-Shapiro sequence truncated to given length."""
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def _grs_decomposed():
    """Generate GRS sequence split into two 12-coefficient halves."""
    full = _grs_sequence(24)
    A = full[:12]
    B = full[12:]
    return A, B

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Use the decomposed approach: P(z) = A(z) + z^12 * B(z)
    # This allows much faster scoring via precomputed DFT
    
    # Initialize with GRS
    A, B = _grs_decomposed()
    best_A, best_B = list(A), list(B)
    best_score = _score_decomposed(best_A, best_B)
    
    # Run local search from GRS
    A_ls, B_ls, score_ls = _local_search_decomposed(best_A, best_B, time_limit=5.0)
    if score_ls < best_score:
        best_score = score_ls
        best_A, best_B = A_ls, B_ls
    
    # Random restarts with decomposed local search
    # Each local search is fast (~0.1-0.5s) due to 24-point DFT evaluation
    # We can do many more restarts
    
    num_restarts = 0
    while time.time() < deadline - 15:
        # Random initialization
        A = [random.choice([-1, 1]) for _ in range(12)]
        B = [random.choice([-1, 1]) for _ in range(12)]
        
        # Local search
        A_ls, B_ls, score_ls = _local_search_decomposed(A, B, time_limit=1.0)
        if score_ls < best_score:
            best_score = score_ls
            best_A, best_B = A_ls, B_ls
            num_restarts += 1
            
            # If we're very close to optimum, do a more thorough search
            if best_score < 1.26 * np.sqrt(24):
                # Try more aggressive local search
                A_ls2, B_ls2, score_ls2 = _local_search_decomposed(best_A, best_B, time_limit=3.0)
                if score_ls2 < best_score:
                    best_score = score_ls2
                    best_A, best_B = A_ls2, B_ls2
    
    # Convert back to full 24 coefficients
    best_coeffs = best_A + best_B
    
    # Final verification with refined scoring
    final_score = _score_decomposed_refined(best_A, best_B)
    
    return best_coeffs

# EVOLVE-BLOCK-END
