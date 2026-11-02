"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def _build_signal_from_heights(heights, K):
    """
    Construct a signal of length N from K equal-width step heights.
    Assumes N is divisible by K.
    """
    W = N // K
    # Tile the heights to fill the signal
    # heights is shape (K,), we need shape (N,)
    # Each height repeats W times
    sig = np.repeat(heights, W)
    return sig

def _compute_score_from_signal(sig):
    """Compute C3' = max|autoconv| / (integral)^2 for a signal."""
    integral = np.sum(sig)
    if abs(integral) < 1e-12:
        return float('inf')
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    f = np.fft.rfft(sig, n_fft)
    ac = np.fft.irfft(f * np.conj(f), n_fft)[:2*N-1]
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)

def _objective_for_K(heights, K):
    """
    Objective function for scipy.optimize.
    heights: array of length K.
    We normalize the signal such that the integral is 1 to remove scale invariance.
    Then we minimize max|autoconv|.
    """
    W = N // K
    # Construct signal
    sig = np.repeat(heights, W)
    integral = np.sum(sig)
    if abs(integral) < 1e-12:
        return float('inf')
    # Normalize to unit integral
    sig_norm = sig / integral
    
    # Compute autoconvolution
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    f = np.fft.rfft(sig_norm, n_fft)
    ac = np.fft.irfft(f * np.conj(f), n_fft)[:2*N-1]
    max_abs = np.max(np.abs(ac))
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with the theoretical global minimum: constant signal
    # This corresponds to K=1
    best_sig = np.ones(N)
    best_score = _compute_score_from_signal(best_sig)
    
    # Search over different numbers of equal-width steps K
    # K must divide N
    divisors = []
    for k in range(2, int(np.sqrt(N)) + 1):
        if N % k == 0:
            divisors.append(k)
            if k != N // k:
                divisors.append(N // k)
    divisors.sort()
    
    # Limit search to reasonable K values to stay within time budget
    # Larger K means more parameters and likely higher score (worse)
    # We check small K first as they are structurally simpler
    max_K_to_check = 20
    
    for K in divisors:
        if K > max_K_to_check:
            break
        if time.time() > deadline - 10:
            break
            
        # Initial guess for heights: alternating signs to try to cancel out autoconv
        # Or just random? Random is safer to avoid local minima.
        # But we know constant is best. We are just verifying no better exists.
        # For K>1, constant is not allowed (heights must be same? No, heights are variables).
        # If all heights are equal, it's the constant signal.
        # So we are searching for non-constant configurations.
        
        # Use a random initialization for heights
        np.random.seed(42) # Fixed seed for reproducibility within the run
        h0 = np.random.randn(K)
        # Ensure non-zero sum
        while abs(np.sum(h0)) < 1e-6:
            h0 = np.random.randn(K)
        
        try:
            # Optimize heights
            # Method: Nelder-Mead is derivative-free and robust for small K
            res = scipy.optimize.minimize(
                _objective_for_K, 
                h0, 
                args=(K,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.success:
                optimal_heights = res.x
                # Reconstruct signal
                candidate_sig = _build_signal_from_heights(optimal_heights, K)
                candidate_score = _compute_score_from_signal(candidate_sig)
                
                if candidate_score < best_score - 1e-12:
                    best_score = candidate_score
                    best_sig = candidate_sig
                    
        except Exception:
            pass

    # Final check: if we haven't improved, best_sig is still np.ones(N)
    return best_sig
# EVOLVE-BLOCK-END
