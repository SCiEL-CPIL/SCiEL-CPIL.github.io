"""
Strategy: Deterministic Low-Discrepancy Sequence Initialization + Local Polish

This approach replaces the random initialization of the seed program with a
Kronecker sequence-based sign assignment. 
1. We generate a sequence of fractional parts {i * alpha} for i in [0, N).
2. We map these to signs {-1, 1} using a threshold parameter 'bias' to control
   the integral I. A higher bias yields more +1s, increasing I^2.
3. The autoconvolution of such quasi-random sequences typically has lower peaks
   than purely random sequences for the same bias, due to uniform distribution
   of differences.
4. We optimize the 'bias' parameter and then perform a rapid local search
   (flipping individual bits) to further reduce the max absolute autoconvolution.

This is distinct from random search because the initial structure is highly
correlated (in a low-discrepancy sense) rather than independent, which often
leads to better global minima in spectral optimization problems.
"""

import numpy as np
import time

N = 400

def _autoconv_max_abs(f):
    """
    Computes the maximum absolute value of the linear autoconvolution of f.
    Uses FFT for efficiency.
    """
    # Pad to next power of 2 for efficient FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=n_fft)
    # Power spectrum is |F|^2. Inverse FFT gives circular autocorrelation.
    # Since we padded to >= 2N-1, the linear autocorrelation is embedded 
    # without wrap-around aliasing in the central part.
    C = np.fft.irfft(F * F.conj(), n=n_fft)
    
    # The linear autocorrelation lags k = 0, ..., N-1 are at indices 0, ..., N-1
    # and k = -(N-1), ..., -1 are at indices N, ..., 2N-2 (due to FFT shift properties 
    # in this specific padding setup, or simply check the first N elements which 
    # correspond to positive lags 0..N-1. By symmetry, max abs is same).
    # Actually, irfft returns the first half. 
    # For real input, C[0:N] contains lags 0 to N-1.
    # C[N:2N-1] contains lags -(N-1) to -1? 
    # Let's verify: 
    # conv(f, f[::-1]) is correlation. 
    # np.fft.irfft(|fft(f, n_fft)|^2) gives the circular autocorrelation.
    # With zero padding to n_fft >= 2N-1, the circular autocorrelation matches 
    # linear autocorrelation for lags |k| < N.
    # The output array C has length n_fft//2 + 1.
    # Indices 0..N-1 correspond to lags 0..N-1.
    # Indices n_fft//2 .. n_fft//2 + N-1 correspond to negative lags? 
    # No, standard irfft output is ordered by frequency. 
    # Let's just take the max over the valid range.
    # Valid lags are 0 to N-1.
    return np.max(np.abs(C[:N]))

def _score(f):
    """
    Computes the ratio max|c(k)| / I^2.
    """
    I = np.sum(f)
    if I == 0:
        return np.inf
    max_c = _autoconv_max_abs(f)
    return max_c / (I * I)

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin
    
    # 1. Generate Kronecker sequence
    # Use a specific irrational number. The golden ratio conjugate (1-sqrt(5))/2 
    # or sqrt(2) are good choices. 
    # Let's try a few alpha values and pick the best initial bias.
    
    best_f = None
    best_score = np.inf
    
    # Candidate alphas
    alphas = [
        np.sqrt(2) - 1,       # ~0.414
        (1 - np.sqrt(5)) / 2, # ~-0.618 (use abs or mod 1)
        np.pi - 3,            # ~0.141
        1/np.e,               # ~0.368
        np.sqrt(3) - 1        # ~0.732
    ]
    
    # We will sweep a range of 'bias' thresholds for each alpha.
    # bias in [0.3, 0.7] maps to integral I roughly N*(2*bias - 1) if uniform.
    # Actually, if threshold is T, P(x < T) = T. 
    # If we set sign = 1 if frac < T, else -1.
    # Then expected number of +1s is N*T.
    # I = sum(f) = N*T - N*(1-T) = N*(2T - 1).
    # To maximize I^2, we want T close to 0 or 1. But that makes f constant, 
    # which has high autoconvolution.
    # Trade-off: We want T such that I is large enough but peaks are low.
    # Let's test T in [0.55, 0.95] and [0.05, 0.45].
    
    biases = np.linspace(0.55, 0.95, 10)
    biases_neg = np.linspace(0.05, 0.45, 10)
    all_biases = np.concatenate([biases, biases_neg])
    
    for alpha in alphas:
        # Compute fractional parts
        # i * alpha mod 1
        idx = np.arange(N)
        fracs = np.mod(idx * alpha, 1.0)
        
        for T in all_biases:
            # Construct f
            # f[i] = 1 if fracs[i] < T else -1
            f = np.where(fracs < T, 1, -1).astype(np.int8)
            
            # Quick score check
            s = _score(f)
            if s < best_score:
                best_score = s
                best_f = f.copy()
        
        # Check time
        if time.time() > deadline - 60:
            break
    
    if best_f is None:
        # Fallback to all ones
        best_f = np.ones(N, dtype=np.int8)
        best_score = _score(best_f)
    
    # 2. Local Polish
    # Now refine the best candidate found by the deterministic construction.
    # Use a greedy hill-climbing approach with random restarts of the search order
    # or just a single pass of random flips accepting improvements.
    
    current_f = best_f
    current_score = best_score
    
    rng = np.random.default_rng(12345)
    
    # Precompute FFT size
    n_fft = 1 << (2 * N - 1).bit_length()
    
    # Helper for fast score update? 
    # Full FFT is fast enough for N=400 (size 1024). ~10k ops.
    # 120s allows for ~100k-1000k FFTs. 
    # Let's do up to 5000 flips.
    
    max_flips = 5000
    no_improve_count = 0
    patience = 200
    
    for i in range(max_flips):
        if time.time() > deadline - 5:
            break
        
        # Pick random index
        idx = rng.integers(0, N)
        
        # Flip
        current_f[idx] *= -1
        
        # Compute new score
        new_score = _score(current_f)
        
        if new_score < current_score:
            current_score = new_score
            best_f = current_f.copy()
            best_score = current_score
            no_improve_count = 0
        else:
            # Revert
            current_f[idx] *= -1
            no_improve_count += 1
            
            # If stuck for a while, try a "kick" (flip 2-3 random bits)
            if no_improve_count > patience:
                no_improve_count = 0
                # Kick: flip 3 random bits
                for _ in range(3):
                    k_idx = rng.integers(0, N)
                    current_f[k_idx] *= -1
                kick_score = _score(current_f)
                if kick_score < current_score:
                    current_score = kick_score
                    best_f = current_f.copy()
                    best_score = current_score
                else:
                    # Revert kick
                    for _ in range(3):
                        k_idx = rng.integers(0, N) # This is wrong, we need the exact indices
                        # We didn't store them. Let's just revert the current_f to best_f
                    current_f = best_f.copy()
                    current_score = best_score

    return best_f

# EVOLVE-BLOCK-END
