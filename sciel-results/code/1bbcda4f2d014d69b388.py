import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    N = 24
    M_grid = 2 ** 14  # 16384 points for accurate sup-norm
    
    # Precompute DFT matrix for fast evaluation
    theta = 2 * np.pi * np.arange(M_grid) / M_grid
    E = np.exp(1j * np.outer(theta, np.arange(N)))
    
    def get_supnorm(coeffs):
        """Compute sup norm of polynomial with given coefficients."""
        P = E @ coeffs
        return float(np.max(np.abs(P)))
    
    def local_search(current_c):
        """
        Fast local search using vectorized evaluation of 1-flip moves.
        Uses the property that flipping bit i changes P(z) by -2*c_i*z^i.
        """
        current_val = get_supnorm(current_c)
        best_c = current_c.copy()
        best_val = current_val
        
        # Precompute basis functions for each coefficient
        # basis[i] = z^i on the grid
        basis = E.T  # Shape (N, M_grid), basis[i] is the i-th row
        
        # Iterative improvement
        # We check all 1-flips, then 2-flips if stuck
        max_iters = 20000
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # --- 1-Flip Search ---
            # Compute new supnorms for all 1-flips simultaneously
            # P_new(i) = P - 2*c_i * basis[i]
            # We need max|P_new(i)| for each i
            # This is a bit expensive if done naively (N * M_grid ops).
            # But N=24, M=16384 -> 24*16384 ~ 400k ops. Very fast in numpy.
            
            # P is current evaluation
            P_curr = E @ current_c
            
            best_flip_val = current_val
            best_flip_idx = -1
            
            # Vectorize: for each i, compute max |P_curr - 2*c_i*basis[i]|
            # We can loop over i (only 24 iterations) and use numpy for the rest
            for i in range(N):
                delta = 2 * current_c[i] * basis[i]
                P_new = P_curr - delta
                val = np.max(np.abs(P_new))
                if val < best_flip_val - 1e-12:
                    best_flip_val = val
                    best_flip_idx = i
            
            if best_flip_idx >= 0:
                current_c[best_flip_idx] *= -1
                current_val = best_flip_val
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                continue
            
            # --- 2-Flip Search ---
            # If no 1-flip improves, try 2-flips
            best_2flip_val = current_val
            best_2flip = None
            
            # 24 choose 2 = 276 pairs.
            # For each pair (i,j), P_new = P - 2*c_i*basis[i] - 2*c_j*basis[j]
            # We can precompute the deltas for 1-flips
            deltas = [2 * current_c[i] * basis[i] for i in range(N)]
            
            for i in range(N - 1):
                for j in range(i + 1, N):
                    P_new = P_curr - deltas[i] - deltas[j]
                    val = np.max(np.abs(P_new))
                    if val < best_2flip_val - 1e-12:
                        best_2flip_val = val
                        best_2flip = (i, j)
            
            if best_2flip is not None:
                i, j = best_2flip
                current_c[i] *= -1
                current_c[j] *= -1
                current_val = best_2flip_val
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                continue
            
            # No improvement, local minimum
            break
        
        return best_c, best_val

    # --- Spectral Synthesis Initialization ---
    # Generate multiple diverse starting points using spectral methods
    
    initial_candidates = []
    
    # 1. Golay-Rudin-Shapiro (known good)
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    initial_candidates.append(np.array(p[:N], dtype=float))
    
    # 2. Spectral Synthesis: Inverse FFT of flattened spectrum
    # We want a time-domain sequence that is +/-1.
    # Start with a frequency domain shape that promotes flatness in time?
    # Actually, we want the *time* domain to be +/-1, which implies the *frequency*
    # domain has certain properties.
    # Conversely, if we pick a frequency domain shape (e.g. flat magnitude with random phase),
    # the time domain will be complex. We can round to +/-1.
    
    # Strategy: Pick a target frequency magnitude profile that is "flat" (uniform),
    # assign random phases, inverse FFT, then round to +/-1.
    # Or, pick a frequency profile that corresponds to a "flat" time signal?
    # A flat time signal (constant magnitude) is hard to get directly.
    # Let's try: Generate random phase spectra with uniform magnitude, IFFT, round.
    
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        # Random phase spectrum
        phases = np.random.uniform(0, 2*np.pi, size=N)
        mags = np.ones(N)
        spectrum = mags * np.exp(1j * phases)
        # IFFT
        c_complex = np.fft.ifft(spectrum)
        # Round to +/-1 based on real part or magnitude?
        # Usually, the real part is used for binary sequences.
        # Or take sign of real part.
        c_real = np.real(c_complex)
        c_bin = np.sign(c_real)
        # Ensure no zeros
        c_bin[c_bin == 0] = 1
        initial_candidates.append(c_bin)
        
        # Also try cosine-tapered magnitude
        taper = np.cos(np.pi * np.arange(N) / (N-1)) ** 2
        spectrum2 = taper * np.exp(1j * phases)
        c_complex2 = np.fft.ifft(spectrum2)
        c_real2 = np.real(c_complex2)
        c_bin2 = np.sign(c_real2)
        c_bin2[c_bin2 == 0] = 1
        initial_candidates.append(c_bin2)

    # 3. Random starts
    for _ in range(20):
        if time.time() > deadline - 20:
            break
        c_rand = np.random.choice([-1, 1], size=N).astype(float)
        initial_candidates.append(c_rand)

    # Run local search on all candidates
    best_overall_c = None
    best_overall_val = float('inf')
    
    for c0 in initial_candidates:
        if time.time() > deadline - 5:
            break
        c_best, v_best = local_search(c0)
        if v_best < best_overall_val:
            best_overall_val = v_best
            best_overall_c = c_best
    
    # If we have time, do a few more random restarts
    while time.time() < deadline - 5:
        c_rand = np.random.choice([-1, 1], size=N).astype(float)
        c_best, v_best = local_search(c_rand)
        if v_best < best_overall_val:
            best_overall_val = v_best
            best_overall_c = c_best

    if best_overall_c is None:
        # Fallback
        return [1] * N
        
    return [int(x) for x in best_overall_c]
