import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = 100.0  # Leave margin for overhead
    
    N = 24
    sqrt_N = math.sqrt(N)
    
    # Precompute FFT frequencies for dense sampling
    # We use a large FFT size for dense sampling to approximate the sup norm
    # 4096 points is usually sufficient for a good initial peak location
    M = 4096 
    theta = 2 * np.pi * np.arange(M) / M
    z = np.exp(1j * theta)
    
    # Function to evaluate |P(z)| on the grid
    def eval_grid(coeffs):
        # coeffs is a numpy array of length N
        # We need to pad coeffs to length M for FFT
        # P(z) = sum c_k z^k. 
        # FFT computes sum c_k exp(2 pi i k n / M).
        # This matches P(z_n) if z_n = exp(2 pi i n / M).
        # Note: FFT order is c_0, c_1, ..., c_{N-1}.
        padded = np.zeros(M, dtype=np.complex128)
        padded[:N] = coeffs
        # Ifft or Fft? 
        # numpy.fft.ifft computes (1/M) sum X_k exp(2 pi i k n / M).
        # numpy.fft.fft computes sum X_k exp(-2 pi i k n / M).
        # We want P(z_n) = sum c_k (exp(2 pi i n / M))^k = sum c_k exp(2 pi i n k / M).
        # This is the inverse FFT formula (without the 1/M factor) if we treat coeffs as the frequency domain?
        # No, standard FFT: X_n = sum_{k=0}^{M-1} x_k e^{-2\pi i n k / M}.
        # If we set x_k = c_k for k<N and 0 otherwise, then X_n = sum_{k=0}^{N-1} c_k e^{-2\pi i n k / M}.
        # This is P(e^{-i \theta_n}) where \theta_n = 2\pi n / M.
        # Since |P(e^{i\theta})| = |P(e^{-i\theta})| for real coefficients, the magnitude is the same.
        # So we can just use fft.
        values = np.fft.fft(padded)
        return np.abs(values)

    def refine_peak(coeffs, peak_idx):
        # Golden section search for the local maximum of |P(z)|
        # around peak_idx
        # The grid point is theta = 2*pi*peak_idx/M
        # We search in a small window around this angle
        
        # Convert index to angle
        center_angle = 2 * np.pi * peak_idx / M
        # Search window: +/- 1/M radians (one grid step)
        # Actually, the peak could be slightly off. A window of +/- 2/M is safe.
        window = 2.0 / M
        a = center_angle - window
        b = center_angle + window
        
        # Ensure a, b are within [0, 2*pi) for consistency, though exp is periodic
        # Golden section constants
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        def f(angle):
            z_val = np.exp(1j * angle)
            # Calculate P(z_val)
            # coeffs is numpy array
            # Use dot product with powers of z_val
            # z_val^k for k=0..N-1
            # This is O(N) per evaluation. N=24, so it's very fast.
            powers = np.power(z_val, np.arange(N))
            val = np.dot(coeffs, powers)
            return np.abs(val)
            
        fc = f(c)
        fd = f(d)
        
        # Iterate
        for _ in range(40):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
                
        # The maximum is at the midpoint of the final interval
        mid = (a + b) / 2.0
        return f(mid)

    def get_sup_norm(coeffs):
        # coeffs: numpy array of +/- 1
        grid_vals = eval_grid(coeffs)
        peak_idx = np.argmax(grid_vals)
        # Refine the peak
        refined_max = refine_peak(coeffs, peak_idx)
        # The grid max might be higher if the peak was exactly on grid? 
        # Refinement should find the true local max.
        # It's possible the global max is at a different peak.
        # For local search, we usually just track the global max.
        # However, for speed, we assume the highest grid peak is the global peak.
        # This is true for most flat polynomials.
        # To be safe, we could check top 3 peaks. But N=24 is small, let's stick to top 1 for speed in inner loop.
        # If we want to be more robust, we can check top 3.
        # Let's check top 3 for the final score calculation, but for the search loop, top 1 is usually fine.
        
        # For the search loop, we want speed.
        # Let's just return the refined value of the highest grid peak.
        return refined_max / sqrt_N

    def try_flip(coeffs, idx):
        # Flip coefficient at idx
        new_coeffs = coeffs.copy()
        new_coeffs[idx] *= -1
        return new_coeffs

    # Initial Candidates
    candidates = []
    
    # 1. Rudin-Shapiro Sequence
    # Generate RS sequence of length 24
    # RS recursion: P_{2n} = P_n Q_n, Q_{2n} = P_n \bar{Q}_n
    # Start P_0 = [1], Q_0 = [1]
    p, q = np.array([1.0]), np.array([1.0])
    while len(p) < N:
        p, q = np.concatenate([p, q]), np.concatenate([p, -q])
    rs_seq = p[:N]
    candidates.append(rs_seq)
    
    # 2. Random Sequences
    # Generate a few random sequences
    for _ in range(10):
        rand_seq = np.random.choice([-1.0, 1.0], N)
        candidates.append(rand_seq)
        
    best_score = float('inf')
    best_coeffs = None
    
    # Local Search Loop
    for init_coeffs in candidates:
        if time.time() - start_time > deadline:
            break
            
        current_coeffs = init_coeffs.copy()
        current_score = get_sup_norm(current_coeffs)
        
        improved = True
        while improved:
            if time.time() - start_time > deadline:
                break
                
            improved = False
            # Try flipping each bit
            for i in range(N):
                if time.time() - start_time > deadline:
                    break
                    
                new_coeffs = try_flip(current_coeffs, i)
                new_score = get_sup_norm(new_coeffs)
                
                if new_score < current_score:
                    current_coeffs = new_coeffs
                    current_score = new_score
                    improved = True
                    # Restart loop? Or continue? 
                    # Hill climbing usually continues until no single flip improves.
                    # Since we changed the state, we should probably restart the scan 
                    # or just continue. Continuing is faster.
                    # However, a different flip might now be better.
                    # Standard steepest descent: pick the best flip.
                    # Standard first improvement: pick the first improving flip.
                    # Let's stick to first improvement for speed, but restart the inner loop 
                    # to ensure we don't miss a better subsequent flip in the same pass?
                    # Actually, just continuing the for loop is fine.
                    # If we want steepest descent, we'd need to evaluate all and pick min.
                    # With N=24, evaluating 24 flips is cheap.
                    # Let's try steepest descent: find the best flip.
                    
            # Optimization: Steepest Descent
            # Instead of breaking on first improvement, let's find the best improvement.
            # But the above code breaks on first improvement.
            # Let's change the inner loop to find the best flip.
            
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current_coeffs.copy()
            
    # Final Polish: Run local search again on the best found to ensure convergence
    if best_coeffs is not None:
        if time.time() - start_time < deadline:
             current_coeffs = best_coeffs.copy()
             current_score = get_sup_norm(current_coeffs)
             improved = True
             while improved:
                 if time.time() - start_time > deadline:
                     break
                 improved = False
                 best_flip_idx = -1
                 best_new_score = current_score
                 
                 # Find the flip that gives the lowest score
                 for i in range(N):
                     if time.time() - start_time > deadline:
                         break
                     new_coeffs = try_flip(current_coeffs, i)
                     new_score = get_sup_norm(new_coeffs)
                     if new_score < best_new_score:
                         best_new_score = new_score
                         best_flip_idx = i
                 
                 if best_flip_idx != -1:
                     current_coeffs = try_flip(current_coeffs, best_flip_idx)
                     current_score = best_new_score
                     improved = True
                     
             if current_score < best_score:
                 best_score = current_score
                 best_coeffs = current_coeffs.copy()

    # Return the coefficients as a list of integers +/- 1
    # The task asks for coefficients in {+1, -1}
    return [int(c) for c in best_coeffs]

# EVOLVE-BLOCK-END
