import time
import math
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # 1. Fast evaluation function using FFT
    def evaluate(coeffs):
        # coeffs is a list of +1/-1
        # P(z) = sum c_k z^k
        # We sample on the unit circle at N points
        N = 2048  # Resolution for FFT
        # Pad coefficients to length N
        c = np.zeros(N, dtype=np.complex128)
        c[:len(coeffs)] = coeffs
        # Evaluate at roots of unity
        # P(w_j) where w_j = exp(2*pi*i*j/N)
        # This is exactly the DFT of the padded sequence if we consider c_k as time domain
        # Note: DFT definition X_k = sum x_n e^{-i 2 pi n k / N}
        # Our P(z) = sum c_k z^k. If z = e^{i theta}, P(e^{i theta}) = sum c_k e^{i k theta}.
        # Let theta_j = 2 pi j / N.
        # P(e^{i theta_j}) = sum c_k e^{i 2 pi k j / N}
        # This is the Inverse DFT (up to scaling) or DFT with negative frequency.
        # np.fft.fft(c) gives X_k = sum c_n e^{-i 2 pi n k / N}.
        # So we need the conjugate of fft(c) or fft of reversed/conjugated?
        # Actually, P(e^{i 2 pi j / N}) = sum c_k e^{i 2 pi k j / N} = conj( sum c_k e^{-i 2 pi k j / N} )
        # If c is real, sum c_k e^{-i ...} is fft(c)[j].
        # So P(e^{i theta_j}) = conj(fft(c)[j])
        # Magnitude |P| = |fft(c)[j]|
        vals = np.fft.fft(c)
        mags = np.abs(vals)
        
        # Find the global max and refine
        max_idx = np.argmax(mags)
        max_val = mags[max_idx]
        
        # Refine around the peak using golden section
        # The peak is between theta_{max_idx-1} and theta_{max_idx+1} roughly?
        # Better: refine in the interval [theta_{max_idx-1}, theta_{max_idx+1}] or just around max_idx.
        # Since the function is smooth, we can refine in a small window around the max.
        # Let's refine in [ (max_idx-1)*2pi/N, (max_idx+1)*2pi/N ]
        # Actually, just refine in [theta_{max_idx-1}, theta_{max_idx+1}]
        # But we need to be careful with wrap-around.
        
        # A simpler robust refinement:
        # Check the top 3 peaks from FFT and refine each.
        # For speed, we only refine the global max.
        
        # Golden section search for max |P(z)| in [a, b]
        def refine_peak(a, b):
            # |P(z)| is not necessarily unimodal in a small window if there are ripples,
            # but for low degree polynomials (24) and high N (2048), it's likely unimodal locally.
            # We'll do a few iterations.
            gr = (math.sqrt(5) - 1) / 2  # 0.618...
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = np.abs(np.sum(coeffs * np.exp(1j * np.arange(len(coeffs)) * c)))
            fd = np.abs(np.sum(coeffs * np.exp(1j * np.arange(len(coeffs)) * d)))
            
            for _ in range(20):
                if fc < fd:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = np.abs(np.sum(coeffs * np.exp(1j * np.arange(len(coeffs)) * d)))
                else:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = np.abs(np.sum(coeffs * np.exp(1j * np.arange(len(coeffs)) * c)))
            return max(fc, fd)
        
        # Determine interval for refinement
        # The FFT bin width is 2*pi/N
        # We refine in the window centered at the max bin.
        theta_max = 2 * np.pi * max_idx / N
        # Handle wrap around
        # Interval: [theta_max - pi/N, theta_max + pi/N]
        a = theta_max - np.pi / N
        b = theta_max + np.pi / N
        
        refined_max = refine_peak(a, b)
        
        # Also check if the max was near 0 or 2pi, might need wrap-around check
        # But usually the FFT max is well isolated.
        # To be safe, we can also check the second highest peak if it's close.
        # But for now, let's trust the top peak refinement.
        
        return refined_max

    # 2. Initial Solution: Rudin-Shapiro
    def get_rs_poly(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    N_COEFFS = 24
    current_coeffs = get_rs_poly(N_COEFFS)
    current_score = evaluate(current_coeffs) / math.sqrt(N_COEFFS)
    best_coeffs = current_coeffs[:]
    best_score = current_score
    
    # 3. Local Search (Hill Climbing with Random Restarts)
    # We flip one coefficient at a time.
    # If it improves, accept.
    # If it stagnates, we can try a random flip (simulated annealing lite) or restart.
    # Given the small dimension (24), simple hill climbing with random restarts is effective.
    
    # To speed up evaluation, we can precompute powers of z? No, z changes in refinement.
    # But for the FFT part, it's fast. The refinement is the bottleneck.
    # Let's optimize the evaluation:
    # We can skip refinement if the FFT max is far from the previous best? No, we need accuracy.
    
    # Optimization: 
    # The FFT gives a good lower bound on the max (actually it's the max at discrete points, which is <= true max).
    # The refinement gives the true max in that local window.
    # We assume the global max is in the window of the FFT max.
    
    # Let's run hill climbing.
    # Budget: 115s.
    # Each eval: FFT (2048) is ~0.1ms. Refinement: 20 steps * 24 mults = 480 ops. Very fast.
    # We can do many iterations.
    
    # Strategy: 
    # 1. Try flipping each coefficient. If score improves, keep it.
    # 2. Repeat until no improvement.
    # 3. If stuck, perform a random flip (or a few) to escape local minimum.
    
    def try_flip(coeffs, idx):
        new_coeffs = coeffs[:]
        new_coeffs[idx] = -new_coeffs[idx]
        return new_coeffs
    
    # Precompute indices
    indices = list(range(N_COEFFS))
    
    while time.time() - start_time < deadline:
        improved = False
        # Shuffle indices to avoid bias
        random.shuffle(indices)
        for i in indices:
            new_coeffs = try_flip(current_coeffs, i)
            new_score = evaluate(new_coeffs) / math.sqrt(N_COEFFS)
            if new_score < current_score:
                current_coeffs = new_coeffs
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_coeffs = current_coeffs[:]
        
        if not improved:
            # Stagnation. 
            # Option 1: Random restart (reset to RS or random)
            # Option 2: Random kick (flip 1-3 random bits)
            # Let's do a random kick to escape local minima.
            # Flip 2 random bits
            kick_indices = random.sample(indices, 2)
            for k in kick_indices:
                current_coeffs[k] = -current_coeffs[k]
            current_score = evaluate(current_coeffs) / math.sqrt(N_COEFFS)
            
            # If the kick was bad, we might want to revert? 
            # No, we just continue searching from this new point.
            # But if we go too far, we might lose the best. We keep best_coeffs.
            
            # Check time
            if time.time() - start_time > deadline - 5:
                break
                
    return best_coeffs
# EVOLVE-BLOCK-END
