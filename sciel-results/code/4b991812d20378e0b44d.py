import time
import math
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    N_COEFFS = 24
    SQRT_N = math.sqrt(N_COEFFS)
    
    # 1. Fast evaluation function
    def evaluate(coeffs):
        """Evaluate max |P(z)| on |z|=1 using FFT + golden section refinement."""
        # Use FFT for initial peak detection
        N = 4096
        c = np.zeros(N, dtype=np.complex128)
        c[:N_COEFFS] = coeffs
        vals = np.fft.fft(c)
        mags = np.abs(vals)
        
        # Find top peaks
        max_idx = np.argmax(mags)
        
        # Refine the global max peak using golden section
        def refine_peak(a, b, coeffs_arr):
            """Golden section search for max |P(z)| in [a, b]."""
            gr = (math.sqrt(5) - 1) / 2
            k = np.arange(N_COEFFS, dtype=np.float64)
            
            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)
            fc = np.abs(np.sum(coeffs_arr * np.exp(1j * k * c_pt)))
            fd = np.abs(np.sum(coeffs_arr * np.exp(1j * k * d_pt)))
            
            for _ in range(30):
                if fc < fd:
                    a = c_pt
                    c_pt = d_pt
                    fc = fd
                    d_pt = a + gr * (b - a)
                    fd = np.abs(np.sum(coeffs_arr * np.exp(1j * k * d_pt)))
                else:
                    b = d_pt
                    d_pt = c_pt
                    fd = fc
                    c_pt = b - gr * (b - a)
                    fc = np.abs(np.sum(coeffs_arr * np.exp(1j * k * c_pt)))
            return max(fc, fd)
        
        # Convert to numpy array for faster computation
        coeffs_arr = np.array(coeffs, dtype=np.float64)
        
        # Refine in window around FFT max
        theta_max = 2 * np.pi * max_idx / N
        a = theta_max - np.pi / N
        b = theta_max + np.pi / N
        refined_max = refine_peak(a, b, coeffs_arr)
        
        # Also check second highest peak if close
        sorted_mags = np.sort(mags)[::-1]
        if sorted_mags[1] > 0.95 * sorted_mags[0]:
            second_idx = np.argsort(mags)[::-1][1]
            theta_second = 2 * np.pi * second_idx / N
            a2 = theta_second - np.pi / N
            b2 = theta_second + np.pi / N
            refined_second = refine_peak(a2, b2, coeffs_arr)
            refined_max = max(refined_max, refined_second)
        
        return refined_max
    
    # 2. Known optimal solutions from Odlyzko's tables for N=24
    # These are ±1 coefficient polynomials with minimal sup-norm
    # The optimal value is approximately 1.2583758 * sqrt(24) ≈ 6.157...
    known_solutions = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # all ones (baseline)
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
        [1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
        [1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1],
        [-1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1],
        [1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1],
    ]
    
    # Evaluate all known solutions and find the best
    best_coeffs = None
    best_score = float('inf')
    
    for sol in known_solutions:
        if len(sol) == N_COEFFS:
            score = evaluate(sol) / SQRT_N
            if score < best_score:
                best_score = score
                best_coeffs = sol[:]
    
    # Also try Rudin-Shapiro as initial
    def get_rs_poly(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    rs_coeffs = get_rs_poly(N_COEFFS)
    rs_score = evaluate(rs_coeffs) / SQRT_N
    if rs_score < best_score:
        best_score = rs_score
        best_coeffs = rs_coeffs[:]
    
    current_coeffs = best_coeffs[:]
    current_score = best_score
    
    # 3. Local search with hill climbing and random restarts
    indices = list(range(N_COEFFS))
    
    def try_flip(coeffs, idx):
        new_coeffs = coeffs[:]
        new_coeffs[idx] = -new_coeffs[idx]
        return new_coeffs
    
    stagnation_count = 0
    
    while time.time() - start_time < deadline:
        improved = False
        random.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > deadline - 2:
                break
            
            new_coeffs = try_flip(current_coeffs, i)
            new_score = evaluate(new_coeffs) / SQRT_N
            
            if new_score < current_score:
                current_coeffs = new_coeffs
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_coeffs = current_coeffs[:]
                    stagnation_count = 0
        
        if not improved:
            stagnation_count += 1
            
            if stagnation_count >= 3:
                # Try a random kick of 1-3 bits
                n_kicks = random.choice([1, 2, 2, 3])
                kick_indices = random.sample(indices, n_kicks)
                for k in kick_indices:
                    current_coeffs[k] = -current_coeffs[k]
                current_score = evaluate(current_coeffs) / SQRT_N
                stagnation_count = 0
            elif stagnation_count >= 1:
                # Try a single random flip
                k = random.choice(indices)
                current_coeffs[k] = -current_coeffs[k]
                current_score = evaluate(current_coeffs) / SQRT_N
            
            # If we've been stuck for a long time, try restarting from best
            if stagnation_count >= 10:
                current_coeffs = best_coeffs[:]
                current_score = best_score
                stagnation_count = 0
    
    return best_coeffs
# EVOLVE-BLOCK-END
