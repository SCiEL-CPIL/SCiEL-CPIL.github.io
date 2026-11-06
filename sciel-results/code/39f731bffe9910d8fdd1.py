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
    
    # Precompute powers of z for evaluation
    k_arr = np.arange(N_COEFFS, dtype=np.float64)
    
    # 1. Fast evaluation function (for search)
    def evaluate_fast(coeffs):
        """Fast evaluation using dense sampling + few refinement steps."""
        coeffs_arr = np.array(coeffs, dtype=np.float64)
        
        # Dense sampling with 2048 points
        N = 2048
        thetas = 2 * np.pi * np.arange(N) / N
        # Compute P(z) for all thetas at once using matrix multiplication
        z = np.exp(1j * np.outer(k_arr, thetas))  # shape (N_COEFFS, N)
        vals = z.T @ coeffs_arr  # shape (N,)
        mags = np.abs(vals)
        
        max_idx = np.argmax(mags)
        theta_max = thetas[max_idx]
        
        # Quick golden section refinement (15 iterations)
        gr = (math.sqrt(5) - 1) / 2
        a = theta_max - 2 * np.pi / N
        b = theta_max + 2 * np.pi / N
        
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        fc = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * c_pt)))
        fd = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * d_pt)))
        
        for _ in range(15):
            if fc < fd:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * d_pt)))
            else:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * c_pt)))
        
        refined_max = max(fc, fd)
        
        # Check if second peak is close
        sorted_mags = np.sort(mags)[::-1]
        if sorted_mags[1] > 0.95 * sorted_mags[0]:
            second_idx = np.argsort(mags)[::-1][1]
            theta_second = thetas[second_idx]
            a2 = theta_second - 2 * np.pi / N
            b2 = theta_second + 2 * np.pi / N
            c2 = b2 - gr * (b2 - a2)
            d2 = a2 + gr * (b2 - a2)
            fc2 = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * c2)))
            fd2 = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * d2)))
            for _ in range(15):
                if fc2 < fd2:
                    a2 = c2
                    c2 = d2
                    fc2 = fd2
                    d2 = a2 + gr * (b2 - a2)
                    fd2 = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * d2)))
                else:
                    b2 = d2
                    d2 = c2
                    fd2 = fc2
                    c2 = b2 - gr * (b2 - a2)
                    fc2 = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * c2)))
            refined_second = max(fc2, fd2)
            refined_max = max(refined_max, refined_second)
        
        return refined_max
    
    # 2. Precise evaluation function (for final polish)
    def evaluate_precise(coeffs):
        """High-precision evaluation with more refinement iterations."""
        coeffs_arr = np.array(coeffs, dtype=np.float64)
        
        # Dense sampling with 4096 points
        N = 4096
        thetas = 2 * np.pi * np.arange(N) / N
        z = np.exp(1j * np.outer(k_arr, thetas))
        vals = z.T @ coeffs_arr
        mags = np.abs(vals)
        
        max_idx = np.argmax(mags)
        theta_max = thetas[max_idx]
        
        # High-precision golden section refinement (40 iterations)
        gr = (math.sqrt(5) - 1) / 2
        a = theta_max - np.pi / N
        b = theta_max + np.pi / N
        
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        fc = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * c_pt)))
        fd = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * d_pt)))
        
        for _ in range(40):
            if fc < fd:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * d_pt)))
            else:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * c_pt)))
        
        refined_max = max(fc, fd)
        
        # Check top 3 peaks
        top_indices = np.argsort(mags)[::-1][:3]
        for idx in top_indices[1:]:
            theta_peak = thetas[idx]
            a2 = theta_peak - np.pi / N
            b2 = theta_peak + np.pi / N
            c2 = b2 - gr * (b2 - a2)
            d2 = a2 + gr * (b2 - a2)
            fc2 = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * c2)))
            fd2 = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * d2)))
            for _ in range(40):
                if fc2 < fd2:
                    a2 = c2
                    c2 = d2
                    fc2 = fd2
                    d2 = a2 + gr * (b2 - a2)
                    fd2 = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * d2)))
                else:
                    b2 = d2
                    d2 = c2
                    fd2 = fc2
                    c2 = b2 - gr * (b2 - a2)
                    fc2 = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * c2)))
            refined_peak = max(fc2, fd2)
            refined_max = max(refined_max, refined_peak)
        
        return refined_max
    
    # 3. Known candidate solutions
    known_solutions = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
        [1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
        [1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1],
        [-1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1],
        [1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1],
        # Additional candidates with better balance
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
    ]
    
    # Evaluate all known solutions and find the best
    best_coeffs = None
    best_score = float('inf')
    
    for sol in known_solutions:
        if len(sol) == N_COEFFS:
            score = evaluate_fast(sol) / SQRT_N
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
    rs_score = evaluate_fast(rs_coeffs) / SQRT_N
    if rs_score < best_score:
        best_score = rs_score
        best_coeffs = rs_coeffs[:]
    
    current_coeffs = best_coeffs[:]
    current_score = best_score
    
    # 4. Local search with hill climbing
    indices = list(range(N_COEFFS))
    stagnation_count = 0
    
    while time.time() - start_time < deadline - 5:  # Leave 5s for final polish
        improved = False
        random.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > deadline - 7:
                break
            
            new_coeffs = current_coeffs[:]
            new_coeffs[i] = -new_coeffs[i]
            new_score = evaluate_fast(new_coeffs) / SQRT_N
            
            if new_score < current_score - 1e-10:
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
                current_score = evaluate_fast(current_coeffs) / SQRT_N
                stagnation_count = 0
            elif stagnation_count >= 1:
                # Try a single random flip
                k = random.choice(indices)
                current_coeffs[k] = -current_coeffs[k]
                current_score = evaluate_fast(current_coeffs) / SQRT_N
            
            # If we've been stuck for a long time, try restarting from best
            if stagnation_count >= 10:
                current_coeffs = best_coeffs[:]
                current_score = best_score
                stagnation_count = 0
    
    # 5. Final high-precision polish: try all single flips on best solution
    if time.time() - start_time < deadline - 2:
        for i in range(N_COEFFS):
            if time.time() - start_time > deadline - 1:
                break
            new_coeffs = best_coeffs[:]
            new_coeffs[i] = -new_coeffs[i]
            new_score = evaluate_precise(new_coeffs) / SQRT_N
            if new_score < best_score:
                best_score = new_score
                best_coeffs = new_coeffs[:]
    
    # Final evaluation with precise method
    final_score = evaluate_precise(best_coeffs) / SQRT_N
    if final_score < best_score:
        best_score = final_score
    
    return best_coeffs
# EVOLVE-BLOCK-END
