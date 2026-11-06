import time
import math
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    N_COEFFS = 24
    sqrt_N = math.sqrt(N_COEFFS)
    
    # Known good starting points
    # Rudin-Shapiro sequence (length 24)
    def get_rs_poly(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    rs_poly = get_rs_poly(N_COEFFS)
    
    # A few other known low-autocorrelation sequences
    known_solutions = [
        rs_poly,
        [1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1],
        [1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1],
        [1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1],
    ]

    # --- Fast Evaluation Machinery ---
    N_FFT = 4096
    
    def evaluate_fast_search(coeffs):
        c_padded = np.zeros(N_FFT, dtype=np.float64)
        c_padded[:N_COEFFS] = coeffs
        vals = np.fft.fft(c_padded)
        mags = np.abs(vals)
        
        max_idx = np.argmax(mags)
        theta_max = 2 * np.pi * max_idx / N_FFT
        
        def local_eval_fast(theta):
            z = complex(math.cos(theta), math.sin(theta))
            val = 0.0 + 0.0j
            z_power = 1.0 + 0.0j
            for k in range(N_COEFFS):
                val += coeffs[k] * z_power
                z_power *= z
            return abs(val)
        
        a = theta_max - math.pi / N_FFT
        b = theta_max + math.pi / N_FFT
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = local_eval_fast(c)
        fd = local_eval_fast(d)
        
        for _ in range(15):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = local_eval_fast(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = local_eval_fast(c)
        
        refined_max = max(fc, fd)
        
        # Check second peak
        max_idx2 = max_idx
        if max_idx > 0:
            mags_copy = mags.copy()
            mags_copy[max_idx] = 0
            max_idx2 = np.argmax(mags_copy)
        else:
            mags_copy = mags.copy()
            mags_copy[0] = 0
            max_idx2 = np.argmax(mags_copy)
            
        if mags[max_idx2] > 0.98 * mags[max_idx]:
            theta_max2 = 2 * np.pi * max_idx2 / N_FFT
            a2 = theta_max2 - math.pi / N_FFT
            b2 = theta_max2 + math.pi / N_FFT
            c2 = b2 - gr * (b2 - a2)
            d2 = a2 + gr * (b2 - a2)
            fc2 = local_eval_fast(c2)
            fd2 = local_eval_fast(d2)
            
            for _ in range(15):
                if fc2 < fd2:
                    a2 = c2
                    c2 = d2
                    fc2 = fd2
                    d2 = a2 + gr * (b2 - a2)
                    fd2 = local_eval_fast(d2)
                else:
                    b2 = d2
                    d2 = c2
                    fd2 = fc2
                    c2 = b2 - gr * (b2 - a2)
                    fc2 = local_eval_fast(c2)
            
            refined_max = max(refined_max, max(fc2, fd2))
            
        return refined_max

    # Initialize best
    best_coeffs = None
    best_score = float('inf')
    
    for sol in known_solutions:
        score = evaluate_fast_search(sol) / sqrt_N
        if score < best_score:
            best_score = score
            best_coeffs = sol[:]
            
    current_coeffs = best_coeffs[:]
    current_score = best_score
    
    # Tabu Search Parameters
    tabu_tenure = 10
    tabu_list = {} # index -> iteration when it expires
    max_iterations = 5000 # Safety cap, though time is the real limit
    
    for iteration in range(max_iterations):
        if time.time() - start_time > deadline - 5:
            break
            
        # Find best non-tabu move
        best_move_idx = -1
        best_move_score = float('inf')
        best_new_coeffs = None
        
        # Try all single flips
        # Optimization: Precompute scores for neighbors? No, too expensive memory-wise or complex to update.
        # Just iterate.
        for i in range(N_COEFFS):
            # Check tabu
            if i in tabu_list and tabu_list[i] > iteration:
                continue
                
            new_coeffs = current_coeffs[:]
            new_coeffs[i] = -new_coeffs[i]
            
            # Aspiration criterion: If the move leads to a solution better than best_score, allow it even if tabu
            # But we already skipped tabu moves above unless we add aspiration logic.
            # Let's stick to simple tabu first.
            
            new_score = evaluate_fast_search(new_coeffs) / sqrt_N
            
            if new_score < best_move_score:
                best_move_score = new_score
                best_move_idx = i
                best_new_coeffs = new_coeffs
        
        # If no non-tabu move found, clear tabu or pick best tabu move?
        # Usually, if no moves, we are stuck. Pick best tabu move or kick.
        if best_move_idx == -1:
            # Pick a random kick
            kick_size = random.choice([2, 3, 4])
            kick_indices = random.sample(range(N_COEFFS), kick_size)
            for k in kick_indices:
                current_coeffs[k] = -current_coeffs[k]
            current_score = evaluate_fast_search(current_coeffs) / sqrt_N
            # Clear tabu list on kick
            tabu_list.clear()
            continue
            
        # Apply the best move
        current_coeffs = best_new_coeffs
        current_score = best_move_score
        
        # Update Tabu List
        tabu_list[best_move_idx] = iteration + tabu_tenure
        
        # Update Global Best
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current_coeffs[:]
            
        # Optional: If stuck for a while (no improvement in best_score), increase kick frequency or tenure?
        # For now, keep it simple.
        
    # Final Polish with Precise Evaluation
    # The search used `evaluate_fast_search` (15 GS iters).
    # We need to verify the final result with `evaluate_fast` (30 GS iters) or similar.
    # But the evaluator does the precise calculation.
    # We just need to ensure we haven't settled for a local minimum that looks good at 1e-6 but is worse at 1e-9.
    # However, the landscape is smooth.
    
    # Let's do a few rounds of precise hill climbing from the best found
    candidate = best_coeffs[:]
    precise_improved = True
    while precise_improved and time.time() - start_time < deadline - 2:
        precise_improved = False
        for i in range(N_COEFFS):
            new_coeffs = candidate[:]
            new_coeffs[i] = -new_coeffs[i]
            # Use a higher precision eval for the final polish
            # We can reuse the logic from the original `evaluate_fast` but inline it for speed?
            # Or just call a high-precision version.
            
            # Let's define a high precision evaluator here
            def evaluate_precise(coeffs):
                N_FFT_HP = 8192
                c_padded = np.zeros(N_FFT_HP, dtype=np.float64)
                c_padded[:N_COEFFS] = coeffs
                vals = np.fft.fft(c_padded)
                mags = np.abs(vals)
                max_idx = np.argmax(mags)
                theta_max = 2 * np.pi * max_idx / N_FFT_HP
                
                def local_eval_hp(theta):
                    z = complex(math.cos(theta), math.sin(theta))
                    val = 0.0 + 0.0j
                    z_power = 1.0 + 0.0j
                    for k in range(N_COEFFS):
                        val += coeffs[k] * z_power
                        z_power *= z
                    return abs(val)
                
                a = theta_max - math.pi / N_FFT_HP
                b = theta_max + math.pi / N_FFT_HP
                gr = (math.sqrt(5) - 1) / 2
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                fc = local_eval_hp(c)
                fd = local_eval_hp(d)
                
                for _ in range(30):
                    if fc < fd:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        fd = local_eval_hp(d)
                    else:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        fc = local_eval_hp(c)
                
                refined_max = max(fc, fd)
                
                # Check second peak
                max_idx2 = max_idx
                if max_idx > 0:
                    mags_copy = mags.copy()
                    mags_copy[max_idx] = 0
                    max_idx2 = np.argmax(mags_copy)
                else:
                    mags_copy = mags.copy()
                    mags_copy[0] = 0
                    max_idx2 = np.argmax(mags_copy)
                    
                if mags[max_idx2] > 0.98 * mags[max_idx]:
                    theta_max2 = 2 * np.pi * max_idx2 / N_FFT_HP
                    a2 = theta_max2 - math.pi / N_FFT_HP
                    b2 = theta_max2 + math.pi / N_FFT_HP
                    c2 = b2 - gr * (b2 - a2)
                    d2 = a2 + gr * (b2 - a2)
                    fc2 = local_eval_hp(c2)
                    fd2 = local_eval_hp(d2)
                    
                    for _ in range(30):
                        if fc2 < fd2:
                            a2 = c2
                            c2 = d2
                            fc2 = fd2
                            d2 = a2 + gr * (b2 - a2)
                            fd2 = local_eval_hp(d2)
                        else:
                            b2 = d2
                            d2 = c2
                            fd2 = fc2
                            c2 = b2 - gr * (b2 - a2)
                            fc2 = local_eval_hp(c2)
                    
                    refined_max = max(refined_max, max(fc2, fd2))
                    
                return refined_max

            new_score = evaluate_precise(new_coeffs) / sqrt_N
            if new_score < best_score - 1e-12:
                candidate = new_coeffs
                best_score = new_score
                best_coeffs = candidate[:]
                precise_improved = True

    return best_coeffs
# EVOLVE-BLOCK-END
