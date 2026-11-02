"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_peak_sidelobe(x):
    """
    Compute the maximum absolute value of the autocorrelation 
    for non-zero lags.
    x: array of +/- 1 of length N.
    """
    # Autocorrelation via FFT
    # r[k] = sum_{i=0}^{N-k-1} x[i] x[i+k]
    # Using circular autocorrelation via FFT is faster but requires zero-padding 
    # or careful interpretation. 
    # For linear autocorrelation, we can use:
    # R = ifft(fft(x, 2*N) * conj(fft(x, 2*N)))
    # The linear autocorrelation is in the first N elements of R (and the rest are wrap-around).
    
    n_fft = 2 * N
    F = np.fft.fft(x, n_fft)
    R = np.fft.ifft(F * np.conj(F)).real
    # R[0] is the energy (N). R[1:N] are lags 1 to N-1.
    # R[N+1:] are negative lags.
    # We want max |R[k]| for k != 0.
    # Due to symmetry, max(R[1:N], R[N+1:2*N])
    # Actually, R is real and symmetric: R[k] = R[-k].
    # So we just need to check R[1:N].
    sidelobes = R[1:N]
    return np.max(np.abs(sidelobes))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial random sequence
    x = np.random.choice([-1, 1], size=N)
    best_x = x.copy()
    best_score = _compute_peak_sidelobe(best_x)
    
    # Simulated Annealing parameters
    T = 1.0 # Initial temperature
    T_min = 1e-5
    cooling = 0.999
    
    # We will try to flip bits. 
    # To be fast, we can precompute the effect of a flip on the autocorrelation?
    # Flipping x[i] -> -x[i] changes the autocorrelation r[k] by:
    # r_new[k] = r_old[k] - 2 * x[i] * (x[i+k] + x[i-k])  (with boundary handling)
    # This is O(N) per flip. FFT is O(N log N).
    # For N=400, N log N ~ 400 * 9 ~ 3600 ops. N ~ 400 ops.
    # The incremental update is faster!
    
    # Let's implement the incremental update for speed.
    # r[k] = sum_{j} x[j] x[j+k]
    # If we flip x[i], it affects terms where j=i or j+k=i.
    # j=i -> x[i] x[i+k] becomes -x[i] x[i+k]. Change: -2 x[i] x[i+k].
    # j+k=i -> j = i-k. x[i-k] x[i] becomes -x[i-k] x[i]. Change: -2 x[i] x[i-k].
    # So delta_r[k] = -2 * x[i] * (x[i+k] + x[i-k]) for valid indices.
    
    # We maintain the autocorrelation array R.
    # R is length 2*N (for FFT compatibility) or just N+1 for lags 0..N-1?
    # Let's keep the full circular autocorrelation array from FFT for simplicity, 
    # but update it manually? 
    # Actually, maintaining the linear autocorrelation array `r` of length N (lags 0..N-1) 
    # is easier to update and check.
    # r[k] for k=0..N-1.
    
    def init_r(x):
        # Compute linear autocorrelation
        r = np.zeros(N)
        for k in range(N):
            # r[k] = dot(x[:N-k], x[k:])
            r[k] = np.dot(x[:N-k], x[k:])
        return r
    
    r = init_r(x)
    
    def update_r_after_flip(x, i, r):
        # x[i] is flipped. Old value was x[i], new is -x[i].
        # Let old_val = x[i].
        # The term x[i] appears in r[k] if:
        # 1. j = i, so i < N-k and i+k < N => k < N-i and k > -i (always true for k>=0)
        #    Specifically, the term is x[i] * x[i+k]. This exists if i+k < N.
        #    So for k in 0..N-i-1, r[k] includes x[i]*x[i+k].
        #    Change: -2 * x[i] * x[i+k]
        # 2. j+k = i, so j = i-k. j >= 0 => i-k >= 0 => k <= i.
        #    Also j < N-k => i-k < N-k => i < N (always true).
        #    The term is x[i-k] * x[i].
        #    So for k in 0..min(i, N-1), r[k] includes x[i-k]*x[i].
        #    Change: -2 * x[i] * x[i-k]
        
        # Note: if k=0, both terms apply? 
        # r[0] = sum x[j]^2 = N. 
        # Flip x[i]: x[i]^2 stays 1. So r[0] does NOT change.
        # My formula: 
        # Case 1: k=0, i+0 < N. Change -2 x[i] x[i].
        # Case 2: k=0, i-0 >= 0. Change -2 x[i] x[i].
        # Total change -4 x[i]^2 = -4. But r[0] should be constant N.
        # Wait, r[k] = sum_{j=0}^{N-k-1} x[j] x[j+k].
        # If we flip x[i], it affects r[k] if i is in [0, N-k-1] OR i-k is in [0, N-k-1] (which implies i in [k, N-1]).
        # Actually, let's just re-calculate the affected k's.
        
        # Affected k's:
        # From j=i: k such that i < N-k => k < N-i. k >= 0.
        # From j=i-k: k such that i-k >= 0 => k <= i. And i-k < N-k => i < N (true).
        # So k ranges from 0 to max(N-i-1, i).
        
        k_max = max(N - i - 1, i)
        if k_max < 0:
            return r
        
        old_val = x[i]
        
        for k in range(k_max + 1):
            delta = 0
            # Contribution from j=i (term x[i] x[i+k])
            if i + k < N:
                delta -= 2 * old_val * x[i+k]
            # Contribution from j=i-k (term x[i-k] x[i])
            if i - k >= 0:
                delta -= 2 * old_val * x[i-k]
            
            r[k] += delta
        
        # r[0] check:
        # k=0. 
        # If i < N (yes), delta -= 2*old*x[i] = -2*old^2 = -2.
        # If i >= 0 (yes), delta -= 2*old*x[i] = -2.
        # Total -4.
        # But r[0] = sum x^2. Flipping one element doesn't change sum of squares.
        # Why the discrepancy?
        # Ah, the formula r[k] = sum x[j]x[j+k].
        # For k=0, r[0] = sum x[j]^2.
        # The term x[i]^2 is present once.
        # My derivation counted it twice?
        # Let's check the indices.
        # Term 1: j=i. Index in sum is j. Value x[i]x[i+k].
        # Term 2: j=i-k. Index in sum is j. Value x[i-k]x[i].
        # For k=0, j=i and j=i are the same term!
        # So for k=0, we should only apply the change once?
        # No, the sum is over j. The term at j=i is x[i]x[i].
        # When we flip x[i], this term changes from 1 to 1. No change!
        # Wait. x[i] becomes -x[i]. (-x[i])^2 = x[i]^2.
        # So r[0] is invariant.
        
        # Let's re-evaluate the delta logic.
        # r_new[k] = sum_{j} x'_j x'_{j+k}
        # x'_i = -x_i, x'_m = x_m for m != i.
        
        # If k=0:
        # r_new[0] = sum x'_j^2 = sum x_j^2 = r[0].
        
        # If k>0:
        # Terms involving x_i:
        # 1. j=i: term x_i x_{i+k}. New: -x_i x_{i+k}. Diff: -2 x_i x_{i+k}.
        # 2. j=i-k: term x_{i-k} x_i. New: x_{i-k} (-x_i). Diff: -2 x_{i-k} x_i.
        
        # So for k>0, the delta is sum of these two if they exist.
        # For k=0, the "two terms" are actually the same term x_i^2, which doesn't change.
        # So we should skip k=0 or handle it specially.
        
        # Correction in loop:
        # The previous loop added delta for k=0 as -4. This is wrong.
        # We must set r[0] back to N or just not update it.
        
        r[0] = N # It's always N for +/- 1 sequences.
        
        return r

    # Refine the update function to be correct
    def update_r_after_flip_correct(x, i, r):
        old_val = x[i]
        # k=0 is invariant
        # k > 0
        # Max k affected:
        # From j=i: i+k < N => k < N-i.
        # From j=i-k: i-k >= 0 => k <= i.
        k_max = max(N - i - 1, i)
        if k_max < 1:
            return r
            
        for k in range(1, k_max + 1):
            delta = 0
            if i + k < N:
                delta -= 2 * old_val * x[i+k]
            if i - k >= 0:
                delta -= 2 * old_val * x[i-k]
            r[k] += delta
        r[0] = N
        return r

    # Reset r with correct function (init_r was correct)
    r = init_r(x)
    
    iterations = 0
    while time.time() < deadline:
        # Pick a random index to flip
        i = np.random.randint(N)
        
        # Compute current score
        # We need max |r[k]| for k>0.
        # Current score is best_score? No, current score of x.
        # We maintain current_score? 
        # Since r changes, we need to recompute max.
        # This is O(N).
        current_score = np.max(np.abs(r[1:]))
        
        # Try flipping i
        x[i] = -x[i]
        r = update_r_after_flip_correct(x, i, r)
        new_score = np.max(np.abs(r[1:]))
        
        # Acceptance criteria
        if new_score < current_score:
            best_x = x.copy()
            best_score = new_score
            # Always accept improvement
        else:
            # Simulated Annealing
            delta = new_score - current_score
            if delta < 0:
                # Improvement
                best_x = x.copy()
                best_score = new_score
            else:
                # Worse
                if np.random.rand() < np.exp(-delta / T):
                    # Accept
                    pass
                else:
                    # Reject: revert
                    x[i] = -x[i]
                    r = update_r_after_flip_correct(x, i, r)
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart with higher T? Or just continue greedy?
            # Let's do a few greedy passes at the end.
            
        iterations += 1
        if iterations % 1000 == 0:
             # Optional: check time
             pass

    # Final Greedy Polish
    # Try to improve by flipping any single bit
    improved = True
    while improved:
        improved = False
        current_score = np.max(np.abs(r[1:]))
        for i in range(N):
            x[i] = -x[i]
            r = update_r_after_flip_correct(x, i, r)
            new_score = np.max(np.abs(r[1:]))
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                x[i] = -x[i]
                r = update_r_after_flip_correct(x, i, r)
        
        if current_score < best_score:
            best_x = x.copy()
            best_score = current_score
            r = init_r(best_x) # Sync r with best_x if we updated x? 
            # Wait, if we updated x in the loop, r is synced with x.
            # If we rejected, x and r are synced.
            # So if we found an improvement, x is the new best.
            # But we might have rejected some flips, so x is not necessarily the best?
            # No, we only keep the flip if it improves. So x is always the best found in this pass.
            
        # Check time
        if time.time() > deadline - 5:
            break

    return best_x

# EVOLVE-BLOCK-END
