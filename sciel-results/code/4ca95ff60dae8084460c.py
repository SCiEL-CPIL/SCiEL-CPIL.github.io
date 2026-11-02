import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1] using FFT."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    """Return the max absolute value of A[1:]."""
    return float(np.max(np.abs(A[1:])))

def construct():
    start_time = time.time()
    # Use a conservative deadline to ensure we return well before 150s
    deadline = 130.0
    
    best_val = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    # Pre-compute FFT size
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    # Helper to compute A and val quickly
    def _eval(s):
        s_f = s.astype(np.float64)
        F = np.fft.rfft(s_f, n=n_fft)
        P = F * np.conj(F)
        acf = np.fft.irfft(P, n=n_fft)
        A = np.real(acf[:N])
        val = float(np.max(np.abs(A[1:])))
        return A, val

    # Multi-start local search with simulated annealing flavor
    # Objective: minimize max(|A[1]|, ..., |A[N-1]|)
    
    # We will perform a series of restarts.
    # In each restart, we start from a random or best-known state and do local search.
    
    # To improve upon the seed (score 1.0), we need to find a sequence with val < 1.0?
    # Wait, the score is seed/value. Seed program returns np.ones(N).
    # For np.ones(N), A[0]=N, A[k]=N-k for k>0.
    # max(|A[1:]|) = N-1 = 399.
    # So seed value is 399.
    # Score = 399 / value.
    # Current program score is 1.0, which implies value = 399.
    # The current program in the prompt is supposed to be the seed?
    # "CURRENT PROGRAM (score 1.0000)"
    # The code in CURRENT PROGRAM does random search.
    # If it scored 1.0, it likely returned the seed result or failed to improve.
    # The seed result is all ones, val=399.
    # We want to minimize val.
    # Random sequences of +/- 1 have autocorrelations that are roughly sqrt(N).
    # sqrt(400) = 20.
    # So a random sequence should have val ~ 20-40.
    # Score would be 399 / 20 ~ 20.
    # Why did the current program score 1.0?
    # Maybe it timed out and returned best_seq which was initialized to np.ones(N)?
    # Yes, `best_seq = np.ones(N)` and if no improvement found, it returns ones.
    # The local search in the current program was flawed or too slow to find improvements?
    # Or maybe it did find improvements but the score calculation is different?
    # "score is the upper bound on C3"
    # "evaluator measures the raw value (lower is better). The search score is seed/value"
    # If seed is all ones, val=399.
    # If we find a random sequence with val=20, score = 399/20 = 19.95.
    # This is a huge improvement.
    # The current program's local search logic was commented out and replaced with a simple random flip loop.
    # `while time.time() - restart_start < 1.0 ...`
    # It only runs for 1 second per restart.
    # And it only accepts if `val_new < current_val`.
    # Starting from random, current_val is already low (~20).
    # Flipping random bits rarely improves the max absolute autocorrelation immediately.
    # It gets stuck in local minima.
    
    # Strategy:
    # 1. Generate many random sequences, pick the best few.
    # 2. Use a more effective local search, e.g., Simulated Annealing or just many random flips accepting worse moves temporarily.
    # 3. Or use a dedicated optimization like Gradient Descent on a smooth surrogate?
    #    Let's try Simulated Annealing on the discrete domain.
    
    # Better local search:
    # Instead of only accepting improvements, accept worse moves with probability exp(-delta/T).
    # This helps escape local minima.
    
    def _local_search_sa(s, A, val, t_max=0.5):
        """Simulated Annealing local search."""
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        s = s.copy()
        A = A.copy()
        val = val
        
        # Precompute indices
        # We flip one index at a time.
        
        while time.time() - start_time < deadline and T > T_min:
            i = np.random.randint(N)
            old_si = s[i]
            s[i] = -old_si
            
            # Recompute A and val
            # This is O(N log N) due to FFT.
            # N=400, FFT is fast.
            A_new, val_new = _eval(s)
            
            delta = val_new - val
            
            if delta < 0:
                # Improvement
                A = A_new
                val = val_new
            else:
                # Accept worse move with probability
                p = np.exp(-delta / T)
                if np.random.random() < p:
                    A = A_new
                    val = val_new
                else:
                    # Reject
                    s[i] = old_si
                    # A and val remain unchanged
            
            T *= cooling
            
        return s, A, val

    # Main loop
    num_restarts = 0
    while time.time() < deadline:
        # Random initialization
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A, val = _eval(s)
        
        # Run Simulated Annealing
        s, A, val = _local_search_sa(s, A, val)
        
        if val < best_val:
            best_val = val
            best_seq = s.copy()
        
        num_restarts += 1
        
        # Optional: If we are stuck, try a more aggressive cooling or different init?
        # For now, just restart with new random init.
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    return best_seq
