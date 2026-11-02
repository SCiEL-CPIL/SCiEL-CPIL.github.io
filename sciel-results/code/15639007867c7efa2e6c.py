import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop searching at 135s.
    deadline = start_time + 135.0
    
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

    def _local_search_sa(s, A, val, t_start, t_end):
        """
        Simulated Annealing local search.
        Accepts worse moves with probability exp(-delta/T) to escape local minima.
        """
        s = s.copy()
        A = A.copy()
        val = val
        
        # Temperature schedule
        # Start high enough to allow exploration, cool down to exploit.
        # val for random seq is ~20-40. Delta can be small.
        T = 5.0 
        T_min = 0.01
        # We want T to reach T_min by t_end.
        # T(t) = T_min * (T_start / T_min)^((t_end - t) / (t_end - t_start))
        # Or simpler: linear or geometric cooling per step.
        # Let's use geometric cooling based on remaining time fraction.
        # Actually, fixed cooling rate per step is easier to reason about if step time is constant.
        # But FFT time is constant ~ O(N log N).
        # Let's just use a fixed number of steps or time-based.
        
        # We will run until time limit.
        # Cooling: T *= 0.999 per step?
        # If we do 1000 steps, 0.999^1000 ~ 0.36. T goes from 5 to 1.8.
        # If we do 5000 steps, 0.999^5000 ~ 0.006. T goes from 5 to 0.03.
        # This seems reasonable.
        
        cooling = 0.999
        
        while time.time() < t_end and T > T_min:
            i = np.random.randint(N)
            old_si = s[i]
            s[i] = -old_si
            
            # Recompute A and val
            A_new, val_new = _eval(s)
            
            delta = val_new - val
            
            if delta < 0:
                # Improvement
                A = A_new
                val = val_new
            else:
                # Accept worse move with probability
                # Avoid underflow in exp
                if delta / T < 700:
                    p = np.exp(-delta / T)
                else:
                    p = 0.0
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
        # Check if we have enough time for at least one SA run
        # A single SA run should take at least 1-2 seconds to be useful.
        if time.time() > deadline - 2.0:
            break
            
        # Random initialization
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A, val = _eval(s)
        
        # Run Simulated Annealing
        # Allocate time for this restart:
        # If we have 100s left, maybe spend 5s on this SA?
        # Let's spend a fixed amount, say 2-5 seconds per restart,
        # or dynamically based on remaining time.
        # Dynamic: spend 10% of remaining time?
        # Or fixed 2 seconds?
        # 135s / 2s = ~67 restarts. That's a lot.
        # 135s / 5s = ~27 restarts.
        # Let's try 3 seconds per restart.
        
        sa_start = time.time()
        sa_end = sa_start + 3.0
        if sa_end > deadline:
            sa_end = deadline
            
        s, A, val = _local_search_sa(s, A, val, sa_start, sa_end)
        
        if val < best_val:
            best_val = val
            best_seq = s.copy()
        
        num_restarts += 1

    return best_seq
