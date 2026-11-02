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
    deadline = 130.0
    
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

    def _eval_multi(s_list):
        """Evaluate multiple sequences at once using batched FFT."""
        # s_list is a list of int8 arrays
        # Stack them
        s_stack = np.stack(s_list, axis=0).astype(np.float64)
        # rfft along last axis
        F = np.fft.rfft(s_stack, n=n_fft, axis=1)
        P = F * np.conj(F)
        acf = np.fft.irfft(P, n=n_fft, axis=1)
        A_stack = np.real(acf[:, :N])
        vals = np.max(np.abs(A_stack[:, 1:]), axis=1)
        return A_stack, vals

    best_val = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    # Strategy:
    # 1. Fast random initialization to find a decent starting point.
    # 2. Simulated Annealing with a specific focus on escaping local minima.
    #    The key is to use a "soft" energy function or accept moves that don't immediately
    #    improve the max, but might lead to a better max later.
    #    However, the directive suggests investigating smooth surrogates or epigraph formulations.
    #    Let's try optimizing a smooth surrogate: sum(exp(beta * |A[k]|)) or similar.
    #    But for now, let's stick to the max objective but with a more effective SA.
    
    # Let's try a different approach:
    # Coordinate descent with random restarts, but using a "temperature" to accept
    # moves that increase the objective slightly if they reduce the variance of the autocorrelation?
    # No, let's stick to the max.
    
    # The previous SA was:
    # T = 10.0, cooling = 0.999.
    # Maybe T was too high or cooling too slow?
    # Let's try multiple SA runs with different parameters.
    
    def _sa_run(s_init, A_init, val_init, t_max):
        s = s_init.copy()
        A = A_init.copy()
        val = val_init
        
        # Dynamic temperature
        # Start high, cool down
        T = 5.0
        T_min = 0.1
        cooling = 0.9995
        
        while time.time() < t_max and T > T_min:
            i = np.random.randint(N)
            old_si = s[i]
            s[i] = -old_si
            
            A_new, val_new = _eval(s)
            
            delta = val_new - val
            
            if delta < 0:
                A = A_new
                val = val_new
            else:
                p = np.exp(-delta / T)
                if np.random.random() < p:
                    A = A_new
                    val = val_new
                else:
                    s[i] = old_si
            
            T *= cooling
            
        return s, A, val

    # Phase 1: Random initialization
    # Try a few random starts
    for i in range(5):
        if time.time() > deadline - 10.0:
            break
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A, val = _eval(s)
        if val < best_val:
            best_val = val
            best_seq = s.copy()
            
    # Phase 2: SA from best found
    if time.time() < deadline - 5.0:
        s_best = best_seq.copy()
        A_best, val_best = _eval(s_best)
        s_out, A_out, val_out = _sa_run(s_best, A_best, val_best, deadline - 5.0)
        if val_out < best_val:
            best_val = val_out
            best_seq = s_out.copy()

    # Phase 3: More random restarts with SA
    # Use remaining time for more restarts
    while time.time() < deadline - 5.0:
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A, val = _eval(s)
        
        # Short SA run
        t_end = min(deadline - 5.0, time.time() + 2.0)
        s, A, val = _sa_run(s, A, val, t_end)
        
        if val < best_val:
            best_val = val
            best_seq = s.copy()
            
    return best_seq
