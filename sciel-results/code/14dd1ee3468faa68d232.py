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

def _smooth_max_abs_acf(A, beta=50.0):
    """Smooth surrogate for max|A[1:]} using log-sum-exp."""
    x = np.abs(A[1:])
    m = np.max(x)
    if m == 0:
        return 0.0
    shifted = x - m
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta + m
    return float(log_sum)

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_val = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    # Pre-compute FFT size
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    def _eval(s):
        s_f = s.astype(np.float64)
        F = np.fft.rfft(s_f, n=n_fft)
        P = F * np.conj(F)
        acf = np.fft.irfft(P, n=n_fft)
        A = np.real(acf[:N])
        val = float(np.max(np.abs(A[1:])))
        return A, val

    def _eval_smooth(s):
        """Evaluate smooth surrogate objective."""
        A = _calc_acf(s)
        val_smooth = _smooth_max_abs_acf(A)
        val_exact = float(np.max(np.abs(A[1:])))
        return A, val_smooth, val_exact

    def _greedy_polish(s, A, val, time_limit=2.0):
        """Greedy local search: only accept improvements."""
        s = s.copy()
        A = A.copy()
        val = val
        local_deadline = time.time() + time_limit
        
        # Try single-bit flips in random order, multiple passes until no improvement
        # We use a "stall" counter to avoid infinite loops if we hit a local minimum
        while time.time() < local_deadline:
            improved = False
            # Random permutation for unbiased search
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > local_deadline:
                    return s, A, val
                
                # Flip bit
                old_val = s[i]
                s[i] = -old_val
                
                # Evaluate
                A_new, val_new = _eval(s)
                
                if val_new < val:
                    A = A_new
                    val = val_new
                    improved = True
                else:
                    s[i] = old_val
            
            if not improved:
                break
        
        return s, A, val

    def _metropolis_hastings(s, A, val, time_limit=10.0):
        """Metropolis-Hastings with smooth surrogate for diverse exploration."""
        s = s.copy()
        A = A.copy()
        val = val
        
        # Initial smooth value
        _, val_smooth, _ = _eval_smooth(s)
        
        T = 10.0
        T_min = 0.01
        local_deadline = time.time() + time_limit
        
        # Block sizes: mostly single bits, some larger for structural changes
        block_sizes = [1, 1, 1, 1, 2, 2, 4, 8]
        
        while time.time() < local_deadline and T > T_min:
            block_size = block_sizes[np.random.randint(0, len(block_sizes))]
            
            if block_size == 1:
                start_idx = np.random.randint(0, N)
            else:
                start_idx = np.random.randint(0, N - block_size + 1)
            
            # Flip block
            old_block = s[start_idx:start_idx + block_size].copy()
            s[start_idx:start_idx + block_size] = -old_block
            
            A_new, val_smooth_new, val_exact_new = _eval_smooth(s)
            
            delta = val_smooth_new - val_smooth
            
            if delta < 0:
                A = A_new
                val = val_exact_new
                val_smooth = val_smooth_new
            else:
                p = np.exp(-delta / T)
                if np.random.random() < p:
                    A = A_new
                    val = val_exact_new
                    val_smooth = val_smooth_new
                else:
                    s[start_idx:start_idx + block_size] = old_block
            
            # Cooling
            T *= 0.9995
            
        return s, A, val

    # Main loop: multiple restarts
    # We alternate between random starts and perturbing the best known solution
    num_restarts = 0
    
    while time.time() < deadline - 15.0:
        # Decide initialization strategy
        # 50% random, 50% perturbed best
        if num_restarts > 0 and np.random.random() < 0.5 and best_val < float('inf'):
            s = best_seq.copy()
            # Perturb ~5% of bits
            num_flips = int(0.05 * N)
            flip_indices = np.random.choice(N, num_flips, replace=False)
            s[flip_indices] = -s[flip_indices]
        else:
            s = np.random.randint(0, 2, size=N) * 2 - 1
        
        A, val = _eval(s)
        
        # Phase 1: MH for exploration
        # Time allocation: try to use ~8s for MH if available, otherwise scale down
        remaining = deadline - time.time()
        mh_time = min(remaining * 0.6, 10.0)
        if mh_time > 1.0:
            s, A, val = _metropolis_hastings(s, A, val, time_limit=mh_time)
        
        # Phase 2: Greedy polish for exploitation
        # Use remaining time minus safety buffer
        polish_time = min(deadline - time.time() - 5.0, 5.0)
        if polish_time > 0.5:
            s, A, val = _greedy_polish(s, A, val, time_limit=polish_time)
        
        if val < best_val:
            best_val = val
            best_seq = s.copy()
        
        num_restarts += 1
        
        # Safety check
        if time.time() > deadline - 10.0:
            break

    # Final polish on the best sequence if time permits
    if time.time() < deadline - 2.0:
        best_A = _calc_acf(best_seq)
        # Do one more tight polish on the global best
        polish_time = min(deadline - time.time() - 1.0, 3.0)
        if polish_time > 0.5:
            best_seq, _, best_val = _greedy_polish(best_seq, best_A, best_val, time_limit=polish_time)

    return best_seq
