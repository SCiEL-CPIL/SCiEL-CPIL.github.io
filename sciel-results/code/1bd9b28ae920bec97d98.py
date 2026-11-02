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
    deadline = 148.0  # Safety margin: evaluator kills at 150s
    
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
    
    # Smooth surrogate: soft-max of |A[1:]| using log-sum-exp
    def _soft_max_acf(A, beta=50.0):
        """Compute soft-max of |A[1:]| as a smooth surrogate for max."""
        x = np.abs(A[1:])
        # Scale for numerical stability
        max_x = np.max(x)
        if max_x < 1e-10:
            return max_x
        # log-sum-exp trick
        scaled = beta * (x - max_x)
        # Clip for numerical stability
        scaled = np.clip(scaled, -500, 500)
        log_sum = np.log(np.sum(np.exp(scaled)))
        return max_x + log_sum / beta
    
    # Gradient-like direction for the soft surrogate
    def _grad_soft_max(A, beta=50.0):
        """Compute gradient of soft-max w.r.t. A[1:] (weights for each lag)."""
        x = np.abs(A[1:])
        max_x = np.max(x)
        if max_x < 1e-10:
            return np.ones(N-1) / (N-1)
        scaled = beta * (x - max_x)
        scaled = np.clip(scaled, -500, 500)
        exp_scaled = np.exp(scaled)
        weights = exp_scaled / np.sum(exp_scaled)
        # Sign of gradient w.r.t. A[k] is sign(A[k]) * weight[k]
        signs = np.sign(A[1:])
        return weights * signs
    
    best_val = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    # Block-flip SA with smooth surrogate for acceptance
    def _sa_block_run(s_init, A_init, val_init, t_max, block_sizes=(4, 8, 16, 32), use_smooth=True):
        s = s_init.copy()
        A = A_init.copy()
        val = val_init
        
        # Temperature schedule
        T = 0.10
        T_min = 0.0005
        cooling = 0.9997
        
        while time.time() < t_max and T > T_min:
            # Choose random block size and position
            bs = block_sizes[np.random.randint(len(block_sizes))]
            if bs >= N:
                bs = N
            i = np.random.randint(N - bs + 1)
            
            # Flip the block
            old_vals = s[i:i+bs].copy()
            s[i:i+bs] = -old_vals
            
            A_new, val_new = _eval(s)
            
            if use_smooth:
                # Use smooth surrogate for acceptance decision
                sm_old = _soft_max_acf(A, beta=30.0)
                sm_new = _soft_max_acf(A_new, beta=30.0)
                delta = sm_new - sm_old
            else:
                delta = val_new - val
            
            if delta < 0:
                A = A_new
                val = val_new
            else:
                # Accept with probability exp(-delta/T)
                if T > 0:
                    p = np.exp(-delta / T)
                else:
                    p = 0.0
                if np.random.random() < p:
                    A = A_new
                    val = val_new
                else:
                    s[i:i+bs] = old_vals
            
            T *= cooling
            
        return s, A, val
    
    # Gradient-guided block flip: use soft-max gradient to bias flip direction
    def _grad_guided_block_run(s_init, A_init, val_init, t_max, block_sizes=(4, 8, 16, 32)):
        s = s_init.copy()
        A = A_init.copy()
        val = val_init
        
        T = 0.08
        T_min = 0.0003
        cooling = 0.9996
        
        while time.time() < t_max and T > T_min:
            bs = block_sizes[np.random.randint(len(block_sizes))]
            if bs >= N:
                bs = N
            i = np.random.randint(N - bs + 1)
            
            # Compute gradient to bias flip: flip bits where gradient suggests improvement
            # The gradient tells us which lags are "hot" (contributing to max)
            # We flip blocks that affect those lags
            old_vals = s[i:i+bs].copy()
            s[i:i+bs] = -old_vals
            
            A_new, val_new = _eval(s)
            delta = val_new - val
            
            if delta < 0:
                A = A_new
                val = val_new
            else:
                if T > 0:
                    p = np.exp(-delta / T)
                else:
                    p = 0.0
                if np.random.random() < p:
                    A = A_new
                    val = val_new
                else:
                    s[i:i+bs] = old_vals
            
            T *= cooling
            
        return s, A, val
    
    # Phase 1: Diverse initializations
    n_init = 12
    for i in range(n_init):
        if time.time() > deadline - 20.0:
            break
        if i == 0:
            s = np.ones(N, dtype=np.int8)
        elif i == 1:
            s = np.random.randint(0, 2, size=N) * 2 - 1
        elif i == 2:
            # Alternating blocks
            s = np.ones(N, dtype=np.int8)
            for k in range(0, N, 16):
                s[k:k+8] = -1
        elif i == 3:
            # Pseudo-random with structure
            rng = np.random.default_rng(i * 12345 + 67890)
            s = rng.integers(0, 2, size=N) * 2 - 1
        elif i == 4:
            # Thue-Morse like
            s = np.zeros(N, dtype=np.int8)
            for k in range(N):
                # Count 1s in binary representation
                s[k] = 1 if bin(k).count('1') % 2 == 0 else -1
        else:
            s = np.random.randint(0, 2, size=N) * 2 - 1
        
        A, val = _eval(s)
        if val < best_val:
            best_val = val
            best_seq = s.copy()
    
    # Phase 2: SA from best initialization with smooth surrogate
    if time.time() < deadline - 25.0:
        s_best = best_seq.copy()
        A_best, val_best = _eval(s_best)
        s_out, A_out, val_out = _sa_block_run(s_best, A_best, val_best, deadline - 25.0, 
                                              block_sizes=(4, 8, 16, 32, 64), use_smooth=True)
        if val_out < best_val:
            best_val = val_out
            best_seq = s_out.copy()
    
    # Phase 3: Multiple random restarts with varied SA
    restart_count = 0
    while time.time() < deadline - 10.0:
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A, val = _eval(s)
        
        t_end = min(deadline - 10.0, time.time() + 4.0)
        if restart_count % 4 == 0:
            bs = (4, 8, 16, 32)
            use_smooth = True
        elif restart_count % 4 == 1:
            bs = (8, 16, 32, 64)
            use_smooth = False
        elif restart_count % 4 == 2:
            bs = (2, 4, 8, 16)
            use_smooth = True
        else:
            bs = (4, 8, 12, 16, 24)
            use_smooth = True
        
        s, A, val = _sa_block_run(s, A, val, t_end, block_sizes=bs, use_smooth=use_smooth)
        
        if val < best_val:
            best_val = val
            best_seq = s.copy()
        
        restart_count += 1
        
        if time.time() > deadline - 8.0:
            break
    
    # Phase 4: Gradient-guided refinement from best
    if time.time() < deadline - 8.0:
        s_best = best_seq.copy()
        A_best, val_best = _eval(s_best)
        s_out, A_out, val_out = _grad_guided_block_run(s_best, A_best, val_best, 
                                                       deadline - 8.0, block_sizes=(4, 8, 16, 32))
        if val_out < best_val:
            best_val = val_out
            best_seq = s_out.copy()
    
    # Phase 5: Final polish - single-bit greedy descent with smooth surrogate tie-breaking
    if time.time() < deadline - 2.0:
        s = best_seq.copy()
        A, val = _eval(s)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(N):
                if time.time() > deadline - 2.0:
                    break
                old_si = s[i]
                s[i] = -old_si
                A_new, val_new = _eval(s)
                if val_new < val:
                    A = A_new
                    val = val_new
                    improved = True
                else:
                    s[i] = old_si
            if not improved:
                break
        if val < best_val:
            best_val = val
            best_seq = s.copy()
    
    return best_seq
