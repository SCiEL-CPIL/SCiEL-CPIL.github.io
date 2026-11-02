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

def _soft_max(x, beta=10.0):
    """Soft max using log-sum-exp for smoothness."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return 0.0
    # Avoid overflow in exp
    shifted = x - x_max
    # Clip to prevent overflow in exp
    shifted = np.clip(shifted, -100, 100)
    exp_sum = np.sum(np.exp(beta * shifted))
    if exp_sum <= 0:
        return x_max
    return x_max + (1.0 / beta) * np.log(exp_sum)

def _soft_min(x, beta=10.0):
    """Soft min using log-sum-exp for smoothness."""
    return -_soft_max(-x, beta)

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

    def _eval_soft(s, beta=5.0):
        s_f = s.astype(np.float64)
        F = np.fft.rfft(s_f, n=n_fft)
        P = F * np.conj(F)
        acf = np.fft.irfft(P, n=n_fft)
        A = np.real(acf[:N])
        abs_A = np.abs(A[1:])
        # Use soft max for a smoother objective
        val = _soft_max(abs_A, beta)
        return A, val

    def _continuous_init():
        """Initialize using continuous optimization to find a good binary seed."""
        # Start with random continuous values in [-1, 1]
        s_cont = np.random.uniform(-1, 1, size=N)
        
        # Gradient descent on L2 norm of autocorrelation
        # A = autocorrelation(s)
        # Minimize sum(A[1:]^2)
        
        for step in range(200):
            if time.time() > deadline:
                break
            
            s_f = s_cont.astype(np.float64)
            F = np.fft.rfft(s_f, n=n_fft)
            P = F * np.conj(F)
            acf = np.fft.irfft(P, n=n_fft)
            A = np.real(acf[:N])
            
            # Gradient of sum(A[1:]^2) with respect to s
            # A_k = sum_{i} s_i * s_{i+k}
            # d(A_k^2)/ds_j = 2 * A_k * (s_{j+k} + s_{j-k})
            
            grad = np.zeros(N)
            for k in range(1, N):
                if time.time() > deadline:
                    break
                # For each j, contribution from lag k
                # s_{j+k} is defined for j in [0, N-1-k]
                # s_{j-k} is defined for j in [k, N-1]
                
                # Vectorized update
                # Forward part: j in [0, N-1-k], s_{j+k} = s_cont[k:N]
                if N - 1 - k >= 0:
                    grad[0:N-k] += 2 * A[k] * s_cont[k:N]
                
                # Backward part: j in [k, N-1], s_{j-k} = s_cont[0:N-k]
                if k < N:
                    grad[k:N] += 2 * A[k] * s_cont[0:N-k]
            
            # Gradient step
            lr = 0.01
            s_cont -= lr * grad
            
            # Project to [-1, 1]
            s_cont = np.clip(s_cont, -1, 1)
        
        # Round to binary
        s_bin = np.sign(s_cont).astype(np.int8)
        s_bin[s_bin == 0] = 1
        
        return s_bin

    def _local_search_sa(s, A, val, t_max=5.0, use_soft=False, beta=5.0):
        """Simulated Annealing with block-flip moves."""
        T = 50.0
        T_min = 0.1
        s = s.copy()
        A = A.copy()
        val = val
        
        # Block sizes to try
        block_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        eval_func = _eval_soft if use_soft else _eval
        
        while time.time() - start_time < deadline and T > T_min:
            # Choose block size probabilistically
            if np.random.random() < 0.2:
                block_size = 1
            else:
                block_size = block_sizes[np.random.randint(1, len(block_sizes))]
            
            # Choose random start position for block
            start_idx = np.random.randint(0, N - block_size + 1)
            
            # Flip the block
            old_block = s[start_idx:start_idx + block_size].copy()
            s[start_idx:start_idx + block_size] = -old_block
            
            A_new, val_new = eval_func(s)
            
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
                    s[start_idx:start_idx + block_size] = old_block
            
            # Cool down
            T *= 0.995
            
        return s, A, val

    def _greedy_polish(s, A, val, t_max=2.0):
        """Greedy local search: only accept improvements."""
        s = s.copy()
        A = A.copy()
        val = val
        
        # Try single-bit flips first, then small blocks
        move_types = [1, 2, 4, 8]
        
        while time.time() - start_time < deadline:
            improved = False
            for move_size in move_types:
                if move_size == 1:
                    # Try all single-bit flips
                    for i in range(N):
                        if time.time() - start_time > deadline:
                            return s, A, val
                        s[i] = -s[i]
                        A_new, val_new = _eval(s)
                        if val_new < val:
                            A = A_new
                            val = val_new
                            improved = True
                        else:
                            s[i] = -s[i]  # revert
                else:
                    # Try random block flips of this size
                    for _ in range(50):
                        if time.time() - start_time > deadline:
                            return s, A, val
                        start_idx = np.random.randint(0, N - move_size + 1)
                        old_block = s[start_idx:start_idx + move_size].copy()
                        s[start_idx:start_idx + move_size] = -old_block
                        A_new, val_new = _eval(s)
                        if val_new < val:
                            A = A_new
                            val = val_new
                            improved = True
                        else:
                            s[start_idx:start_idx + move_size] = old_block
            
            if not improved:
                break
        
        return s, A, val

    # Main loop: multiple restarts with continuous init + SA + greedy polish
    num_restarts = 0
    while time.time() < deadline - 10.0:
        # Use continuous optimization for initialization
        s = _continuous_init()
        A, val = _eval(s)
        
        # Run SA with soft objective first, then hard objective
        remaining = deadline - time.time()
        sa_time_soft = min(remaining * 0.3, 10.0)
        s, A, val_soft = _local_search_sa(s, A, val_soft, t_max=sa_time_soft, use_soft=True, beta=5.0)
        
        # Re-evaluate with hard objective
        A, val = _eval(s)
        
        # Run SA with hard objective
        remaining = deadline - time.time()
        sa_time_hard = min(remaining * 0.4, 15.0)
        s, A, val = _local_search_sa(s, A, val, t_max=sa_time_hard, use_soft=False)
        
        # Greedy polish
        s, A, val = _greedy_polish(s, A, val)
        
        if val < best_val:
            best_val = val
            best_seq = s.copy()
        
        num_restarts += 1
        
        # If we're getting close to deadline, do one more quick polish
        if time.time() > deadline - 5.0:
            break
    
    # Final polish on the best sequence if time permits
    if time.time() < deadline - 2.0:
        best_seq, _, best_val = _greedy_polish(best_seq, _calc_acf(best_seq), best_val, t_max=2.0)
    
    return best_seq
