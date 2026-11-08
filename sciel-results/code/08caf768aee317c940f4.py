import numpy as np
import time

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_c3(signed_step, M):
    """
    Compute the C3 upper bound: max |autoconv(t)| for the signed step function.
    """
    dx = 0.5 / M
    
    # Build fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    f_fine = signed_step.astype(np.float64)[i_indices]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def _structured_initialization(M):
    """Generate multiple structured candidates and return the best."""
    candidates = []
    
    # All ones (baseline)
    candidates.append(np.ones(N, dtype=np.int8))
    
    # Alternating
    candidates.append(np.tile(np.array([1, -1], dtype=np.int8), N // 2))
    
    # Balanced Random
    np.random.seed(42)
    for _ in range(10):
        # Ensure roughly equal number of +1 and -1
        seq = np.random.choice([-1, 1], size=N)
        # If not balanced, flip some
        if np.sum(seq) != 0:
             # Simple balancing: flip random elements to reduce sum
             s = np.sum(seq)
             if s > 0:
                 idx = np.where(seq == 1)[0]
                 to_flip = np.random.choice(idx, size=s//2, replace=False)
                 seq[to_flip] = -1
             else:
                 idx = np.where(seq == -1)[0]
                 to_flip = np.random.choice(idx, size=-s//2, replace=False)
                 seq[to_flip] = 1
        candidates.append(seq.astype(np.int8))
    
    # M-sequence based candidates (pseudo-random binary sequences)
    for seed in [1, 7, 13, 29, 37]:
        rng = np.random.RandomState(seed)
        m_seq = np.zeros(N, dtype=np.int8)
        state = 1
        for i in range(N):
            m_seq[i] = 1 if state > 0 else -1
            # Simple LFSR tap
            new_bit = ((state >> 0) ^ (state >> 3) ^ (state >> 5) ^ (state >> 7)) & 1
            state = (state >> 1) | (new_bit << 7)
        candidates.append(m_seq)
    
    best = None
    best_c3 = np.inf
    for c in candidates:
        c3 = _compute_c3(c, M)
        if c3 < best_c3:
            best_c3 = c3
            best = c.copy()
    
    return best, best_c3

def construct():
    start_time = time.time()
    
    # Increase resolution for better peak detection
    # M=4*N ensures we sample midpoints and boundaries more densely
    M = 4 * N 
    dx = 0.5 / M
    
    # Get a good structured start
    best_step, best_c3 = _structured_initialization(M)
    
    # Precompute fine grid mapping
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    
    # Build fine grid signal
    current_step = best_step.copy()
    f_fine = current_step.astype(np.float64)[i_indices]
    
    # Compute initial autoconvolution
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    current_c3 = np.max(np.abs(autoconv))
    
    # Precompute which fine-grid positions correspond to each coarse index
    coarse_to_fine = [[] for _ in range(N)]
    for m in range(M):
        coarse_to_fine[i_indices[m]].append(m)
    
    coarse_to_fine_arrays = [np.array(coarse_to_fine[i], dtype=np.int64) for i in range(N)]
    
    # Precompute FFTs for delta patterns
    fft_delta_cache = np.zeros((N, 2 * M), dtype=np.complex128)
    for i in range(N):
        delta_f = np.zeros(M, dtype=np.float64)
        delta_f[coarse_to_fine_arrays[i]] = -2.0
        fft_delta_cache[i] = np.fft.fft(delta_f, n=2 * M)
        
    # Precompute delta_conv
    delta_conv_cache = np.zeros((N, 2 * M), dtype=np.float64)
    for i in range(N):
        fd = fft_delta_cache[i]
        delta_conv_cache[i] = np.fft.ifft(fd * np.conj(fd)).real * dx

    # Simulated annealing
    T_start = 0.02
    T_end = 0.001
    total_budget = DEADLINE - 5.0
    restart_count = 8
    time_per_restart = total_budget / restart_count
    
    # Pre-allocate buffer for new_autoconv to reduce GC pressure
    new_autoconv_buf = np.zeros(2 * M, dtype=np.float64)
    
    for restart in range(restart_count):
        if restart > 0:
            # Perturb current best with random flips
            current_step = best_step.copy()
            n_flips = N // 8
            flip_indices = np.random.choice(N, size=n_flips, replace=False)
            current_step[flip_indices] *= -1
            f_fine = current_step.astype(np.float64)[i_indices]
            fft_f = np.fft.fft(f_fine, n=2 * M)
            autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
            current_c3 = np.max(np.abs(autoconv))
        
        T = T_start
        max_iter = 50000
        cooling_rate = (T_end / T_start) ** (1.0 / max_iter)
        
        restart_start = time.time()
        
        for iteration in range(max_iter):
            if time.time() - start_time > DEADLINE - 2.0:
                break
            if time.time() - restart_start > time_per_restart:
                break
            
            # Propose a move: flip 1 random position
            flip_idx = np.random.randint(N)
            old_val = current_step[flip_idx]
            
            if old_val == 1:
                fft_delta = fft_delta_cache[flip_idx]
                delta_conv = delta_conv_cache[flip_idx]
            else:
                fft_delta = -fft_delta_cache[flip_idx]
                delta_conv = delta_conv_cache[flip_idx]
            
            # Compute cross term
            # ifft(fft_f * fft_delta)
            cross_term = np.fft.ifft(fft_f * fft_delta).real * dx
            
            # Update autoconv in buffer
            np.add(autoconv, 2.0 * cross_term, out=new_autoconv_buf)
            np.add(new_autoconv_buf, delta_conv, out=new_autoconv_buf)
            
            new_c3 = np.max(np.abs(new_autoconv_buf))
            
            delta = new_c3 - current_c3
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
                # Accept the move
                current_step[flip_idx] = -old_val
                new_val = float(-old_val)
                f_fine[coarse_to_fine_arrays[flip_idx]] = new_val
                
                # Update fft_f incrementally
                fft_f += fft_delta
                # Copy buffer to autoconv
                np.copyto(autoconv, new_autoconv_buf)
                current_c3 = new_c3
                
                if new_c3 < best_c3:
                    best_c3 = new_c3
                    best_step = current_step.copy()
            
            T *= cooling_rate
    
    return best_step
