import numpy as np
import time

N = 400
DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound: max |autoconv(t)| for the signed step function.
    """
    M = 2 * N  # fine grid for accuracy
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

def _structured_initialization():
    """Generate multiple structured candidates and return the best."""
    candidates = []
    
    # All ones (baseline)
    candidates.append(np.ones(N, dtype=np.int8))
    
    # Alternating
    candidates.append(np.tile(np.array([1, -1], dtype=np.int8), N // 2))
    
    # Random
    np.random.seed(42)
    for _ in range(10):
        candidates.append(np.random.choice([-1, 1], size=N).astype(np.int8))
    
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
        c3 = _compute_c3(c)
        if c3 < best_c3:
            best_c3 = c3
            best = c.copy()
    
    return best, best_c3

def construct():
    start_time = time.time()
    
    # Get a good structured start
    best_step, best_c3 = _structured_initialization()
    
    # Precompute fine grid mapping
    M = 2 * N
    dx = 0.5 / M
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    
    # coarse_to_fine: list of fine indices for each coarse index
    coarse_to_fine = [[] for _ in range(N)]
    for m in range(M):
        coarse_to_fine[i_indices[m]].append(m)
    coarse_to_fine_arrays = [np.array(coarse_to_fine[i], dtype=np.int64) for i in range(N)]
    
    # Precompute FFTs for delta patterns (unit change in x_i)
    # If x_i increases by 1, u changes by +1 at coarse_to_fine[i]
    fft_delta_unit_cache = np.zeros((N, 2 * M), dtype=np.complex128)
    for i in range(N):
        delta_f = np.zeros(M, dtype=np.float64)
        delta_f[coarse_to_fine_arrays[i]] = 1.0
        fft_delta_unit_cache[i] = np.fft.fft(delta_f, n=2 * M)
        
    # Current continuous state
    x = best_step.astype(np.float64)
    
    # Current fine grid and FFT
    u = x[i_indices]
    F_u = np.fft.fft(u, n=2 * M)
    Y_u = np.fft.ifft(F_u * np.conj(F_u)).real * dx
    current_c3 = np.max(np.abs(Y_u))
    
    best_x = x.copy()
    best_c3_cont = current_c3
    
    # Continuous Coordinate Descent
    while time.time() - start_time < DEADLINE - 10.0:
        # Random order of coordinates
        order = np.random.permutation(N)
        
        for i in order:
            if time.time() - start_time > DEADLINE - 10.0:
                break
            
            # Determine valid range for delta
            # x[i] + delta must be in [-1, 1]
            lo = -1.0 - x[i]
            hi = 1.0 - x[i]
            
            if hi <= lo:
                continue
                
            # Ternary search for optimal delta
            # Objective: max |IFFT((F_u + delta * F_delta)^2)|
            F_delta = fft_delta_unit_cache[i]
            
            def obj(delta):
                F_new = F_u + delta * F_delta
                Y_new = np.fft.ifft(F_new * np.conj(F_new)).real
                return np.max(np.abs(Y_new))
            
            a, b = lo, hi
            # Golden section search
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = obj(c)
            fd = obj(d)
            
            for _ in range(20): # ~20 iterations for precision
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = obj(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = obj(d)
                
            delta_opt = (a + b) / 2.0
            
            # Update x and u, F_u, Y_u
            if abs(delta_opt) > 1e-6:
                x[i] += delta_opt
                # Clip just in case
                if x[i] > 1.0: x[i] = 1.0
                elif x[i] < -1.0: x[i] = -1.0
                
                # Update u
                u[coarse_to_fine_arrays[i]] = x[i]
                
                # Update F_u
                F_u += delta_opt * F_delta
                
                # Update Y_u
                Y_u = np.fft.ifft(F_u * np.conj(F_u)).real * dx
                current_c3 = np.max(np.abs(Y_u))
                
                if current_c3 < best_c3_cont:
                    best_c3_cont = current_c3
                    best_x = x.copy()
                    
    # Discretize
    x_discrete = np.sign(best_x)
    # Handle zeros
    x_discrete[x_discrete == 0] = 1.0
    
    # Final Polish: Local Search on discrete
    # Convert to int8 for _compute_c3
    step_discrete = x_discrete.astype(np.int8)
    c3_discrete = _compute_c3(step_discrete)
    
    if c3_discrete < best_c3:
        best_step = step_discrete
        best_c3 = c3_discrete
        
    # Quick local search on discrete to see if we can improve
    # Use the same FFT cache logic but for discrete flips
    # This part is optional if time is tight, but good for polish
    if time.time() - start_time < DEADLINE - 2.0:
        # Re-initialize discrete state
        current_step = best_step.copy()
        current_c3 = best_c3
        
        # Precompute fft_delta for discrete flips (delta = +/- 2)
        fft_delta_cache = np.zeros((N, 2 * M), dtype=np.complex128)
        for i in range(N):
            delta_f = np.zeros(M, dtype=np.float64)
            delta_f[coarse_to_fine_arrays[i]] = -2.0  # flip from +1 to -1
            fft_delta_cache[i] = np.fft.fft(delta_f, n=2 * M)
            
        # Current fine grid and FFT
        f_fine = current_step.astype(np.float64)[i_indices]
        fft_f = np.fft.fft(f_fine, n=2 * M)
        autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
        
        # Simple SA / Hill Climbing
        T = 0.01
        for _ in range(5000):
            if time.time() - start_time > DEADLINE - 2.0:
                break
            
            flip_idx = np.random.randint(N)
            old_val = current_step[flip_idx]
            
            if old_val == 1:
                fft_delta = fft_delta_cache[flip_idx]
            else:
                fft_delta = -fft_delta_cache[flip_idx]
                
            cross_term = np.fft.ifft(fft_f * fft_delta).real * dx
            delta_conv = np.fft.ifft(fft_delta * np.conj(fft_delta)).real * dx
            
            new_autoconv = autoconv + 2 * cross_term + delta_conv
            new_c3 = np.max(np.abs(new_autoconv))
            
            if new_c3 < current_c3:
                # Accept
                current_step[flip_idx] = -old_val
                new_val = float(-old_val)
                f_fine[coarse_to_fine_arrays[flip_idx]] = new_val
                fft_f = fft_f + fft_delta
                autoconv = new_autoconv
                current_c3 = new_c3
                
                if current_c3 < best_c3:
                    best_c3 = current_c3
                    best_step = current_step.copy()
            else:
                # Reject (or accept with prob if T > 0, but let's stick to hill climbing for speed)
                pass
                
    return best_step
