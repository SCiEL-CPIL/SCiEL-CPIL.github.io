import numpy as np
import time

N = 400
DEADLINE = 118.0  # Safety margin before 150s kill
M = 2 * N  # fine grid for accuracy
dx = 0.5 / M

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound: max |autoconv(t)| for the signed step function.
    """
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    f_fine = signed_step.astype(np.float64)[i_indices]
    
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def _spectral_candidate(omega, phi):
    """
    Generate a signed step function from spectral parameters.
    f(x) = sign(sin(omega * x + phi))
    """
    x = np.linspace(-0.25, 0.25, N + 1)[:-1]  # midpoints of intervals
    vals = np.sin(omega * x + phi)
    return np.sign(vals).astype(np.int8)

def _nelder_mead_spectral(start_time, deadline):
    """
    Optimize (omega, phi) using Nelder-Mead to minimize C3.
    """
    # Initial simplex: 3 points in 2D space
    # Try a few different initializations
    best_step = None
    best_c3 = np.inf
    
    initial_params_list = [
        (2 * np.pi * 2, 0.0),   # frequency 2
        (2 * np.pi * 3, 0.0),   # frequency 3
        (2 * np.pi * 4, 0.0),   # frequency 4
        (2 * np.pi * 1.5, 0.0), # frequency 1.5
        (2 * np.pi * 2.5, 0.0), # frequency 2.5
        (2 * np.pi * 5, 0.0),   # frequency 5
        (2 * np.pi * 2, np.pi/4),
        (2 * np.pi * 3, np.pi/4),
        (2 * np.pi * 4, np.pi/4),
    ]
    
    for init_omega, init_phi in initial_params_list:
        if time.time() > deadline - 5.0:
            break
        
        # Nelder-Mead implementation
        # Parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        tol = 1e-4
        max_iter = 200
        
        # Initial simplex
        simplex = [
            (init_omega, init_phi),
            (init_omega * 1.1, init_phi),
            (init_omega, init_phi + 0.5),
        ]
        
        # Evaluate initial points
        c3_values = []
        for params in simplex:
            step = _spectral_candidate(params[0], params[1])
            c3 = _compute_c3(step)
            c3_values.append(c3)
        
        # Track best
        best_idx = np.argmin(c3_values)
        best_c3_local = c3_values[best_idx]
        best_step_local = _spectral_candidate(simplex[best_idx][0], simplex[best_idx][1])
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Sort simplex by objective value
            order = np.argsort(c3_values)
            simplex = [simplex[i] for i in order]
            c3_values = [c3_values[i] for i in order]
            best_idx = 0
            worst_idx = len(simplex) - 1
            
            # Check convergence
            if abs(c3_values[0] - c3_values[-1]) < tol:
                break
            
            # Centroid of all points except the worst
            centroid = np.array([
                np.mean([p[0] for p in simplex[:-1]]),
                np.mean([p[1] for p in simplex[:-1]])
            ])
            
            # Reflection
            worst = simplex[worst_idx]
            reflected = centroid + alpha * (centroid - np.array(worst))
            step_reflected = _spectral_candidate(reflected[0], reflected[1])
            c3_reflected = _compute_c3(step_reflected)
            
            if c3_reflected < c3_values[0]:
                # Expansion
                expanded = centroid + gamma * (reflected - centroid)
                step_expanded = _spectral_candidate(expanded[0], expanded[1])
                c3_expanded = _compute_c3(step_expanded)
                
                if c3_expanded < c3_reflected:
                    simplex[worst_idx] = (expanded[0], expanded[1])
                    c3_values[worst_idx] = c3_expanded
                else:
                    simplex[worst_idx] = (reflected[0], reflected[1])
                    c3_values[worst_idx] = c3_reflected
            elif c3_reflected < c3_values[-1]:
                # Accept reflection
                simplex[worst_idx] = (reflected[0], reflected[1])
                c3_values[worst_idx] = c3_reflected
            else:
                # Contraction
                contracted = centroid + rho * (np.array(worst) - centroid)
                step_contracted = _spectral_candidate(contracted[0], contracted[1])
                c3_contracted = _compute_c3(step_contracted)
                
                if c3_contracted < c3_values[-1]:
                    simplex[worst_idx] = (contracted[0], contracted[1])
                    c3_values[worst_idx] = c3_contracted
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = (
                            simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0]),
                            simplex[0][1] + sigma * (simplex[i][1] - simplex[0][1])
                        )
                        step_shrink = _spectral_candidate(simplex[i][0], simplex[i][1])
                        c3_shrink = _compute_c3(step_shrink)
                        c3_values[i] = c3_shrink
            
            # Update best
            if c3_values[0] < best_c3_local:
                best_c3_local = c3_values[0]
                best_step_local = _spectral_candidate(simplex[0][0], simplex[0][1])
        
        # Update global best
        if best_c3_local < best_c3:
            best_c3 = best_c3_local
            best_step = best_step_local.copy()
    
    return best_step, best_c3

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

def _simulated_annealing(current_step, current_c3, start_time, deadline):
    """Run simulated annealing with incremental FFT updates."""
    best_step = current_step.copy()
    best_c3 = current_c3
    
    # Precompute fine grid mapping
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    
    # Build fine grid signal
    f_fine = current_step.astype(np.float64)[i_indices]
    
    # Compute initial autoconvolution
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
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
    
    # Precompute delta_conv for each coarse index
    delta_conv_cache = np.zeros((N, 2 * M), dtype=np.float64)
    for i in range(N):
        fd = fft_delta_cache[i]
        delta_conv_cache[i] = np.fft.ifft(fd * np.conj(fd)).real * dx
    
    # Simulated annealing parameters
    T_start = 0.03
    T_end = 0.0005
    max_iter = 50000
    cooling_rate = (T_end / T_start) ** (1.0 / max_iter)
    
    T = T_start
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5.0:
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
        
        # cross_term = ifft(fft_f * fft_delta).real * dx
        cross_term = np.fft.ifft(fft_f * fft_delta).real * dx
        
        new_autoconv = autoconv + 2 * cross_term + delta_conv
        new_c3 = np.max(np.abs(new_autoconv))
        
        delta = new_c3 - current_c3
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
            # Accept the move
            current_step[flip_idx] = -old_val
            new_val = float(-old_val)
            f_fine[coarse_to_fine_arrays[flip_idx]] = new_val
            
            # Update fft_f incrementally
            fft_f = fft_f + fft_delta
            autoconv = new_autoconv
            current_c3 = new_c3
            
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            pass
        
        T *= cooling_rate
    
    return best_step, best_c3

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Phase 1: Spectral optimization to find a good global structure
    spectral_step, spectral_c3 = _nelder_mead_spectral(start_time, DEADLINE)
    
    # Phase 2: Structured initialization as backup
    structured_step, structured_c3 = _structured_initialization()
    
    # Choose the best starting point
    if spectral_c3 < structured_c3:
        current_step = spectral_step.copy()
        current_c3 = spectral_c3
    else:
        current_step = structured_step.copy()
        current_c3 = structured_c3
    
    # Phase 3: Simulated annealing to refine
    best_step, best_c3 = _simulated_annealing(current_step, current_c3, start_time, DEADLINE)
    
    return best_step
