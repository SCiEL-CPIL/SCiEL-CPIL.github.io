"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute FFT size
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# Precompute basis vectors for incremental FFT updates
# basis[i][k] = exp(-2j * pi * k * i / _FSIZE)
# We store it as a matrix of shape (N, _FSIZE//2 + 1)
_FREQS = np.arange(_FSIZE // 2 + 1)
# Vectorized creation of basis
# angles = -2 * pi * outer(arange(N), arange(FSIZE//2+1)) / FSIZE
# This might be large: 400 * 513 complex numbers. ~2MB. Acceptable.
_ANGLES = -2 * np.pi * np.outer(np.arange(N), _FREQS) / _FSIZE
_BASIS = np.exp(1j * _ANGLES)

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2 * N - 1]

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _grad_smooth(h, beta=50.0):
    """
    Analytical gradient of the smooth objective.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    ac = _autoconv_fft(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        S = 0.0
        w = np.zeros_like(ac)
    else:
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        log_sum = np.log(np.sum(exp_scaled))
        S = m + log_sum / beta
        w = exp_scaled / np.sum(exp_scaled)
    
    v = w * np.sign(ac)
    
    # Gradient of S w.r.t h
    # dS/dh_i = 2 * sum_k v_k h_{k-i}
    # Using FFT for cross-correlation
    fsize_corr = 1
    while fsize_corr < 3 * N:
        fsize_corr *= 2
        
    V = np.fft.rfft(v, fsize_corr)
    H = np.fft.rfft(h, fsize_corr)
    C = np.fft.irfft(V * np.conj(H), fsize_corr)
    
    dS_dh = 2 * C[:N]
    
    grad = (dS_dh * s - 2 * S) / (s ** 3)
    
    return S / (s * s), grad

def _local_search_binary(h, deadline, max_passes=20):
    """
    Local search on binary sequence {-1, 1} using incremental FFT.
    h is a numpy array of 1s and -1s.
    """
    h = h.copy()
    # Ensure binary
    h = np.sign(h)
    h[h == 0] = 1
    
    best_val = _score(h)
    H = np.fft.rfft(h, _FSIZE)
    
    # Precompute integral
    integral = np.sum(h)
    
    improved = True
    pass_count = 0
    
    while improved and (deadline is None or time.time() < deadline) and pass_count < max_passes:
        improved = False
        pass_count += 1
        
        # 1. Single flips
        for i in range(N):
            if time.time() > deadline:
                break
            
            old_val = h[i]
            new_val = -old_val
            h[i] = new_val
            
            # Update FFT
            # Delta H = -2 * old_val * basis[i]
            # Because h[i] changed by (new - old) = -2 * old
            delta = -2.0 * old_val * _BASIS[i]
            H_new = H + delta
            
            # Compute convolution
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            max_abs_conv = np.max(np.abs(conv))
            
            # Update integral
            integral_new = integral - 2.0 * old_val
            
            if abs(integral_new) < 1e-9:
                score_new = 1e10
            else:
                score_new = max_abs_conv / (integral_new * integral_new)
            
            if score_new < best_val:
                best_val = score_new
                H = H_new
                integral = integral_new
                improved = True
            else:
                # Revert
                h[i] = old_val
                # H and integral revert implicitly by not updating them
        
        # 2. Block flips
        if time.time() > deadline:
            break
            
        block_sizes = [2, 3, 5, 7, 11]
        # Use a stride to reduce number of checks if needed, but N=400 is small
        # Total checks: sum(N-bs) for bs in sizes. ~ 2000 checks per pass.
        # Each check is ~ O(N_fft). 
        # 2000 * 1000 ops = 2M ops. Very fast.
        
        for bs in block_sizes:
            for start in range(0, N - bs + 1):
                if time.time() > deadline:
                    break
                
                # Flip block
                block_indices = np.arange(start, start + bs)
                old_block_vals = h[block_indices].copy()
                h[block_indices] = -h[block_indices]
                
                # Update FFT
                # Sum of deltas
                delta = np.zeros(_FSIZE // 2 + 1, dtype=complex)
                for idx in block_indices:
                    delta += -2.0 * old_block_vals[idx - start] * _BASIS[idx]
                
                H_new = H + delta
                
                # Compute convolution
                conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                max_abs_conv = np.max(np.abs(conv))
                
                # Update integral
                integral_change = -2.0 * np.sum(old_block_vals)
                integral_new = integral + integral_change
                
                if abs(integral_new) < 1e-9:
                    score_new = 1e10
                else:
                    score_new = max_abs_conv / (integral_new * integral_new)
                
                if score_new < best_val:
                    best_val = score_new
                    H = H_new
                    integral = integral_new
                    improved = True
                else:
                    # Revert
                    h[block_indices] = old_block_vals
                    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = None

    rng = np.random.default_rng(42)
    
    # Phase 1: Continuous optimization to find a good starting point
    # We use a few diverse initializations
    inits = []
    
    # 1. All ones (normalized)
    inits.append(np.ones(N) / N)
    
    # 2. Cosine window
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    
    # 3. Gaussian window
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-(x - mu)**2 / (2*sigma**2)))
    
    # 4. Chirp
    phase = np.pi * x**2 / N
    inits.append(np.cos(phase))
    
    # 5. Random binary sequences (good for max-abs problems)
    for i in range(5):
        inits.append(rng.choice([1.0, -1.0], N))
        
    # 6. Random smoothed
    for i in range(5):
        noise = rng.standard_normal(N)
        kernel = np.ones(10)/10
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)

    candidates = []
    
    for h_init in inits:
        if time.time() > deadline - 40:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Gradient descent on continuous space
        try:
            for beta in [20, 50]:
                if time.time() > deadline - 30:
                    break
                
                def obj(x):
                    val, grad = _grad_smooth(x, beta)
                    return val, grad
                
                res = minimize(obj, h_norm, method='L-BFGS-B', jac=True,
                               options={'maxiter': 100, 'disp': False, 'gtol': 1e-5})
                h_opt = res.x
                
                # Add to candidates
                candidates.append(h_opt)
                
                # Also add the binarized version
                h_bin = np.sign(h_opt)
                h_bin[h_bin == 0] = 1
                candidates.append(h_bin)
                
        except Exception:
            pass

    # Phase 2: Local search on the best candidates
    # We will take the top 10 candidates from Phase 1 (by score) and run local search on their binarized versions
    # Also include the original random binary inits
    
    # Evaluate candidates
    scored_candidates = []
    for h in candidates:
        if time.time() > deadline - 20:
            break
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        # Binarize for local search
        h_bin = np.sign(h)
        h_bin[h_bin == 0] = 1
        val = _score(h_bin)
        scored_candidates.append((val, h_bin))
    
    # Sort by score
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 10 unique-ish candidates
    # Simple deduplication by checking if identical to previous
    unique_candidates = []
    prev_h = None
    for val, h in scored_candidates:
        if prev_h is None or not np.array_equal(h, prev_h):
            unique_candidates.append((val, h))
            prev_h = h
        if len(unique_candidates) >= 10:
            break
    
    # If we didn't get enough, add some random binary sequences
    while len(unique_candidates) < 5:
        h_rand = rng.choice([1.0, -1.0], N)
        val = _score(h_rand)
        unique_candidates.append((val, h_rand))
        
    # Run local search on each
    for val, h in unique_candidates:
        if time.time() > deadline - 10:
            break
        
        # Update global best if this candidate is already better
        if val < best_score:
            best_score = val
            best_h = h.copy()
            
        # Run local search
        h_opt, val_opt = _local_search_binary(h, deadline=deadline - 5, max_passes=50)
        
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()

    # Final check: if no valid best_h found, return a default
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score(best_h)
        
    return best_h

# EVOLVE-BLOCK-END
