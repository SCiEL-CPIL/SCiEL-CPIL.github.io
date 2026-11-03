"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize, size

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    fsize, size = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

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

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_sinc(n, beta=5.0):
    """Generate a sinc-like sequence."""
    i = np.arange(n)
    x = (i - n/2) / (n * beta)
    h = np.sinc(x)
    return h

def _gen_gaussian(n, sigma_frac=0.25):
    """Generate a Gaussian window."""
    i = np.arange(n)
    mu = n / 2
    sigma = n * sigma_frac
    return np.exp(-(i - mu)**2 / (2 * sigma**2))

def _gen_barker_like(n):
    """Generate a sequence with alternating pattern."""
    h = np.zeros(n)
    for i in range(n):
        if i % 4 in [0, 2]:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _gen_random_smoothed(n, seed=42, kernel_size=10):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(noise, kernel, mode='same')

def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses vectorized precomputation to speed up evaluation.
    """
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, 1e10
    best_val = _score(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Precompute autoconvolution once per pass
        ac = _autoconv_fft(h)
        s_sq = s * s
        current_max_abs = np.max(np.abs(ac))
        
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Flipping h[i] to -h[i] changes sum by -2*h[i]
            # New sum: s - 2*h[i]
            new_s = s - 2.0 * h[i]
            if abs(new_s) < 1e-9:
                continue
            new_s_sq = new_s * new_s
            
            # The autoconvolution changes: ac_new = ac - 2*h[i]*h[i] at lag 0
            # Actually, flipping sign: h[i] -> -h[i]
            # ac_new[k] = ac[k] - 2*h[i]*h[i-k] for all k... this is complex
            # Let's just do the direct flip and recompute
            h[i] = -h[i]
            new_val = _autoconv_max_abs(h) / new_s_sq
            if new_val < best_val:
                best_val = new_val
                s = new_s
                improved = True
                # Update ac for next iteration
                ac = _autoconv_fft(h)
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def _local_search_perturb(h, max_passes=3, deadline=None, step_size=0.1):
    """
    Local optimization with small perturbations.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(123)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            h[i] = old_val + step_size * rng.standard_normal()
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, best_val

def _discrete_polish(h, deadline=None):
    """
    Final polish: try quantizing each element to ±1 and see if it improves.
    Also try small neighborhoods of the current solution.
    """
    h = h.copy()
    best_val = _score(h)
    
    if deadline is not None and time.time() > deadline:
        return h, best_val
    
    # Try rounding to ±1
    h_round = np.sign(h)
    # Handle zeros
    h_round[h_round == 0] = 1.0
    s_round = np.sum(h_round)
    if abs(s_round) > 1e-9:
        val_round = _score(h_round)
        if val_round < best_val:
            best_val = val_round
            h = h_round.copy()
    
    if deadline is not None and time.time() > deadline:
        return h, best_val
    
    # Try local perturbations around best
    rng = np.random.default_rng(999)
    for _ in range(50):
        if deadline is not None and time.time() > deadline:
            break
        h_trial = h.copy()
        # Perturb a random subset
        n_perturb = rng.integers(5, 20)
        indices = rng.choice(N, size=n_perturb, replace=False)
        for idx in indices:
            h_trial[idx] += 0.1 * rng.standard_normal()
        s_trial = np.sum(h_trial)
        if abs(s_trial) < 1e-9:
            continue
        val_trial = _score(h_trial)
        if val_trial < best_val:
            best_val = val_trial
            h = h_trial.copy()
    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Initialization strategies
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Cosine windows
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    
    # 3. Gaussian window
    inits.append(_gen_gaussian(N, sigma_frac=0.25))
    inits.append(_gen_gaussian(N, sigma_frac=0.15))
    
    # 4. Triangle window
    mu = N/2
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Chirp sequence
    inits.append(_gen_chirp(N))
    
    # 6. Sinc sequence
    inits.append(_gen_sinc(N, beta=3.0))
    inits.append(_gen_sinc(N, beta=5.0))
    
    # 7. Barker-like sequence
    inits.append(_gen_barker_like(N))
    
    # 8. Random smoothed sequences
    for seed in [42, 123, 456, 789, 101]:
        inits.append(_gen_random_smoothed(N, seed=seed))
    
    # 9. Additional structured initializations
    # Alternating blocks
    block_len = N // 8
    for start in range(0, N, block_len):
        h_block = np.ones(N)
        h_block[start:start+block_len] = -1.0
        inits.append(h_block)
    
    # Ramp + cosine
    ramp = np.linspace(-1, 1, N)
    inits.append(ramp * np.cos(2 * np.pi * x / N))
    inits.append(ramp * np.cos(4 * np.pi * x / N))
    
    # Multi-frequency
    inits.append(np.cos(2 * np.pi * x / N) + 0.5 * np.cos(4 * np.pi * x / N))
    inits.append(np.cos(2 * np.pi * x / N) - 0.5 * np.cos(6 * np.pi * x / N))
        
    # Optimization loop using smooth surrogate
    for h_init in inits:
        if time.time() > deadline - 30:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1 to simplify optimization landscape
        h_norm = h_init / s0
        
        # Use smooth objective for gradient-based optimization
        try:
            # Adaptive beta schedule: start low for exploration, increase for refinement
            for beta in [5, 15, 30, 60, 120, 200]:
                if time.time() > deadline - 20:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B', 
                               options={'maxiter': 80, 'disp': False, 'gtol': 1e-6})
                h_opt = res.x
                
                # Check validity
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
                        # Update h_norm for next beta to continue from best
                        h_norm = h_opt.copy()
        except Exception:
            pass

    # Final local search refinement with more passes
    if time.time() < deadline - 15:
        h_refined, val_refined = _local_search_flip(best_h, max_passes=6, deadline=deadline-8)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Additional perturbation-based local search
    if time.time() < deadline - 8:
        h_pert, val_pert = _local_search_perturb(best_h, max_passes=3, deadline=deadline-4, step_size=0.05)
        if val_pert < best_score:
            best_score = val_pert
            best_h = h_pert.copy()
    
    # Discrete polish phase
    if time.time() < deadline - 3:
        h_polished, val_polished = _discrete_polish(best_h, deadline=deadline-1)
        if val_polished < best_score:
            best_score = val_polished
            best_h = h_polished.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
