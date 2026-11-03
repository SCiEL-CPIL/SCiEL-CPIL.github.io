"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_fft_size(n):
    """Get the next power of 2 size for FFT of linear convolution of length n."""
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    n = len(h)
    fsize = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:2*n-1]

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _soft_max_abs(ac, beta=100.0):
    """Smooth approximation of max(|ac|) via log-sum-exp."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    """Smooth objective for optimization."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_chirp(n):
    """Generate a chirp sequence with flat magnitude spectrum."""
    i = np.arange(n)
    # Real chirp: cos(2*pi*i^2/n) has approximately flat magnitude spectrum
    return np.cos(2 * np.pi * i**2 / n)

def _gen_chirp2(n):
    """Generate a second chirp variant."""
    i = np.arange(n)
    return np.cos(np.pi * i**2 / n + np.pi/4)

def _gen_random_flat_spectrum(n, seed=42):
    """Generate random sequence with flat magnitude spectrum via inverse FFT."""
    rng = np.random.default_rng(seed)
    # Create random phase with constant magnitude
    fsize = _get_fft_size(n)
    H_mag = np.ones(fsize // 2 + 1)
    H_phase = rng.uniform(0, 2*np.pi, size=fsize // 2 + 1)
    H = H_mag * (np.cos(H_phase) + 1j * np.sin(H_phase))
    # Ensure Hermitian symmetry for real output
    H_full = np.zeros(fsize, dtype=complex)
    H_full[:fsize//2+1] = H
    H_full[fsize//2+1:] = np.conj(H[1:fsize//2+1][::-1])
    h = np.fft.ifft(H_full).real[:n]
    return h

def _gen_golden_ratio(n):
    """Generate a sequence based on golden ratio properties."""
    phi = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    frac = (i * phi) % 1.0
    return 2 * frac - 1

def _gen_alternating_gaussian(n, sigma_ratio=0.3):
    """Alternating signs with Gaussian envelope."""
    x = np.arange(n)
    mu = n / 2
    sigma = n * sigma_ratio
    gauss = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    alt = np.where(x % 2 == 0, 1.0, -1.0)
    return gauss * alt

def _local_search(h, max_passes=8, deadline=None):
    """
    Local search with sign flips and small perturbations.
    """
    h = h.copy()
    best_val = _score(h)
    s = np.sum(h)
    ac = _autoconv_fft(h)
    
    step_sizes = [0.15, 0.08, 0.04, 0.02, 0.01]
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        step = step_sizes[min(pass_num, len(step_sizes)-1)]
        
        # 1. Try flipping signs
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            old_val = h[i]
            h[i] = -old_val
            new_s = s - 2 * old_val
            
            if abs(new_s) < 1e-9:
                h[i] = old_val
                continue
            
            new_ac = _autoconv_fft(h)
            new_max = np.max(np.abs(new_ac))
            new_val = new_max / (new_s * new_s)
            
            if new_val < best_val:
                best_val = new_val
                s = new_s
                ac = new_ac
                improved = True
            else:
                h[i] = old_val
        
        # 2. Try small multiplicative perturbations
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            orig = h[i]
            for factor in [1 + step, 1 - step]:
                h[i] = orig * factor
                new_s = s + orig * (factor - 1)
                
                if abs(new_s) < 1e-9:
                    h[i] = orig
                    continue
                
                new_ac = _autoconv_fft(h)
                new_max = np.max(np.abs(new_ac))
                new_val = new_max / (new_s * new_s)
                
                if new_val < best_val:
                    best_val = new_val
                    s = new_s
                    ac = new_ac
                    improved = True
                    break
            else:
                h[i] = orig
        
        if not improved:
            break
            
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit
    
    best_score = np.inf
    best_h = np.ones(N)
    
    # Generate diverse initial candidates
    inits = []
    
    # 1. Chirp sequences (flat magnitude spectrum)
    inits.append(_gen_chirp(N))
    inits.append(_gen_chirp2(N))
    
    # 2. Random flat-spectrum sequences
    for seed in [42, 123, 456, 789]:
        inits.append(_gen_random_flat_spectrum(N, seed=seed))
    
    # 3. Golden ratio
    inits.append(_gen_golden_ratio(N))
    
    # 4. Alternating Gaussian
    inits.append(_gen_alternating_gaussian(N, sigma_ratio=0.25))
    inits.append(_gen_alternating_gaussian(N, sigma_ratio=0.35))
    
    # 5. Simple cosine
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    
    # 6. Constant
    inits.append(np.ones(N))
    
    for h_init in inits:
        if time.time() > deadline - 25:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to unit sum for numerical stability
        h_norm = h_init / s0
        
        # Gradient optimization with adaptive beta annealing
        try:
            betas = [20, 50, 100, 200, 400]
            current_h = h_norm.copy()
            
            for beta in betas:
                if time.time() > deadline - 15:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, current_h, method='L-BFGS-B', 
                               options={'maxiter': 100, 'disp': False, 'gtol': 1e-7})
                
                h_opt = res.x
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
                
                current_h = h_opt.copy()
                
        except Exception:
            pass

    # Local search on the best candidate
    if time.time() < deadline - 8:
        h_refined, val_refined = _local_search(best_h, max_passes=10, deadline=deadline-2)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
            
    # Final gradient polish with high beta
    if time.time() < deadline - 4:
        try:
            s_final = np.sum(best_h)
            if abs(s_final) > 1e-6:
                h_polish_init = best_h / s_final
                
                def obj_polish(x):
                    return _objective_smooth(x, beta=1000.0)
                
                res = minimize(obj_polish, h_polish_init, method='L-BFGS-B', 
                               options={'maxiter': 150, 'disp': False, 'gtol': 1e-9})
                
                h_polish = res.x
                if abs(np.sum(h_polish)) > 1e-6:
                    val_polish = _score(h_polish)
                    if val_polish < best_score:
                        best_score = val_polish
                        best_h = h_polish.copy()
        except Exception:
            pass

    # Ensure we return a valid sequence
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
