"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import eigh

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

def _build_circulant_matrix():
    """
    Build the circulant matrix C such that C @ h gives the circular autoconvolution.
    For circular convolution, C is circulant with first row [h_0, h_{n-1}, ..., h_1].
    But we want the matrix M such that h^T M h = sum_k (sum_i h_i h_{i+k})^2 for circular conv.
    
    Actually, for circular autoconvolution: ac_k = sum_i h_i * h_{i+k mod n}
    Then h^T C h = sum_k ac_k^2 where C is the circulant matrix with first row being
    the autocorrelation of the impulse response... 
    
    Let me think differently. The circular autoconvolution is:
    ac_k = sum_i h_i h_{i+k mod n}
    
    We can write this as: ac = F^{-1} |F(h)|^2 where F is DFT.
    
    So h^T C h = sum_k |F(h)_k|^2 * ... no that's just ||h||^4.
    
    Actually: sum_k ac_k^2 = sum_k |F(h)_k|^4 (by Parseval for circular conv)
    
    But we want max_k |ac_k|, not sum_k ac_k^2.
    
    For the spectral approach: the circular autoconvolution vector is:
    ac = IFFT(|FFT(h)|^2)
    
    This is NOT a simple quadratic form in h. It's quartic.
    
    However, if we fix the sum S = sum(h), we can decompose h = S/n * 1 + v where 1^T v = 0.
    Then ac = ac_S + ac_v + cross terms.
    
    This is getting complex. Let me use a different approach:
    Use the fact that for the max absolute value, we can use the dual formulation:
    max_k |ac_k| = max_{||u||_2=1} u^T D ac where D is a diagonal matrix selecting the max component.
    
    Actually, let me just use the smooth surrogate but with a better initialization
    from spectral considerations.
    """
    pass

def _spectral_init(n):
    """
    Generate initial candidates using spectral properties.
    The all-ones vector gives ac_k = n for all k, so max|ac|/S^2 = n/n^2 = 1/n.
    We want to minimize this ratio.
    
    For a sequence with sum S, the 'average' autoconvolution is S^2/n (by Parseval).
    So the minimum possible max|ac| is at least S^2/n, giving ratio >= 1/n.
    
    To achieve this, we need |ac_k| = S^2/n for all k, which means |FFT(h)_k| is constant.
    This is achieved by h being a scaled version of a chirp or similar.
    """
    candidates = []
    
    # Chirp: h_i = exp(2*pi*i*i/n) has flat magnitude spectrum
    i = np.arange(n)
    chirp = np.cos(2 * np.pi * i**2 / n)
    candidates.append(chirp)
    
    # Real chirp variant
    chirp2 = np.cos(np.pi * i**2 / n)
    candidates.append(chirp2)
    
    # Alternating with Gaussian
    gauss = np.exp(-0.5 * ((i - n/2) / (n*0.2))**2)
    alt = np.where(i % 2 == 0, 1.0, -1.0)
    candidates.append(gauss * alt)
    
    # Random with zero mean component
    rng = np.random.default_rng(123)
    for seed in [42, 123, 456]:
        rng_s = np.random.default_rng(seed)
        v = rng_s.standard_normal(n)
        v -= np.mean(v)  # zero mean
        candidates.append(v)
    
    return candidates

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _soft_max_abs(ac, beta=100.0):
    """
    Smooth approximation of max(|ac|) via log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    # Clip to avoid overflow
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    """
    Smooth objective for optimization.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_gaussian(n, sigma_ratio=0.25):
    """Generate a Gaussian sequence."""
    x = np.arange(n)
    mu = n / 2
    sigma = n * sigma_ratio
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_barker_like(n):
    """Generate a pseudo-random Barker-like sequence."""
    h = np.ones(n)
    for i in range(1, n):
        h[i] = 1.0 if ((i * 7 + 3) % 11) < 5 else -1.0
    return h

def _gen_random_smoothed(n, seed=42, kernel_size=10):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(noise, kernel, mode='same')
    return smooth

def _gen_triangle(n):
    """Generate a triangle window."""
    x = np.arange(n)
    mu = n / 2
    return np.maximum(0, 1 - np.abs(x - mu) / mu)

def _gen_golden_ratio(n):
    """Generate a sequence based on golden ratio properties."""
    phi = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    frac = (i * phi) % 1.0
    return 2 * frac - 1

def _local_search(h, max_passes=5, deadline=None):
    """
    Local search with O(N) incremental updates for sign flips.
    Uses the exact score function.
    Optimized for time awareness.
    """
    h = h.copy()
    best_val = _score(h)
    s = np.sum(h)
    
    # Precompute autoconvolution for incremental updates
    ac = _autoconv_fft(h)
    max_abs_ac = np.max(np.abs(ac))
    best_val = max_abs_ac / (s * s)
    
    # Adaptive step sizes for perturbation
    step_sizes = [0.2, 0.1, 0.05, 0.02]
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        step = step_sizes[min(pass_num, len(step_sizes)-1)]
        
        # 1. Try flipping signs with incremental update
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            old_val = h[i]
            h[i] = -old_val
            new_s = s - 2 * old_val
            
            if abs(new_s) < 1e-9:
                h[i] = old_val
                continue
            
            # Incremental autoconvolution update:
            # ac_new[k] = ac_old[k] - 4*old_val*sum_j h[j]*delta(j+k-i) + ... 
            # Actually: ac[k] = sum_j h[j]*h[j+k]
            # When h[i] -> -h[i], the change is:
            # ac_new[k] = ac_old[k] - 2*old_val*h[(k-i) mod n] - 2*old_val*h[(i-k) mod n] + 4*old_val^2*delta(k)
            # For linear (not circular) autoconv, this is more complex.
            # Let's just recompute for correctness, but use a faster method.
            
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
    deadline = t_start + 145.0 # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N)
    
    # Initialize with a few diverse candidates
    inits = []
    
    # 1. Constant
    inits.append(np.ones(N))
    
    # 2. Gaussian
    inits.append(_gen_gaussian(N))
    
    # 3. Chirp
    inits.append(_gen_chirp(N))
    
    # 4. Barker-like
    inits.append(_gen_barker_like(N))
    
    # 5. Random Smoothed
    inits.append(_gen_random_smoothed(N, seed=1))
    inits.append(_gen_random_smoothed(N, seed=2))
    inits.append(_gen_random_smoothed(N, seed=3))
    
    # 6. Cosine window
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    
    # 7. Alternating signs with Gaussian envelope
    gauss_env = _gen_gaussian(N, sigma_ratio=0.3)
    alt_signs = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    inits.append(gauss_env * alt_signs)
    
    # 8. Triangle
    inits.append(_gen_triangle(N))
    
    # 9. Golden Ratio
    inits.append(_gen_golden_ratio(N))
    
    # 10. Spectral inits
    inits.extend(_spectral_init(N))

    for h_init in inits:
        if time.time() > deadline - 30:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Gradient Optimization with adaptive beta annealing
        try:
            # Start with lower beta for broader search, then refine
            betas = [10, 30, 80, 200, 500]
            current_h = h_norm.copy()
            
            for beta in betas:
                if time.time() > deadline - 20:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, current_h, method='L-BFGS-B', 
                               options={'maxiter': 150, 'disp': False, 'gtol': 1e-6})
                
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

    # Final Local Search on the best candidate found so far
    if time.time() < deadline - 10:
        h_refined, val_refined = _local_search(best_h, max_passes=10, deadline=deadline-2)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
            
    # Optional: One more gradient polish on the refined result with high beta
    if time.time() < deadline - 5:
        try:
            s_final = np.sum(best_h)
            if abs(s_final) > 1e-6:
                h_polish_init = best_h / s_final
                
                def obj_polish(x):
                    return _objective_smooth(x, beta=1000.0)
                
                res = minimize(obj_polish, h_polish_init, method='L-BFGS-B', 
                               options={'maxiter': 100, 'disp': False, 'gtol': 1e-8})
                
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
