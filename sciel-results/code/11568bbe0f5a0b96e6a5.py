"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

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
    """Smooth objective for gradient-based optimization."""
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
    n = len(h)
    ac = _autoconv_fft(h)
    size = len(ac)
    
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return np.zeros(n)
    
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    w = exp_scaled / sum_exp  # weights for soft-max
    
    g = w * beta * np.sign(ac)
    
    fsize = 1
    while fsize < 2 * n + size:
        fsize *= 2
    
    H = np.fft.rfft(h, fsize)
    G = np.fft.rfft(np.concatenate([g, np.zeros(fsize - size)]), fsize)
    
    term1 = np.fft.irfft(np.conj(G) * H, fsize)[:n]
    
    G_rev = np.fft.rfft(np.concatenate([g[::-1], np.zeros(fsize - size)]), fsize)
    term2 = np.fft.irfft(np.conj(G_rev) * H, fsize)[:n]
    
    d_sm_dh = term1 + term2
    
    S = np.sum(h)
    if abs(S) < 1e-9:
        return np.zeros(n)
    
    sm = _soft_max_abs(ac, beta)
    grad = d_sm_dh / (S * S) - 2 * sm / (S ** 3)
    
    return grad

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_barker_like(n):
    """Generate a sequence with good autocorrelation properties."""
    h = np.zeros(n)
    for i in range(n):
        if i % 4 == 0:
            h[i] = 1.0
        elif i % 4 == 1:
            h[i] = -1.0
        elif i % 4 == 2:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _fast_local_search_flip(h, deadline=None, max_passes=10):
    """
    Fast local optimization using incremental FFT updates.
    """
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    
    ac = _autoconv_fft(h)
    S = np.sum(h)
    best_val = np.max(np.abs(ac)) / (S * S)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            old_hi = h[i]
            h[i] = -old_hi
            S -= 2 * old_hi
            
            if abs(S) < 1e-9:
                h[i] = old_hi
                S += 2 * old_hi
                continue
            
            j_min = max(0, i - n + 1)
            j_max = min(size - 1, i + n - 1)
            
            delta = np.zeros(size)
            for j in range(j_min, j_max + 1):
                d = 0.0
                if 0 <= j - i < n:
                    d += -2 * old_hi * h[j - i]
                if 0 <= i - j < n:
                    d += -2 * old_hi * h[i - j]
                delta[j] = d
            
            ac_new = ac + delta
            new_val = np.max(np.abs(ac_new)) / (S * S)
            
            if new_val < best_val:
                best_val = new_val
                ac = ac_new
                improved = True
            else:
                h[i] = old_hi
                S += 2 * old_hi
                ac = ac - delta
        
        if not improved:
            break
    
    return h, best_val

def _gradient_polish(h, deadline=None, max_iter=50):
    """
    Gradient-based polish using the smooth objective and analytical gradient.
    """
    h = h.copy()
    S = np.sum(h)
    if abs(S) < 1e-9:
        return h, _score(h)
    
    h = h / S
    
    best_val = _score(h)
    beta = 50.0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        try:
            obj_val = _objective_smooth(h, beta)
            grad = _grad_smooth(h, beta)
            
            step = 0.01
            for _ in range(10):
                h_new = h - step * grad
                S_new = np.sum(h_new)
                if abs(S_new) < 1e-9:
                    step *= 0.5
                    continue
                h_new = h_new / S_new
                val_new = _score(h_new)
                if val_new < best_val:
                    h = h_new
                    best_val = val_new
                    break
                else:
                    step *= 0.5
            
            if step < 1e-8:
                break
        except Exception:
            break
    
    return h, best_val

def _spectral_design(n, deadline=None):
    """
    Design a sequence via spectral optimization.
    We optimize the amplitude spectrum |H(f)| to minimize the peak of the 
    inverse FFT of |H(f)|^2 (which is the autoconvolution), subject to 
    the constraint that the time-domain signal has a fixed sum.
    
    This is equivalent to: minimize max|IFFT(|H|^2)| subject to sum(h)=1.
    
    We parameterize h in the frequency domain and optimize the real-valued 
    amplitude spectrum.
    """
    t_start = time.time()
    
    # FFT size
    fsize = 1
    while fsize < 2 * n - 1:
        fsize *= 2
    
    # Initial spectrum: flat (corresponds to delta in time, but we want smooth)
    # Start with a smooth window in frequency
    freqs = np.arange(fsize // 2)
    # Low-pass filter to start
    H_init = np.zeros(fsize // 2 + 1)
    cutoff = fsize // 4
    H_init[:cutoff] = 1.0
    H_init[cutoff:cutoff+10] = np.linspace(1, 0, 10)
    
    # Convert to time domain to get initial h
    h_init = np.fft.irfft(H_init, fsize)[:n]
    s_init = np.sum(h_init)
    if abs(s_init) < 1e-9:
        return None, np.inf
    h_init = h_init / s_init
    
    best_h = h_init.copy()
    best_val = _score(h_init)
    
    # Optimize in frequency domain
    # Variable: real-valued amplitude spectrum A (length fsize//2+1)
    # Constraint: the resulting time-domain signal must have sum = 1
    
    # Actually, let's optimize h directly but use a spectral penalty
    # to encourage low PAPR.
    
    # Alternative: use the fact that min max|autoconv| is related to 
    # minimizing the L-infinity norm of the autoconvolution.
    
    # Let's try a different approach: iterative water-filling in frequency
    # to flatten the power spectrum, which minimizes the peak autoconv.
    
    # Actually, the optimal sequence for minimizing max|autoconv|/S^2 
    # is related to Zadoff-Chu or CAZAC sequences, but those are complex.
    # For real sequences, we want to minimize the peak of the real autoconv.
    
    # Let's do gradient descent on the amplitude spectrum.
    
    def _obj_spectrum(A):
        """Objective given amplitude spectrum A."""
        # Reconstruct complex spectrum with random phases (fixed)
        # This is tricky. Let's just optimize h directly with a 
        # spectral regularization.
        pass
    
    # Simpler approach: use the existing gradient optimizer but with 
    # a better initialization from spectral considerations.
    
    # Initialize with a raised cosine in time, which has a smooth spectrum
    x = np.arange(n)
    h = np.cos(np.pi * x / (n + 1))**2
    h = h - np.mean(h)  # Zero mean to reduce DC peak
    s = np.sum(h)
    if abs(s) < 1e-9:
        h = np.ones(n) / n
        return h, _score(h)
    h = h / s
    
    # Optimize
    beta = 40.0
    h_opt = h.copy()
    
    for it in range(100):
        if deadline is not None and time.time() > deadline:
            break
        
        obj_val = _objective_smooth(h_opt, beta)
        grad = _grad_smooth(h_opt, beta)
        
        # Gradient descent with backtracking
        step = 0.05
        for _ in range(15):
            h_new = h_opt - step * grad
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-9:
                step *= 0.5
                continue
            h_new = h_new / s_new
            val_new = _score(h_new)
            if val_new < best_val:
                h_opt = h_new
                best_val = val_new
                break
            else:
                step *= 0.5
        
        if step < 1e-8:
            break
    
    return h_opt, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit

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
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-(x - mu)**2 / (2*sigma**2)))
    
    # 4. Triangle window
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Chirp sequence
    inits.append(_gen_chirp(N))
    
    # 6. Barker-like sequence
    inits.append(_gen_barker_like(N))
    
    # 7. Random smoothed sequences
    rng = np.random.default_rng(42)
    for i in range(3):
        noise = rng.standard_normal(N)
        kernel = np.ones(10)/10
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
    
    # 8. Sparse random with smoothing
    for i in range(3):
        sparse = rng.choice([-1, 1], size=N)
        kernel = np.ones(5)/5
        smooth = np.convolve(sparse, kernel, mode='same')
        inits.append(smooth)
    
    # 9. Spectral design initialization
    if time.time() < deadline - 60:
        h_spec, val_spec = _spectral_design(N, deadline=deadline - 50)
        if h_spec is not None and val_spec < best_score:
            best_score = val_spec
            best_h = h_spec.copy()
    
    # Optimization loop using smooth surrogate
    time_per_init = 10.0
    for h_init in inits:
        if time.time() > deadline - 30:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        try:
            for beta in [30, 60]:
                if time.time() > deadline - 25:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                def grad(x):
                    return _grad_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B', jac=grad,
                               options={'maxiter': 60, 'disp': False, 'gtol': 1e-4})
                h_opt = res.x
                
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
        except Exception:
            pass

    # Final local search refinement
    if time.time() < deadline - 10:
        h_refined, val_refined = _fast_local_search_flip(best_h, deadline=deadline-5, max_passes=8)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()

    # Gradient polish
    if time.time() < deadline - 3:
        h_polished, val_polished = _gradient_polish(best_h, deadline=deadline-1, max_iter=30)
        if val_polished < best_score:
            best_score = val_polished
            best_h = h_polished.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
