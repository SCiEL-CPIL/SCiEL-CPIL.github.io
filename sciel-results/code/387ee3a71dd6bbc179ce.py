"""Rewritten program: signed step heights minimizing max|autoconv|/(sum h)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute FFT size
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    H = np.fft.rfft(h, _FFT_SIZE)
    ac = np.fft.irfft(H * H, _FFT_SIZE)
    return ac[:2 * N - 1]

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

def _soft_max_abs(ac, beta=100.0):
    """
    Smooth approximation of max(|ac|) via log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    """
    Smooth objective for optimization: soft_max_abs(autoconv(h)) / (sum h)^2.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm, _ = _soft_max_abs_grad(ac, beta)
    return sm / (s * s)

def _objective_smooth_grad(h, beta=100.0):
    """
    Smooth objective with analytical gradient for L-BFGS-B.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    ac = _autoconv_fft(h)
    sm, grad_ac = _soft_max_abs_grad(ac, beta)
    value = sm / (s * s)
    # Gradient
    g_fft = np.fft.rfft(grad_ac, _FFT_SIZE)
    H = np.fft.rfft(h, _FFT_SIZE)
    grad_h = 2.0 * np.fft.irfft(g_fft * np.conj(H), _FFT_SIZE)[:N]
    grad_obj = grad_h / (s * s) - 2.0 * sm / (s * s * s) * np.ones(N)
    return value, grad_obj

def _spectral_papr_objective(h, beta=100.0):
    """
    Spectral PAPR objective using Parseval identity.
    The autoconvolution is the inverse FFT of |H|^2.
    We minimize max|IFFT(|H|^2)| / (sum h)^2.
    Using Parseval: the peak of the time-domain autoconv relates to spectral properties.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    H = np.fft.rfft(h, _FFT_SIZE)
    power = np.abs(H) ** 2
    # The autoconvolution is IFFT(H^2), not IFFT(|H|^2)
    # But we can use the spectral representation for PAPR-like reduction
    ac = np.fft.irfft(H * H, _FFT_SIZE)
    ac = ac[:2 * N - 1]
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _soft_max_abs_grad(ac, beta=100.0):
    """
    Compute soft_max_abs(ac) and its gradient w.r.t. ac.
    Returns (value, grad_ac) where grad_ac has same shape as ac.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(ac)
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -100, 100)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    value = m + np.log(sum_exp) / beta
    # Gradient: d/dac_k [soft_max] = (1/beta) * exp(beta*|ac_k|-beta*m) / sum_exp * sign(ac_k)
    grad_ac = (exp_scaled / sum_exp) * np.sign(ac) / beta
    return value, grad_ac

def _grad_soft_obj(h, beta=100.0):
    """
    Compute gradient of _objective_smooth w.r.t. h using FFT-based autoconv gradient.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return np.zeros_like(h)
    ac = _autoconv_fft(h)
    sm, grad_ac = _soft_max_abs_grad(ac, beta)
    # Gradient of sm w.r.t. h:
    # ac[k] = sum_j h[j]*h[k-j], so dac[k]/dh[i] = h[k-i] + h[i-k] (with bounds)
    # In FFT domain: if A = H^2, then grad_h[sum_k g_k * A_k] = 2 * IFFT(FFT(g) * conj(H))
    # where g_k = grad_ac[k]
    g_fft = np.fft.rfft(grad_ac, _FFT_SIZE)
    H = np.fft.rfft(h, _FFT_SIZE)
    grad_h = 2.0 * np.fft.irfft(g_fft * np.conj(H), _FFT_SIZE)[:N]
    # Chain rule for division by s^2:
    # d(sm/s^2)/dh = (grad_h * s^2 - sm * 2*s * ones) / s^4 = grad_h/s^2 - 2*sm/(s^3)
    grad_obj = grad_h / (s * s) - 2.0 * sm / (s * s * s) * np.ones(N)
    return grad_obj

def _spectral_papr_reduce(h, n_iter=50, gamma=0.5, deadline=None):
    """
    Gradient-based reduction of max|autoconv| using the soft-max surrogate.
    """
    h = h.copy().astype(float)
    s_target = np.sum(h)
    if abs(s_target) < 1e-9:
        return h
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        grad = _grad_soft_obj(h, beta=100.0)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        h = h - gamma * grad / grad_norm
        # Re-normalize to preserve sum
        s = np.sum(h)
        if abs(s) > 1e-9:
            h = h * (s_target / s)
    return h

def _spectral_init(n):
    """
    Generate initial sequence using spectral flatness principles.
    A flat spectrum corresponds to a delta-like time-domain signal.
    For minimizing autoconv peaks, we want a signal whose |H|^2 is as flat as possible,
    which means h should be close to white noise but with controlled phase.
    """
    rng = np.random.default_rng(7)
    # Start with random phases (flat magnitude spectrum)
    freqs = np.fft.rfftfreq(_FFT_SIZE) * 2 * np.pi
    # Only use frequencies that correspond to our signal length
    valid_freqs = np.abs(freqs) <= np.pi  # Nyquist
    
    # Generate random phases
    phases = rng.uniform(0, 2 * np.pi, size=_FFT_SIZE // 2 + 1)
    
    # Start with flat magnitude
    H = np.exp(1j * phases)
    
    # Convert to time domain and truncate
    h = np.fft.irfft(H, _FFT_SIZE)[:n]
    
    # Normalize
    s = np.sum(h)
    if abs(s) < 1e-9:
        return np.ones(n)
    return h / s

def _gen_barker_like(n):
    """
    Generate a Barker-like sequence: +1/-1 with good autocorrelation properties.
    """
    rng = np.random.default_rng(123)
    h = rng.choice([-1.0, 1.0], size=n)
    kernel = np.array([0.2, 0.3, 0.3, 0.2])
    h = np.convolve(h, kernel, mode='same')
    return h

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_gaussian(n):
    """Generate a Gaussian-shaped sequence."""
    x = np.arange(n)
    mu = n / 2
    sigma = n / 4
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _gen_random_smooth(n, seed=42):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(8) / 8
    return np.convolve(noise, kernel, mode='same')

def _local_search(h, max_passes=12, deadline=None):
    """
    Local search with incremental FFT updates for efficiency.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(42)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try flipping signs at random positions
        for _ in range(N):
            if deadline is not None and time.time() > deadline:
                break
            i = rng.integers(N)
            orig = h[i]
            h[i] = -orig
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = orig
        
        # Try small multiplicative perturbations
        for _ in range(N):
            if deadline is not None and time.time() > deadline:
                break
            i = rng.integers(N)
            orig = h[i]
            found = False
            for factor in [1.05, 0.95, 1.1, 0.9]:
                if deadline is not None and time.time() > deadline:
                    break
                h[i] = orig * factor
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                    found = True
                    break
            if not found:
                h[i] = orig
        
        if not improved:
            break
    
    return h, best_val

def _epigraph_objective(h, t, beta=100.0):
    """
    Epigraph formulation: minimize t subject to soft_max_abs(ac) <= t * (sum h)^2.
    This is handled by the optimizer; we just provide the objective.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    # Penalty for constraint violation
    violation = np.maximum(0, sm - t * s * s)
    return t + 10.0 * violation

def _high_precision_polish(h, deadline=None):
    """
    High-precision polish using tighter tolerances and more iterations.
    """
    h = h.copy()
    current_score = _score(h)
    
    # Use very tight tolerances with analytical gradient
    try:
        def obj_tight(x):
            return _objective_smooth_grad(x, beta=500.0)
        
        res = minimize(obj_tight, h, method='L-BFGS-B', jac=True,
                       options={'maxiter': 500, 'disp': False, 'gtol': 1e-10, 'ftol': 1e-12})
        h_opt = res.x
        if abs(np.sum(h_opt)) > 1e-6:
            val = _score(h_opt)
            if val < current_score:
                h = h_opt
                current_score = val
    except Exception:
        pass
    
    # Final local search with very small steps
    rng = np.random.default_rng(7)
    for _ in range(20):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for _ in range(N):
            if deadline is not None and time.time() > deadline:
                break
            i = rng.integers(N)
            orig = h[i]
            found = False
            for factor in [1.01, 0.99, 1.02, 0.98]:
                h[i] = orig * factor
                new_val = _score(h)
                if new_val < current_score:
                    current_score = new_val
                    improved = True
                    found = True
                    break
            if not found:
                h[i] = orig
        if not improved:
            break
    
    return h, current_score

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_h = np.ones(N)
    
    base_score = _score(best_h)
    best_score = base_score
    
    inits = []
    
    # 1. Constant
    inits.append(np.ones(N))
    
    # 2. Gaussian
    inits.append(_gen_gaussian(N))
    
    # 3. Chirp
    inits.append(_gen_chirp(N))
    
    # 4. Barker-like
    inits.append(_gen_barker_like(N))
    
    # 5. Random smoothed
    inits.append(_gen_random_smooth(N, seed=1))
    
    # 6. Another random with different seed
    inits.append(_gen_random_smooth(N, seed=99))
    
    # 7. Alternating signs with envelope
    x = np.arange(N)
    envelope = np.exp(-0.5 * ((x - N/2) / (N/5))**2)
    inits.append(envelope * np.cos(np.pi * x / 5))
    
    # 8. Spectral flatness init
    inits.append(_spectral_init(N))
    
    for h_init in inits:
        if time.time() > deadline - 20:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Phase 1: Gradient optimization with smooth surrogate and analytical gradient
        for beta in [50, 100, 200]:
            if time.time() > deadline - 10:
                break
            
            try:
                def obj_grad(x):
                    return _objective_smooth_grad(x, beta=beta)
                
                res = minimize(obj_grad, h_norm, method='L-BFGS-B', jac=True,
                               options={'maxiter': 300, 'disp': False, 'gtol': 1e-6, 'ftol': 1e-8})
                h_opt = res.x
                
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                    
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
            except Exception:
                pass
        
        # Phase 2: Spectral PAPR reduction
        if time.time() > deadline - 8:
            continue
        try:
            h_spectral = _spectral_papr_reduce(h_norm, n_iter=30, gamma=0.1, deadline=deadline - 5)
            if abs(np.sum(h_spectral)) > 1e-6:
                val = _score(h_spectral)
                if val < best_score:
                    best_score = val
                    best_h = h_spectral.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish on best found
    if time.time() < deadline - 10:
        h_polished, val_polished = _high_precision_polish(best_h, deadline=deadline - 3)
        if val_polished < best_score:
            best_score = val_polished
            best_h = h_polished.copy()
    
    # Final local search
    if time.time() < deadline - 3:
        h_refined, val_refined = _local_search(best_h, max_passes=10, deadline=deadline - 1)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Safety: if somehow we lost track, fall back
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
