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
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

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

def _spectral_papr_reduce(h, n_iter=50, gamma=0.5, deadline=None):
    """
    Gradient-based reduction of max|autoconv| using the soft-max surrogate.
    Computes the true gradient of sum_k w_k * |ac_k| w.r.t. h via FFT,
    takes a step, then re-normalizes sum(h) to preserve the denominator.
    """
    h = h.copy().astype(float)
    s_target = np.sum(h)
    if abs(s_target) < 1e-9:
        return h
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        ac = _autoconv_fft(h)
        ac_abs = np.abs(ac)
        total = np.sum(ac_abs)
        if total < 1e-15:
            break
        weights = ac_abs / total

        # Gradient of sum_k w_k * |ac_k| w.r.t. h:
        # ac = h * h (linear). d(ac_k)/d(h_i) = h_{k-i} + h_{i-k} (bounded).
        # In FFT domain: grad = 2 * IFFT(FFT(w) * conj(H)) truncated to N.
        H = np.fft.rfft(h, _FFT_SIZE)
        w_fft = np.fft.rfft(weights, _FFT_SIZE)
        grad = 2.0 * np.fft.irfft(w_fft * np.conj(H), _FFT_SIZE)[:N]

        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break

        # Adaptive step: scale so the change is a small fraction of h's norm
        hnorm = np.linalg.norm(h)
        step = gamma * hnorm / gnorm
        h_new = h - step * grad

        # Re-normalize to preserve sum
        s_new = np.sum(h_new)
        if abs(s_new) < 1e-9:
            break
        h_new = h_new * (s_target / s_new)

        # Accept only if the true (non-soft) score improves
        val_new = _score(h_new)
        val_cur = _score(h)
        if val_new < val_cur:
            h = h_new
        else:
            # Halve the step and retry once
            h_new = h - 0.5 * step * grad
            s_new = np.sum(h_new)
            if abs(s_new) > 1e-9:
                h_new = h_new * (s_target / s_new)
                val_new = _score(h_new)
                if val_new < val_cur:
                    h = h_new

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
    Local search with batched evaluations for efficiency.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(42)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try flipping signs at random positions (batched)
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
            for factor in [1.05, 0.95, 1.1, 0.9]:
                if deadline is not None and time.time() > deadline:
                    break
                h[i] = orig * factor
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                    break
            if not improved:
                h[i] = orig
        
        if not improved:
            break
    
    return h, best_val

def _sign_flip_search(h, deadline=None, max_evals=20000):
    """
    Dedicated sign-flip local search. Since the problem is about *signed* step
    functions, flipping signs at individual positions is a natural move that
    can dramatically change the autoconvolution structure. We use a hill-climbing
    strategy with multiple random restarts of the flip order.
    """
    h = h.copy().astype(float)
    best_val = _score(h)
    rng = np.random.default_rng(2024)
    evals = 0
    s_target = np.sum(h)
    if abs(s_target) < 1e-9:
        return h, best_val

    # Normalize to unit sum for consistent scaling
    h = h / s_target

    for restart in range(5):
        if deadline is not None and time.time() > deadline:
            break
        improved = True
        while improved and evals < max_evals:
            improved = False
            # Shuffle positions for diversity
            order = rng.permutation(N)
            for i in order:
                if deadline is not None and time.time() > deadline:
                    return h, best_val
                if evals >= max_evals:
                    break
                orig = h[i]
                h[i] = -orig
                evals += 1
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[i] = orig

    # Restore original scale
    h = h * s_target
    final_val = _score(h)
    if final_val < best_val:
        best_val = final_val
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
    
    # Use very tight tolerances
    try:
        def obj_tight(x):
            return _objective_smooth(x, beta=500.0)
        
        res = minimize(obj_tight, h, method='L-BFGS-B',
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
            for factor in [1.01, 0.99, 1.02, 0.98]:
                h[i] = orig * factor
                new_val = _score(h)
                if new_val < current_score:
                    current_score = new_val
                    improved = True
                    break
            if not improved:
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
        if time.time() > deadline - 30:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Phase 1: Gradient optimization with smooth surrogate (fewer betas, fewer iters)
        for beta in [100, 200]:
            if time.time() > deadline - 15:
                break
            
            try:
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': 150, 'disp': False, 'gtol': 1e-6, 'ftol': 1e-8})
                h_opt = res.x
                
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                    
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
            except Exception:
                pass
        
        # Phase 2: Spectral PAPR reduction (fixed gradient-based)
        if time.time() > deadline - 12:
            continue
        try:
            h_spectral = _spectral_papr_reduce(h_norm, n_iter=40, gamma=0.3, deadline=deadline - 8)
            if abs(np.sum(h_spectral)) > 1e-6:
                val = _score(h_spectral)
                if val < best_score:
                    best_score = val
                    best_h = h_spectral.copy()
        except Exception:
            pass
    
    # Phase 3: Sign-flip local search (exploits the signed structure)
    if time.time() < deadline - 20:
        try:
            h_sign, val_sign = _sign_flip_search(best_h, deadline=deadline - 15)
            if val_sign < best_score:
                best_score = val_sign
                best_h = h_sign.copy()
        except Exception:
            pass

    # Phase 4: High-precision polish on best found
    if time.time() < deadline - 10:
        h_polished, val_polished = _high_precision_polish(best_h, deadline=deadline - 5)
        if val_polished < best_score:
            best_score = val_polished
            best_h = h_polished.copy()
    
    # Final local search
    if time.time() < deadline - 3:
        h_refined, val_refined = _local_search(best_h, max_passes=8, deadline=deadline - 1)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Safety: if somehow we lost track, fall back
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
