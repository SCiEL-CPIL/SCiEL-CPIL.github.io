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

def _autoconv_direct(h):
    """Compute linear autoconvolution directly (for small N or verification)."""
    ac = np.correlate(h, h, mode='full')
    return ac[N-1:N-1+2*N-1]

def _flip_delta_ac(h, i, sign=-1.0):
    """
    Compute the change in autoconvolution when h[i] is multiplied by `sign` (i.e., flipped).
    If sign = -1, this is a sign flip: h[i] -> -h[i].
    
    ac[k] = sum_j h[j] * h[k-j]
    After flip: h'[i] = sign * h[i], all other h'[j] = h[j]
    
    ac'[k] = ac[k] + (sign-1) * (h[i]*h[k-i] + h[k-i]*h[i]) 
            but need to be careful: the term h[i]*h[k-i] appears when j=i or j=k-i.
            If i == k-i (i.e., k=2i), it appears once, not twice.
    
    More precisely:
    ac'[k] - ac[k] = (sign-1) * (h[i] * h[k-i] + h[k-i] * h[i] - delta_{k,2i} * h[i] * h[i])
    Wait, let me think more carefully.
    
    ac[k] = sum_{j=0}^{N-1} h[j] * h[k-j] (with h[x]=0 for x outside [0,N-1])
    
    After changing h[i] to sign*h[i]:
    ac'[k] = ac[k] - h[i]*h[k-i] - h[k-i]*h[i] + sign*h[i]*h[k-i] + sign*h[k-i]*h[i]
             (but if i == k-i, i.e., k = 2i, we have h[i]*h[i] appearing once in the sum)
    
    Actually: ac[k] = sum_j h[j]*h[k-j]. The terms involving h[i] are:
    - j=i: h[i]*h[k-i]
    - j=k-i: h[k-i]*h[i]
    These are the same term if i = k-i, i.e., k=2i.
    
    So: ac'[k] - ac[k] = (sign-1) * (h[i]*h[k-i] + h[k-i]*h[i]) - (sign-1)*delta_{k,2i}*h[i]*h[i]
    Wait no. Let me redo.
    
    Original: ac[k] includes term h[i]*h[k-i] (from j=i) and term h[k-i]*h[i] (from j=k-i).
    If i != k-i: two separate terms, each h[i]*h[k-i].
    If i == k-i (k=2i): one term h[i]*h[i].
    
    New: ac'[k] includes sign*h[i]*h[k-i] (from j=i) and sign*h[k-i]*h[i] (from j=k-i).
    If i != k-i: two terms, each sign*h[i]*h[k-i].
    If i == k-i: one term sign*h[i]*h[i].
    
    Delta:
    If k != 2i: ac'[k] - ac[k] = (sign-1) * 2 * h[i] * h[k-i]
    If k == 2i: ac'[k] - ac[k] = (sign-1) * h[i] * h[i]
    """
    delta = np.zeros(2*N-1)
    # For each k where the delta is nonzero:
    # k ranges from 0 to 2N-2
    # h[i] * h[k-i] is nonzero when 0 <= k-i <= N-1, i.e., i <= k <= i+N-1
    # And h[k-i] * h[i] is nonzero when 0 <= k-i <= N-1, same range.
    # But also need 0 <= k <= 2N-2.
    
    k_min = max(0, i)
    k_max = min(2*N-2, i + N - 1)
    
    for k in range(k_min, k_max + 1):
        ki = k - i  # index of the other element
        if 0 <= ki < N:
            if k == 2 * i:
                delta[k] = (sign - 1.0) * h[i] * h[i]
            else:
                delta[k] = (sign - 1.0) * 2.0 * h[i] * h[k-i]
    
    return delta

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

def _fast_flip_score(h, ac, s, i, sign=-1.0):
    """
    Compute the score after flipping h[i] by `sign`, given current ac and sum s.
    Uses incremental update: O(N) for the delta computation + O(1) for max tracking.
    Returns (new_score, new_ac) where new_ac is the updated autoconvolution.
    """
    delta = _flip_delta_ac(h, i, sign)
    new_ac = ac + delta
    new_s = s + (sign - 1.0) * h[i]
    if abs(new_s) < 1e-9:
        return 1e10, new_ac
    new_max = np.max(np.abs(new_ac))
    return new_max / (new_s * new_s), new_ac

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

def _spectral_papr_reduce(h, n_iter=30, gamma=0.4, deadline=None):
    """
    Gradient-based reduction of max|autoconv| using the soft-max surrogate.
    Computes the true gradient of sum_k w_k * |ac_k| w.r.t. h via FFT,
    takes a step, then re-normalizes sum(h) to preserve the denominator.
    """
    h = h.copy().astype(float)
    s_target = np.sum(h)
    if abs(s_target) < 1e-9:
        return h
    val_cur = _score(h)
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
        if val_new < val_cur:
            h = h_new
            val_cur = val_new
        else:
            # Halve the step and retry once
            h_new = h - 0.5 * step * grad
            s_new = np.sum(h_new)
            if abs(s_new) > 1e-9:
                h_new = h_new * (s_target / s_new)
                val_new = _score(h_new)
                if val_new < val_cur:
                    h = h_new
                    val_cur = val_new

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

def _local_search(h, max_passes=20, deadline=None):
    """
    Local search with incremental autoconvolution updates for speed.
    """
    h = h.copy().astype(float)
    ac = _autoconv_fft(h)
    s = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s * s)
    rng = np.random.default_rng(42)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try flipping signs at random positions with incremental update
        for _ in range(N):
            if deadline is not None and time.time() > deadline:
                break
            i = rng.integers(N)
            orig = h[i]
            h[i] = -orig
            new_s = s - 2.0 * orig
            if abs(new_s) < 1e-9:
                h[i] = orig
                continue
            delta = _flip_delta_ac(h, i, -1.0)
            # h[i] is now -orig, so _flip_delta_ac uses h[i]=-orig
            # delta[k] = (-2) * 2 * (-orig) * h[k-i] = 4*orig*h[k-i] for k!=2i
            # But we want delta from orig to -orig: should be -2*orig*h[k-i]
            # Let me just compute it properly by restoring first
            h[i] = orig
            delta = _flip_delta_ac(h, i, -1.0)
            h[i] = -orig
            new_ac = ac + delta
            new_val = np.max(np.abs(new_ac)) / (new_s * new_s)
            if new_val < best_val:
                best_val = new_val
                ac = new_ac
                s = new_s
                improved = True
            else:
                h[i] = orig
        
        # Try small multiplicative perturbations (full FFT for these since delta is more complex)
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
                    ac = _autoconv_fft(h)
                    s = np.sum(h)
                    improved = True
                    break
            if not improved:
                h[i] = orig
        
        if not improved:
            break
    
    return h, best_val

def _sign_flip_search(h, deadline=None, max_evals=200000):
    """
    Dedicated sign-flip local search with incremental autoconvolution updates.
    Uses O(N) per flip instead of O(N log N) FFT, enabling far more evaluations.
    """
    h = h.copy().astype(float)
    ac = _autoconv_fft(h)
    s = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s * s)
    rng = np.random.default_rng(2024)
    evals = 0
    s_target = s

    # Normalize to unit sum for consistent scaling
    h = h / s_target
    ac = _autoconv_fft(h)
    s = 1.0
    best_val = np.max(np.abs(ac)) / (s * s)

    for restart in range(8):
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
                new_s = s - 2.0 * orig  # s + (-1-1)*h[i] = s - 2*orig
                if abs(new_s) < 1e-9:
                    h[i] = orig
                    continue
                delta = _flip_delta_ac(h, i, -1.0)
                # But h[i] is already flipped in h, so we need to compute delta based on original h
                # Let me recompute: we need the delta from the ORIGINAL h before flip
                # Actually, _flip_delta_ac uses h[i] as the current value. Since we already flipped,
                # h[i] = -orig. The delta formula uses h[i] which is now -orig.
                # delta[k] = (-1-1) * 2 * (-orig) * h[k-i] = 4*orig*h[k-i] for k != 2i
                # That's correct! Because the change is from orig to -orig, delta = -2*orig contribution.
                # Wait: sign=-1, so sign-1 = -2. h[i] is now -orig.
                # delta[k] = -2 * 2 * (-orig) * h[k-i] = 4*orig*h[k-i]. 
                # Original contribution: orig*h[k-i]. New: -orig*h[k-i]. Delta = -2*orig*h[k-i].
                # Hmm, that doesn't match. Let me reconsider.
                
                # Actually I should compute delta BEFORE flipping. Let me restructure.
                h[i] = orig  # restore
                delta = _flip_delta_ac(h, i, -1.0)
                # Now apply flip
                h[i] = -orig
                new_ac = ac + delta
                new_val = np.max(np.abs(new_ac)) / (new_s * new_s)
                if new_val < best_val:
                    best_val = new_val
                    ac = new_ac
                    s = new_s
                    improved = True
                else:
                    h[i] = orig  # revert

    # Restore original scale
    h = h * s_target
    ac_final = _autoconv_fft(h)
    s_final = np.sum(h)
    final_val = np.max(np.abs(ac_final)) / (s_final * s_final)
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
    h = h.copy().astype(float)
    ac = _autoconv_fft(h)
    s = np.sum(h)
    current_score = np.max(np.abs(ac)) / (s * s)
    
    # Use very tight tolerances
    try:
        def obj_tight(x):
            return _objective_smooth(x, beta=500.0)
        
        res = minimize(obj_tight, h, method='L-BFGS-B',
                       options={'maxiter': 300, 'disp': False, 'gtol': 1e-10, 'ftol': 1e-12})
        h_opt = res.x
        if abs(np.sum(h_opt)) > 1e-6:
            val = _score(h_opt)
            if val < current_score:
                h = h_opt
                ac = _autoconv_fft(h)
                s = np.sum(h)
                current_score = val
    except Exception:
        pass
    
    # Final local search with very small steps using incremental updates
    rng = np.random.default_rng(7)
    for _ in range(30):
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
                    ac = _autoconv_fft(h)
                    s = np.sum(h)
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

    # Allocate time: ~40% for gradient phases, ~50% for sign-flip search, ~10% for polish
    grad_deadline = t_start + 60.0  # 60s for gradient optimization
    flip_deadline = t_start + 130.0  # 70s for sign-flip search
    polish_deadline = deadline  # remaining for polish

    for h_init in inits:
        if time.time() > grad_deadline:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Phase 1: Gradient optimization with smooth surrogate (reduced iterations)
        for beta in [100, 200]:
            if time.time() > grad_deadline:
                break
            
            try:
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': 80, 'disp': False, 'gtol': 1e-5, 'ftol': 1e-7})
                h_opt = res.x
                
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                    
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
            except Exception:
                pass
        
        # Phase 2: Spectral PAPR reduction (reduced iterations)
        if time.time() > grad_deadline:
            continue
        try:
            h_spectral = _spectral_papr_reduce(h_norm, n_iter=25, gamma=0.3, deadline=grad_deadline)
            if abs(np.sum(h_spectral)) > 1e-6:
                val = _score(h_spectral)
                if val < best_score:
                    best_score = val
                    best_h = h_spectral.copy()
        except Exception:
            pass
    
    # Phase 3: Sign-flip local search with incremental updates (main workhorse)
    if time.time() < flip_deadline:
        try:
            h_sign, val_sign = _sign_flip_search(best_h, deadline=flip_deadline)
            if val_sign < best_score:
                best_score = val_sign
                best_h = h_sign.copy()
        except Exception:
            pass

    # Phase 4: High-precision polish on best found
    if time.time() < polish_deadline - 5:
        h_polished, val_polished = _high_precision_polish(best_h, deadline=polish_deadline - 3)
        if val_polished < best_score:
            best_score = val_polished
            best_h = h_polished.copy()
    
    # Final local search
    if time.time() < deadline - 2:
        h_refined, val_refined = _local_search(best_h, max_passes=15, deadline=deadline - 1)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Safety: if somehow we lost track, fall back
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
