"""Chebyshev spectral envelope design + phase retrieval + local search."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _fft_len(n):
    L = 1
    while L < n:
        L *= 2
    return L

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = _fft_len(L)
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Raw objective: max|autoconv| / (sum x)^2"""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _remez_envelope(amps_init, n_iters=300, rng=None):
    """
    Remez-like exchange on the frequency grid to design a spectral envelope
    that minimizes the L-infinity norm of the periodic autoconvolution.
    
    The autoconvolution in frequency domain is |X(f)|^2. We want to shape
    |X(f)| so that the inverse DFT of |X(f)|^2 has minimal peak magnitude.
    
    Algorithm: iteratively identify the peak lag in time domain, compute the
    corresponding frequency response, and adjust the envelope at frequencies
    that contribute most to that peak (Remez exchange principle).
    """
    if rng is None:
        rng = np.random.default_rng(7)
    
    L_fft = _fft_len(2 * N - 1)
    amps = amps_init.copy().astype(float)
    n_freq = len(amps)
    
    best_val = 1e10
    best_amps = amps.copy()
    
    # Initialize phases randomly
    phases = rng.uniform(0, 2 * np.pi, size=n_freq)
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        # Reconstruct time domain
        X = amps * np.exp(1j * phases)
        x = np.fft.irfft(X, n=L_fft)[:N]
        s = np.sum(x)
        if abs(s) < 1e-10:
            phases = rng.uniform(0, 2 * np.pi, size=n_freq)
            continue
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        val = np.max(abs_ac) / (s * s)
        
        if val < best_val:
            best_val = val
            best_amps = amps.copy()
        
        # Remez exchange: find the peak lag and adjust envelope
        peak_lag = np.argmax(abs_ac)
        
        # The contribution to autoconv at lag k is:
        # ac[k] = sum_i x[i]*x[i+k]
        # In frequency domain, ac is the IFFT of |X|^2
        # So the peak is driven by the distribution of |X|^2
        # To reduce the peak at lag k, we want to reduce |X|^2 at frequencies
        # where the phase alignment causes constructive interference at that lag
        
        # Compute the frequency-domain autoconvolution (|X|^2)
        X_mag_sq = amps ** 2
        
        # The time-domain autoconvolution at lag k is:
        # ac[k] = (1/L) * sum_f |X(f)|^2 * exp(i*2*pi*f*k/L)
        # To reduce ac[k], reduce |X|^2 where exp(i*2*pi*f*k/L) aligns with sign(ac[k])
        
        # Compute the "error signal" in frequency domain
        # The target is flat autoconvolution (all zeros except DC)
        # Error = ac - 0 = ac
        # Frequency response of error: E(f) = FFT(ac)
        L_ac = _fft_len(2 * N - 1)
        E = np.fft.rfft(ac, n=L_ac)
        
        # The gradient of the peak w.r.t. |X|^2 at frequency f is:
        # d(ac[k])/d(|X(f)|^2) = (1/L) * exp(i*2*pi*f*k/L)
        # We want to reduce |X|^2 where this is positive (constructive)
        
        freqs = np.arange(n_freq)
        # Phase of contribution to peak lag
        contrib_phase = np.exp(1j * 2 * np.pi * freqs * peak_lag / L_ac)
        
        # Reduce amps where they contribute constructively to the peak
        # Remez-like: adjust by a small amount proportional to contribution
        step = 0.02 / (1 + it * 0.005)
        
        # Compute how much each frequency contributes to the peak
        # contribution_f = |X(f)|^2 * Re(contrib_phase_f)
        contrib = (amps ** 2) * np.real(contrib_phase)
        
        # Normalize
        contrib_max = np.max(np.abs(contrib)) + 1e-10
        contrib_norm = contrib / contrib_max
        
        # Update: reduce where positive contribution, increase where negative
        # (Chebyshev equiripple principle: push down peaks, push up valleys)
        amps = amps - step * contrib_norm * amps
        amps = np.maximum(amps, 0.01)  # keep positive
        
        # Also do a phase update to reduce the peak
        # Fienup-style phase correction
        X_cur = amps * np.exp(1j * phases)
        with np.errstate(divide='ignore', invalid='ignore'):
            phase_shift = -np.imag(X_cur * np.conj(E[:n_freq])) / (amps ** 2 + 1e-10)
        
        phase_step = 0.3 / (1 + it * 0.01)
        phases = (phases + phase_step * phase_shift) % (2 * np.pi)
        
        # Periodic randomization to escape local minima
        if it % 30 == 29 and it > 0:
            n_perturb = max(1, n_freq // 15)
            idx = rng.choice(n_freq, size=n_perturb, replace=False)
            amps[idx] *= rng.uniform(0.7, 1.3, size=n_perturb)
            phases[idx] += rng.uniform(-0.5, 0.5, size=n_perturb)
    
    return best_amps, best_val

def _phase_retrieval_with_envelope(amps, n_iters=100, rng=None):
    """
    Given a fixed spectral envelope (amplitudes), find phases that
    minimize the peak autoconvolution. Fienup-style iterative algorithm.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    L_fft = _fft_len(2 * N - 1)
    n_freq = len(amps)
    
    best_val = 1e10
    best_x = np.zeros(N)
    best_phases = rng.uniform(0, 2 * np.pi, size=n_freq)
    
    # Multiple random phase initializations
    n_restarts = 3
    for restart in range(n_restarts):
        if _check_deadline():
            break
        
        phases = rng.uniform(0, 2 * np.pi, size=n_freq)
        
        for it in range(n_iters):
            if _check_deadline():
                break
            
            X = amps * np.exp(1j * phases)
            x = np.fft.irfft(X, n=L_fft)[:N]
            s = np.sum(x)
            if abs(s) < 1e-10:
                phases = rng.uniform(0, 2 * np.pi, size=n_freq)
                continue
            
            ac = _autoconv_fft(x)
            val = np.max(np.abs(ac)) / (s * s)
            
            if val < best_val:
                best_val = val
                best_x = x.copy()
                best_phases = phases.copy()
            
            # Fienup phase update
            peak_lag = np.argmax(np.abs(ac))
            L_ac = _fft_len(2 * N - 1)
            E = np.fft.rfft(ac, n=L_ac)
            
            X_cur = amps * np.exp(1j * phases)
            with np.errstate(divide='ignore', invalid='ignore'):
                phase_shift = -np.imag(X_cur * np.conj(E[:n_freq])) / (amps ** 2 + 1e-10)
            
            step = 0.4 / (1 + it * 0.01)
            phases = (phases + step * phase_shift) % (2 * np.pi)
    
    return best_x, best_val, best_phases

def _local_search(x, n_rounds=60, rng=None):
    """
    Local search: identify peak lags and apply targeted perturbations
    to reduce them. Uses vectorized evaluation.
    """
    if rng is None:
        rng = np.random.default_rng(99)
    
    x = x.copy()
    best_val = _objective(x)
    
    for round_idx in range(n_rounds):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        
        current_val = np.max(abs_ac) / (s * s)
        
        # Find top peak lags
        k = min(3, len(abs_ac))
        top_lags = np.argsort(abs_ac)[-k:][::-1]
        
        improved = False
        for lag in top_lags:
            if _check_deadline():
                break
            
            # Try multiple perturbation strategies for this lag
            for n_trials in range(15):
                if _check_deadline():
                    break
                
                # Strategy: perturb indices that contribute to this lag
                # ac[lag] = sum_i x[i]*x[i+lag]
                # To reduce it, flip signs of x[i] where x[i]*x[i+lag] > 0
                # (constructive contribution)
                
                valid_i = np.arange(N - lag) if lag < N else np.arange(0)
                if len(valid_i) == 0:
                    continue
                
                # Find indices with positive product
                products = x[valid_i] * x[valid_i + lag]
                pos_idx = valid_i[products > 0]
                
                if len(pos_idx) == 0:
                    # Try negative products
                    neg_idx = valid_i[products < 0]
                    if len(neg_idx) == 0:
                        continue
                    candidates = neg_idx
                else:
                    candidates = pos_idx
                
                # Pick a subset to flip
                n_flip = max(1, min(5, len(candidates)))
                flip_idx = rng.choice(candidates, size=n_flip, replace=False)
                
                x_trial = x.copy()
                x_trial[flip_idx] *= -1  # flip sign
                
                val = _objective(x_trial)
                if val < best_val * (1 - 1e-8):
                    best_val = val
                    x = x_trial
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try random sign flips
            for _ in range(10):
                if _check_deadline():
                    break
                n_flip = rng.integers(2, 8)
                flip_idx = rng.choice(N, size=n_flip, replace=False)
                x_trial = x.copy()
                x_trial[flip_idx] *= -1
                val = _objective(x_trial)
                if val < best_val * (1 - 1e-8):
                    best_val = val
                    x = x_trial
                    improved = True
                    break
        
        if not improved:
            break
    
    return x, best_val

def _signed_quantize(x):
    """Quantize to ±1 while preserving the sign pattern and relative magnitudes."""
    # Simple approach: sign function with small noise for ties
    signs = np.sign(x)
    signs[signs == 0] = 1
    return signs

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(12345)
    L_fft = _fft_len(2 * N - 1)
    n_freq = L_fft // 2 + 1
    
    # Phase 1: Design spectral envelope via Remez-like algorithm
    # Start with several different envelope initializations
    freqs = np.arange(n_freq)
    freq_norm = freqs / n_freq
    
    envelope_inits = [
        np.ones(n_freq),  # flat
        1.0 + 0.3 * np.cos(2 * np.pi * freq_norm * 3),  # low-freq boost
        1.0 + 0.3 * np.cos(2 * np.pi * freq_norm * 7),  # mid-freq ripple
        np.exp(-0.5 * freq_norm ** 2),  # Gaussian low-pass
        1.0 - 0.3 * freq_norm,  # linear decay
        np.ones(n_freq) * (1 + 0.2 * np.sin(2 * np.pi * freq_norm * 11)),  # high-freq ripple
    ]
    
    # Add random envelope initializations
    for _ in range(6):
        env = 1.0 + 0.3 * rng.normal(0, 1, size=n_freq)
        env = np.maximum(env, 0.1)
        envelope_inits.append(env)
    
    # For each envelope init, run Remez envelope design then phase retrieval
    for env_init in envelope_inits:
        if _check_deadline():
            break
        
        # Step 1: Optimize the envelope shape
        amps_opt, env_val = _remez_envelope(env_init, n_iters=200, rng=rng)
        
        if _check_deadline():
            break
        
        # Step 2: Phase retrieval with the optimized envelope
        x_rec, pr_val, _ = _phase_retrieval_with_envelope(amps_opt, n_iters=80, rng=rng)
        
        if pr_val < best_val:
            best_val = pr_val
            best_x = x_rec.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Local search on the best result
    best_x, best_val = _local_search(best_x, n_rounds=50, rng=rng)
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Additional local search rounds with different strategies
    best_x, best_val = _local_search(best_x, n_rounds=40, rng=np.random.default_rng(777))
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Try to improve by sign-flipping individual elements
    x = best_x.copy()
    val = best_val
    for _ in range(30):
        if _check_deadline():
            break
        
        improved = False
        # Try flipping each element
        for j in range(0, N, 2):  # step by 2 for speed
            if _check_deadline():
                break
            x_trial = x.copy()
            x_trial[j] *= -1
            v = _objective(x_trial)
            if v < val * (1 - 1e-8):
                val = v
                x = x_trial
                improved = True
        if not improved:
            break
    
    if val < best_val:
        best_val = val
        best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
