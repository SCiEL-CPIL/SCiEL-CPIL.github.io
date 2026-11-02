"""Model-and-optimize: Chebyshev spectral envelope + decoupled amp/phase + peak polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 3)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    ac[k] = sum_i x[i]*x[k-i]
    d/dx[n] ac[k] = 2*x[k-n] if 0 <= k-n < N, else 0
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    soft_val = m + np.log(sum_exps) / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # Cross-correlation: corr[n] = sum_k grad_ac[k] * x[k-n] for 0<=k-n<N
    L_corr = 1
    while L_corr < 3 * N:
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    grad_x = 2.0 * corr[:N]
    
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_envelope_optimize(x0, rng, max_iters=300, lr=0.02):
    """
    Chebyshev-like optimization of spectral envelope E(f) to minimize
    L_infinity norm of periodic autoconvolution.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    amp = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_amp = amp.copy()
    best_val = _objective_only(x0)
    best_x = x0.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Compute gradient of max|ac| w.r.t. amplitude
        val, grad = _objective_and_grad(best_x, 100.0)
        
        # Transform gradient to frequency domain
        G = np.fft.rfft(grad, n=L_fft)
        
        # Gradient of objective w.r.t. amplitude:
        # d/d|X[f]| of ac = Re(X[f] * conj(G[f])) / |X[f]|
        amp_grad = np.real(X * np.conj(G)) / (amp + 1e-10)
        amp_grad[0] = 0  # DC
        
        # Gradient descent on amplitude
        amp_trial = amp - lr * amp_grad
        amp_trial = np.maximum(0.01, amp_trial)
        
        # Reconstruct
        spectrum = amp_trial * np.exp(1j * phase)
        spectrum[0] = amp_trial[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
            phase = -phase
            phase[0] = 0
        
        val_trial = _objective_only(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            best_amp = amp_trial.copy()
            amp = amp_trial.copy()
            if it % 30 == 0 and it > 0:
                lr = max(0.002, lr * 0.95)
        else:
            lr = max(0.002, lr * 0.85)
    
    return best_x, best_val

def _phase_optimize(x0, rng, max_iters=200, lr=0.01):
    """
    Optimize phase while keeping amplitude fixed.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    amp = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        val, grad = _objective_and_grad(best_x, 100.0)
        G = np.fft.rfft(grad, n=L_fft)
        
        # Phase gradient
        X_curr = amp * np.exp(1j * phase)
        phase_grad = np.imag(X_curr * np.conj(G)) / (amp + 1e-10)
        phase_grad[0] = 0
        
        phase_trial = phase - lr * phase_grad
        
        spectrum = amp * np.exp(1j * phase_trial)
        spectrum[0] = amp[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
            phase_trial = -phase_trial
            phase_trial[0] = 0
        
        val_trial = _objective_only(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            phase = phase_trial.copy()
            if it % 30 == 0 and it > 0:
                lr = max(0.001, lr * 0.95)
        else:
            lr = max(0.001, lr * 0.85)
    
    return best_x, best_val

def _peak_targeted_polish(x0, rng, max_iters=800, step_size=0.01):
    """
    Vectorized peak-targeted polish: identify dominant lags and apply
    targeted perturbations to reduce peak absolute autoconvolution.
    """
    best_x = x0.copy()
    best_val = _objective_only(x0)
    ac = _autoconv_fft(best_x)
    abs_ac = np.abs(ac)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        if it % 20 == 0:
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
        
        # Find top k lags
        top_k = min(15, len(abs_ac))
        top_idx = np.argsort(abs_ac)[-top_k:]
        top_idx = top_idx[np.argsort(abs_ac[top_idx])[::-1]]
        
        # For each top lag, compute gradient direction to reduce |ac[lag]|
        # ac[k] = sum_i x[i]*x[k-i]
        # d|ac[k]|/dx[n] = sign(ac[k]) * (x[k-n] + x[n-k]) if valid
        
        # Vectorized: for all top lags at once
        # Build contribution matrix: for lag k, d|ac[k]|/dx[n] = sign(ac[k])*(x[k-n]+x[n-k])
        # We want to move in direction that reduces max
        
        # Simpler: random perturbation weighted by gradient
        n_perturb = rng.integers(3, 10)
        indices = rng.choice(N, size=n_perturb, replace=False)
        
        # Compute local gradient at these indices for the peak lag
        peak_lag = top_idx[0]
        sign_peak = np.sign(ac[peak_lag]) if ac[peak_lag] != 0 else 1.0
        
        # Gradient at index n for peak lag k:
        # d|ac[k]|/dx[n] = sign(ac[k]) * (x[k-n] if 0<=k-n<N else 0 + x[n-k] if 0<=n-k<N else 0)
        grad_local = np.zeros(n_perturb)
        for i, n in enumerate(indices):
            contrib = 0.0
            if 0 <= peak_lag - n < N:
                contrib += x0[peak_lag - n] if False else best_x[peak_lag - n]
            if 0 <= n - peak_lag < N:
                contrib += best_x[n - peak_lag]
            grad_local[i] = sign_peak * 2.0 * contrib
        
        # Perturb in negative gradient direction
        norm_g = np.sqrt(np.sum(grad_local**2)) + 1e-10
        direction = -grad_local / norm_g
        perturbation = np.zeros(N)
        perturbation[indices] = direction * step_size
        
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.001, step_size * 0.97)
        else:
            # Try positive direction too
            perturbation[indices] = -direction * step_size
            x_trial = best_x + perturbation
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
                step_size = max(0.001, step_size * 0.97)
            else:
                step_size = min(0.05, step_size * 1.03)
    
    return best_x, best_val

def _spectral_init(rng, mode=0):
    """Generate initial x via spectral construction."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    if mode == 0:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(n_freq)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=n_freq)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_fft)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    elif mode == 5:
        # Flat-top with random notches (Chebyshev-like)
        mag = np.ones(n_freq)
        n_notches = rng.integers(3, 8)
        notch_freqs = rng.uniform(0, 0.4, size=n_notches)
        notch_widths = rng.uniform(0.01, 0.05, size=n_notches)
        freqs = np.fft.rfftfreq(L_fft)
        for nf, nw in zip(notch_freqs, notch_widths):
            mag *= 1.0 - 0.5 * np.exp(-((freqs - nf) / nw) ** 2)
        mag = np.maximum(0.05, mag)
    else:
        mag = np.ones(n_freq)
    
    phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    t = np.linspace(0, 1, N)
    initial_points = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.5 * t),
        np.exp(-1.5 * t),
        np.exp(0.5 * t),
        np.exp(-0.5 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5)**2,
        1.0 + 0.5 * np.sin(6 * np.pi * t) * np.cos(2 * np.pi * t),
        1.0 + 0.3 * (t - 0.5)**3,
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),
        np.maximum(0.5, 1.0 - 3.0 * np.abs(t - 0.5)),
    ]
    
    # Add spectral initializations
    for mode in range(6):
        for _ in range(2):
            if _check_deadline():
                break
            initial_points.append(_spectral_init(rng, mode))
    
    # Add random initializations
    for _ in range(4):
        if _check_deadline():
            break
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))
    
    candidates = []
    
    for x0 in initial_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        # Phase 1: Spectral envelope optimization
        x_envelope, val_envelope = _spectral_envelope_optimize(x_start, rng, max_iters=150, lr=0.02)
        if val_envelope < best_val:
            best_val = val_envelope
            best_x = x_envelope.copy()
        candidates.append((val_envelope, x_envelope))
        
        if _check_deadline():
            break
        
        # Phase 2: Phase optimization
        x_phase, val_phase = _phase_optimize(x_envelope, rng, max_iters=150, lr=0.01)
        if val_phase < best_val:
            best_val = val_phase
            best_x = x_phase.copy()
        candidates.append((val_phase, x_phase))
        
        if _check_deadline():
            break
        
        # Phase 3: L-BFGS refinement
        for beta in [100.0]:
            if _check_deadline():
                break
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_phase, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 80, 'gtol': 1e-8, 'ftol': 1e-10})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                candidates.append((result.fun, result.x))
                
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Sort candidates and polish the top few
    candidates.sort(key=lambda item: item[0])
    top_candidates = [x for _, x in candidates[:6]]
    
    # Peak-targeted polish on top candidates
    for x_init in top_candidates:
        if _check_deadline():
            break
        
        x_polished, val_polished = _peak_targeted_polish(x_init, rng, max_iters=600, step_size=0.01)
        if val_polished < best_val:
            best_val = val_polished
            best_x = x_polished.copy()
        
        if _check_deadline():
            break
        
        # Final L-BFGS polish with high beta
        for beta in [200.0, 500.0]:
            if _check_deadline():
                break
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_polished, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-12})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    # Final coordinate descent
    if not _check_deadline():
        step_size = 0.01
        no_improve = 0
        max_no_improve = 1500
        
        while no_improve < max_no_improve and not _check_deadline():
            n_perturb = rng.integers(1, 5)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
            x_trial = best_x + perturbation
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
                step_size = max(0.001, step_size * 0.95)
                no_improve = 0
            else:
                no_improve += 1
                step_size = min(0.05, step_size * 1.05)
                
            if no_improve > 400:
                perturbation = np.zeros(N)
                indices = rng.choice(N, size=10, replace=False)
                perturbation[indices] = rng.normal(0, 0.05, size=10)
                x_trial = best_x + perturbation
                val = _objective_only(x_trial)
                if val < best_val:
                    best_val = val
                    best_x = x_trial.copy()
                    no_improve = 0
                    step_size = 0.01
                else:
                    no_improve += 1

    return best_x
# EVOLVE-BLOCK-END
