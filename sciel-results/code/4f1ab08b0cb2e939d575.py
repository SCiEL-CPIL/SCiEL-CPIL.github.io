"""Phase-only optimization with fixed amplitude spectrum for C3' minimization."""
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

def _build_fixed_amplitude_spectrum(mode=0, rng=None):
    """
    Build a fixed amplitude spectrum for phase-only optimization.
    Different modes produce different amplitude profiles.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    freqs = np.fft.rfftfreq(L_fft)
    abs_freqs = np.abs(freqs)
    
    if mode == 0:
        # Prolate-like: concentrated in low frequencies
        # Using a raised cosine window that's concentrated
        width = 0.3
        mag = np.where(abs_freqs < width, 1.0, 0.0)
        # Smooth the transition
        trans = 0.05
        mag = np.where(abs_freqs < width - trans, 1.0,
                       np.where(abs_freqs < width + trans,
                                0.5 * (1 + np.cos(np.pi * (abs_freqs - width) / trans)),
                                0.0))
    elif mode == 1:
        # Gaussian in frequency
        sigma = 0.15
        mag = np.exp(-0.5 * (abs_freqs / sigma) ** 2)
    elif mode == 2:
        # Flat with gentle taper at edges
        mag = np.ones(n_freq)
        edge = 0.4
        mag *= np.where(abs_freqs > edge,
                        np.maximum(0.0, 1 - (abs_freqs - edge) / (0.5 - edge)),
                        1.0)
    elif mode == 3:
        # Banded: energy in a specific band
        lo, hi = 0.1, 0.35
        mag = np.where((abs_freqs >= lo) & (abs_freqs <= hi), 1.0, 0.05)
        trans = 0.03
        mag = np.where((abs_freqs >= lo - trans) & (abs_freqs < lo),
                       (abs_freqs - (lo - trans)) / trans, mag)
        mag = np.where((abs_freqs > hi) & (abs_freqs <= hi + trans),
                       1 - (abs_freqs - hi) / trans, mag)
    elif mode == 4:
        # Very narrow low-pass
        cutoff = 0.15
        mag = np.where(abs_freqs < cutoff, 1.0, 0.0)
        trans = 0.02
        mag = np.where((abs_freqs < cutoff - trans), 1.0,
                       np.where(abs_freqs < cutoff + trans,
                                0.5 * (1 + np.cos(np.pi * (abs_freqs - cutoff) / trans)),
                                0.0))
    elif mode == 5:
        # Cosine bell in frequency
        mag = 0.5 + 0.5 * np.cos(np.pi * abs_freqs / 0.4)
        mag = np.maximum(0.0, mag)
    elif mode == 6:
        # Inverse: more energy at higher frequencies
        mag = np.exp(-0.5 * ((0.4 - abs_freqs) / 0.15) ** 2)
    elif mode == 7:
        # Two bumps
        bump1 = np.exp(-0.5 * ((abs_freqs - 0.1) / 0.08) ** 2)
        bump2 = np.exp(-0.5 * ((abs_freqs - 0.25) / 0.08) ** 2)
        mag = bump1 + bump2
    else:
        mag = np.ones(n_freq)
    
    # Ensure DC is nonzero
    mag[0] = max(mag[0], 0.1)
    
    return mag, L_fft, n_freq

def _phase_objective_and_grad(phase, fixed_mag, L_fft, n_freq):
    """
    Objective and gradient for phase-only optimization.
    phase: array of length n_freq (radians)
    fixed_mag: fixed amplitude spectrum
    """
    # Build complex spectrum
    spectrum = fixed_mag * np.exp(1j * phase)
    spectrum[0] = fixed_mag[0]  # Keep DC real
    
    # Inverse FFT to get time domain
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(n_freq)
    
    # Compute autoconvolution
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    # Soft-max surrogate with adaptive beta
    beta = 100.0
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    if sum_exps < 1e-300:
        return m / (s * s), np.zeros(n_freq)
    log_sum = np.log(sum_exps)
    soft_val = m + log_sum / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # Gradient w.r.t. phase:
    # d(x)/d(phase[k]) = i * fixed_mag[k] * exp(i*phase[k]) * (d x_full / d spectrum[k])
    # For irfft, x_full[j] = (1/L_fft) * sum_k spectrum[k] * exp(2*pi*i*k*j/L_fft) + c.c.
    # d(x_full[j])/d(spectrum[k]) = exp(2*pi*i*k*j/L_fft) for k < L_fft/2
    #                               exp(-2*pi*i*(L_fft-k)*j/L_fft) for k > L_fft/2 (conjugate)
    # For rfft, we only have k = 0..L_fft/2, and irfft handles the symmetry.
    
    # d(x[j])/d(phase[k]) = i * fixed_mag[k] * exp(i*phase[k]) * d(x_full[j])/d(spectrum[k])
    #                      + (-i) * fixed_mag[k] * exp(-i*phase[k]) * d(x_full[j])/d(conj(spectrum[k]))
    
    # For k=0 and k=L_fft/2 (if L_fft even), the spectrum is real, so phase doesn't matter.
    # For 0 < k < L_fft/2:
    #   d(x_full[j])/d(spectrum[k]) = exp(2*pi*i*k*j/L_fft)
    #   d(x_full[j])/d(conj(spectrum[k])) = exp(-2*pi*i*k*j/L_fft)
    
    # Let's compute the gradient numerically-friendly way:
    # grad_phase[k] = Re[ d(soft_val)/d(spectrum[k]) * conj(d(spectrum[k])/d(phase[k])) ]
    # d(spectrum[k])/d(phase[k]) = i * fixed_mag[k] * exp(i*phase[k]) = i * spectrum[k]
    # So grad_phase[k] = Re[ d(soft_val)/d(spectrum[k]) * (-i) * conj(spectrum[k]) ]
    
    # d(soft_val)/d(x[j]) = sum_m grad_ac[m] * d(ac[m])/d(x[j])
    # d(ac[m])/d(x[j]) = x[j+m-(N-1)] + x[j-m+(N-1)] (with boundary handling)
    
    # Compute grad_x first
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
    
    # Center grad_ac
    h = np.zeros(2 * N - 1)
    h[N-1 : 2*N-1] = grad_ac[0 : N]
    h[0 : N-1] = grad_ac[2*N-2 : N-1 : -1]
    
    X_f = np.fft.rfft(x, n=L_corr)
    H_f = np.fft.rfft(h, n=L_corr)
    
    corr1 = np.fft.irfft(np.conj(H_f) * X_f, n=L_corr)[:N]
    h_rev = h[::-1]
    H_rev_f = np.fft.rfft(h_rev, n=L_corr)
    corr2 = np.fft.irfft(np.conj(H_rev_f) * X_f, n=L_corr)[:N]
    
    grad_x = corr1 + corr2
    
    # Now d(soft_val)/d(x[j]) = grad_x[j] / (s*s) - 2*soft_val * grad_x[j] / (s*s*s)
    # Wait, that's not right. Let me redo.
    # val = soft_val / s^2
    # d(val)/d(x[j]) = (d(soft_val)/d(x[j]) * s^2 - soft_val * 2*s) / s^4
    #                = d(soft_val)/d(x[j]) / s^2 - 2*soft_val / s^3
    
    # d(soft_val)/d(x[j]) = grad_x[j] (from the soft-max gradient)
    # Actually, the soft-max gradient gives d(soft_val)/d(ac[m]) = grad_ac[m]
    # and d(ac[m])/d(x[j]) is what we computed. So d(soft_val)/d(x[j]) = grad_x[j].
    
    grad_val_x = grad_x / (s * s) - 2.0 * soft_val / (s * s * s) * np.ones(N)
    # Wait, that's not right either. Let me be more careful.
    # val = soft_val / s^2
    # d(val)/d(x[j]) = (1/s^2) * d(soft_val)/d(x[j]) + soft_val * d(1/s^2)/d(x[j])
    #                = (1/s^2) * d(soft_val)/d(x[j]) + soft_val * (-2/s^3) * d(s)/d(x[j])
    #                = (1/s^2) * d(soft_val)/d(x[j]) - 2*soft_val/s^3
    
    # So:
    grad_val_x = grad_x / (s * s) - 2.0 * soft_val / (s * s * s)
    
    # Now d(val)/d(phase[k]) = sum_j d(val)/d(x[j]) * d(x[j])/d(phase[k])
    # d(x[j])/d(phase[k]) = i * fixed_mag[k] * exp(i*phase[k]) * exp(2*pi*i*k*j/L_fft)
    #                      + (-i) * fixed_mag[k] * exp(-i*phase[k]) * exp(-2*pi*i*k*j/L_fft)
    # = i * spectrum[k] * exp(2*pi*i*k*j/L_fft) - i * conj(spectrum[k]) * exp(-2*pi*i*k*j/L_fft)
    # = 2 * Im[spectrum[k] * exp(2*pi*i*k*j/L_fft)]
    
    # This is getting complex. Let me use a different approach:
    # d(val)/d(phase[k]) = Re[conj(d(val)/d(spectrum[k])) * d(spectrum[k])/d(phase[k])]
    #                     = Re[conj(d(val)/d(spectrum[k])) * i * spectrum[k]]
    
    # d(val)/d(spectrum[k]) = sum_j d(val)/d(x[j]) * d(x[j])/d(spectrum[k])
    # d(x[j])/d(spectrum[k]) = exp(2*pi*i*k*j/L_fft) for 0<k<L_fft/2
    
    # So d(val)/d(spectrum[k]) = FFT of grad_val_x at frequency k
    # i.e., d(val)/d(spectrum[k]) = (1/L_fft) * sum_j grad_val_x[j] * exp(-2*pi*i*k*j/L_fft)
    # Wait, that's the DFT. Let me use FFT.
    
    G = np.fft.rfft(grad_val_x, n=L_fft)
    # G[k] = sum_j grad_val_x[j] * exp(-2*pi*i*k*j/L_fft)
    # So d(val)/d(spectrum[k]) = G[k] / L_fft * L_fft = G[k] (since irfft has 1/L_fft factor)
    # Actually, x_full = irfft(spectrum), so d(x_full[j])/d(spectrum[k]) = exp(2*pi*i*k*j/L_fft)/L_fft * 2 (for symmetry)
    # This is getting messy. Let me just compute it directly.
    
    # d(val)/d(phase[k]) for 0 < k < L_fft/2:
    # = Re[conj(d(val)/d(spectrum[k])) * i * spectrum[k]]
    # d(val)/d(spectrum[k]) = sum_j grad_val_x[j] * d(x[j])/d(spectrum[k])
    # For irfft: x_full[j] = (2/L_fft) * Re[sum_{k=0}^{L_fft/2} spectrum[k] * exp(2*pi*i*k*j/L_fft)]
    # (with special handling for k=0 and k=L_fft/2)
    # d(x[j])/d(spectrum[k]) = (2/L_fft) * Re[exp(2*pi*i*k*j/L_fft)] ... no, that's not right.
    
    # Let me just do it numerically with a small perturbation check for now,
    # or use the fact that:
    # d(val)/d(phase[k]) = sum_j grad_val_x[j] * d(x[j])/d(phase[k])
    # d(x[j])/d(phase[k]) = 2 * Im[spectrum[k] * exp(2*pi*i*k*j/L_fft)] / L_fft * L_fft
    #                      = 2 * Im[spectrum[k] * exp(2*pi*i*k*j/L_fft)]
    
    # So d(val)/d(phase[k]) = sum_j grad_val_x[j] * 2 * Im[spectrum[k] * exp(2*pi*i*k*j/L_fft)]
    # = 2 * Im[spectrum[k] * sum_j grad_val_x[j] * exp(2*pi*i*k*j/L_fft)]
    # = 2 * Im[spectrum[k] * conj(G[k])]  (since G[k] = sum_j grad_val_x[j] * exp(-2*pi*i*k*j/L_fft))
    
    grad_phase = np.zeros(n_freq)
    for k in range(1, n_freq):
        if k == L_fft // 2:
            continue  # Nyquist frequency, real
        grad_phase[k] = 2.0 * np.imag(spectrum[k] * np.conj(G[k]))
    
    return soft_val / (s * s), grad_phase

def _phase_optimize(fixed_mag, L_fft, n_freq, rng, initial_phase=None, maxiter=300, beta_adapt=True):
    """
    Optimize phase for fixed amplitude spectrum.
    """
    if initial_phase is None:
        initial_phase = rng.uniform(0, 2 * np.pi, size=n_freq)
        initial_phase[0] = 0.0
    
    def obj_and_grad(phase):
        return _phase_objective_and_grad(phase, fixed_mag, L_fft, n_freq)
    
    try:
        result = minimize(obj_and_grad, initial_phase, method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12})
        return result.x, result.fun
    except Exception:
        return initial_phase, 1e10

def _phase_to_sequence(phase, fixed_mag, L_fft):
    """Convert phase spectrum to time-domain sequence."""
    spectrum = fixed_mag * np.exp(1j * phase)
    spectrum[0] = fixed_mag[0]
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
    
    # Try multiple amplitude spectrum modes
    for mode in range(8):
        if _check_deadline():
            break
        
        fixed_mag, L_fft, n_freq = _build_fixed_amplitude_spectrum(mode, rng)
        
        # Multiple random phase initializations
        for restart in range(3):
            if _check_deadline():
                break
            
            initial_phase = rng.uniform(0, 2 * np.pi, size=n_freq)
            initial_phase[0] = 0.0
            
            phase_opt, val_opt = _phase_optimize(fixed_mag, L_fft, n_freq, rng,
                                                  initial_phase, maxiter=200)
            
            if val_opt < best_val:
                best_val = val_opt
                x_candidate = _phase_to_sequence(phase_opt, fixed_mag, L_fft)
                best_x = x_candidate.copy()
    
    if _check_deadline():
        return best_x
    
    # Polish the best solution with full L-BFGS on the time-domain sequence
    # using a soft-max surrogate
    
    def _soft_max_and_grad(x, beta):
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        m = np.max(abs_ac)
        exps = np.exp(beta * (abs_ac - m))
        sum_exps = np.sum(exps)
        if sum_exps < 1e-300:
            return m, np.zeros(N)
        log_sum = np.log(sum_exps)
        soft_val = m + log_sum / beta
        weights = exps / sum_exps
        signs = np.sign(ac)
        signs[ac == 0] = 1.0
        grad_ac = weights * signs
        
        L_corr = 1
        while L_corr < 4 * N:
            L_corr *= 2
        
        h = np.zeros(2 * N - 1)
        h[N-1 : 2*N-1] = grad_ac[0 : N]
        h[0 : N-1] = grad_ac[2*N-2 : N-1 : -1]
        
        X_f = np.fft.rfft(x, n=L_corr)
        H_f = np.fft.rfft(h, n=L_corr)
        
        corr1 = np.fft.irfft(np.conj(H_f) * X_f, n=L_corr)[:N]
        h_rev = h[::-1]
        H_rev_f = np.fft.rfft(h_rev, n=L_corr)
        corr2 = np.fft.irfft(np.conj(H_rev_f) * X_f, n=L_corr)[:N]
        
        grad_x = corr1 + corr2
        return soft_val, grad_x
    
    def _objective_and_grad(x, beta):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros(N)
        soft_val, grad_soft = _soft_max_and_grad(x, beta)
        val = soft_val / (s * s)
        grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
        return val, grad
    
    for beta in [50.0, 100.0, 200.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-14})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Coordinate descent refinement
    step_size = 0.005
    no_improve = 0
    max_no_improve = 3000
    
    while no_improve < max_no_improve and not _check_deadline():
        n_perturb = rng.integers(2, 20)
        indices = rng.choice(N, size=n_perturb, replace=False)
        perturbation = np.zeros(N)
        perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.0005, step_size * 0.95)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.02, step_size * 1.01)
    
    return best_x
# EVOLVE-BLOCK-END
