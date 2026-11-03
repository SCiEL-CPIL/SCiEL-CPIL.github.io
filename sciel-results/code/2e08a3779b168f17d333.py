"""Model-and-optimize: spectral envelope + epigraph + exact peak gradient + high-precision polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 4)

def _fft_len(n):
    L = 1
    while L < n:
        L *= 2
    return L

def _autoconv_fft(x):
    L = 2 * N - 1
    L_fft = _fft_len(L)
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    m = np.max(np.abs(ac))
    return m / (s * s)

def _active_peak_grad(x, beta=40.0):
    """
    Gradient of soft-max surrogate for max|ac| w.r.t. x.
    ac[k] = sum_i x[i]*x[k-i]; d|ac[k]|/dx[n] = sign(ac[k])*(x[k-n] + x[n-k]) (valid terms).
    We compute via cross-correlation with weights w[k]=beta*exp(beta*(|ac[k]|-m)).
    """
    L = 2 * N - 1
    L_fft = _fft_len(L)
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

    # d/dx[n] ac[k] = x[k-n] (if 0<=k-n<N) + x[n-k] (if 0<=n-k<N)
    # For fixed n, sum over k: grad_ac[k]*x[k-n] + grad_ac[k]*x[n-k]
    # First term: corr1[n] = sum_k grad_ac[k] x[k-n]
    # Second term: corr2[n] = sum_k grad_ac[k] x[n-k] = corr1_reflected
    L_corr = _fft_len(3 * N)
    g_padded = np.zeros(L_corr)
    g_padded[:L] = grad_ac
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    # corr1[n] = sum_k g[k] x[k-n] = (g * x)(n) with appropriate shift
    # Use: y[n] = sum_k g[k] x[k-n]. This is cross-correlation.
    # y = ifft( conj(fft(x)) * fft(g) ) with proper length handling.
    corr1 = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)[:N]
    # For second term: sum_k g[k] x[n-k]. This equals sum_k g[k] x'[-(k-n)]
    # Equivalent to corr with x reversed.
    x_rev = x[::-1]
    Xr_f = np.fft.rfft(x_rev, n=L_corr)
    corr2 = np.fft.irfft(G_f * np.conj(Xr_f), n=L_corr)[:N]
    # corr2[n] corresponds to sum_k g[k] x_rev[k-n] = sum_k g[k] x[N-1-(k-n)]
    # We need sum_k g[k] x[n-k]. Let j = n-k => x[j], k = n-j, sum_j g[n-j] x[j].
    # This is a different cross-correlation. Instead, note by symmetry:
    # sum_k g[k] x[n-k] = sum_k g[k] x[-(k-n)] = corr of g with reversed x, shifted.
    # Simpler: since x is real and we want d|ac|/dx = sign*(x[k-n]+x[n-k]),
    # and ac is symmetric (ac[k]=ac[L-1-k] for real? no). Actually for real x, ac[k] = ac[L-1-k].
    # So g[k] = g[L-1-k]. Then the two terms are equal! So grad = 2*corr1.
    # Let's verify: ac[k]=sum_i x[i]x[k-i]. ac[L-1-k]=sum_i x[i]x[L-1-k-i].
    # With j = L-1-k-i => i = L-1-k-j: sum_j x[L-1-k-j] x[j] = ac[k] (by symmetry of sum). Yes symmetric.
    # So grad_ac symmetric, and the two cross-correlation terms are identical.
    grad_x = 2.0 * corr1[:N]
    return soft_val, grad_x

def _objective_and_grad(x, beta=40.0):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _active_peak_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _epigraph_obj_and_grad(x, beta=40.0, t_penalty=10.0):
    """
    Epigraph: minimize max|ac|/s^2 + t_penalty * (t - max|ac|/s^2)^+ approximated.
    We use a smooth penalty: t_penalty * softplus(t - val).
    Returns objective and gradient w.r.t. x (t optimized implicitly as t=val).
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _active_peak_grad(x, beta)
    val = soft_val / (s * s)
    # t = val, penalty softplus(t - val) = 0. So just return val.
    # Instead, add a small regularization to encourage smoothness.
    reg = 1e-6 * np.sum(x * x)
    val_total = val + reg
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s) + 2e-6 * x
    return val_total, grad

def _spectral_envelope_optimize(x0, rng, max_iters=250, lr=0.02):
    L = 2 * N - 1
    L_fft = _fft_len(L)
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
        val, grad = _objective_and_grad(best_x, 40.0)
        G = np.fft.rfft(grad, n=L_fft)
        amp_grad = np.real(X * np.conj(G)) / (amp + 1e-10)
        amp_grad[0] = 0
        amp_trial = amp - lr * amp_grad
        amp_trial = np.maximum(0.01, amp_trial)
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
    L = 2 * N - 1
    L_fft = _fft_len(L)
    X = np.fft.rfft(x0, n=L_fft)
    amp = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    best_x = x0.copy()
    best_val = _objective_only(x0)
    for it in range(max_iters):
        if _check_deadline():
            break
        val, grad = _objective_and_grad(best_x, 40.0)
        G = np.fft.rfft(grad, n=L_fft)
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
        top_k = min(15, len(abs_ac))
        top_idx = np.argsort(abs_ac)[-top_k:]
        peak_lag = top_idx[np.argmax(abs_ac[top_idx])]
        sign_peak = np.sign(ac[peak_lag]) if ac[peak_lag] != 0 else 1.0
        n_perturb = rng.integers(3, 10)
        indices = rng.choice(N, size=n_perturb, replace=False)
        grad_local = np.zeros(n_perturb)
        for i, n in enumerate(indices):
            contrib = 0.0
            if 0 <= peak_lag - n < N:
                contrib += best_x[peak_lag - n]
            if 0 <= n - peak_lag < N:
                contrib += best_x[n - peak_lag]
            grad_local[i] = sign_peak * 2.0 * contrib
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
    L = 2 * N - 1
    L_fft = _fft_len(L)
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
    for mode in range(6):
        for _ in range(2):
            if _check_deadline():
                break
            initial_points.append(_spectral_init(rng, mode))
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
        x_envelope, val_envelope = _spectral_envelope_optimize(x_start, rng, max_iters=150, lr=0.02)
        if val_envelope < best_val:
            best_val = val_envelope
            best_x = x_envelope.copy()
        candidates.append((val_envelope, x_envelope))
        if _check_deadline():
            break
        x_phase, val_phase = _phase_optimize(x_envelope, rng, max_iters=150, lr=0.01)
        if val_phase < best_val:
            best_val = val_phase
            best_x = x_phase.copy()
        candidates.append((val_phase, x_phase))
        if _check_deadline():
            break
        for beta in [40.0]:
            if _check_deadline():
                break
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                result = minimize(obj_and_grad_func, x_phase, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10})
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                candidates.append((result.fun, result.x))
            except Exception:
                pass
    if _check_deadline():
        return best_x
    candidates.sort(key=lambda item: item[0])
    top_candidates = [x for _, x in candidates[:6]]
    for x_init in top_candidates:
        if _check_deadline():
            break
        x_polished, val_polished = _peak_targeted_polish(x_init, rng, max_iters=600, step_size=0.01)
        if val_polished < best_val:
            best_val = val_polished
            best_x = x_polished.copy()
        if _check_deadline():
            break
        for beta in [100.0, 300.0]:
            if _check_deadline():
                break
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                result = minimize(obj_and_grad_func, x_polished, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 150, 'gtol': 1e-10, 'ftol': 1e-12})
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
    # Final coordinate descent polish
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
