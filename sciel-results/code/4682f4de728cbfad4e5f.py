"""Model-and-optimize: spectral init + fast L-BFGS + epigraph + peak-targeted + coordinate descent."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 143

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 4)

# Precompute FFT size
L = 2 * N - 1
L_fft = 1
while L_fft < L:
    L_fft *= 2

def _autoconv_fft(x):
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

def _soft_max_and_grad(x, beta):
    """Soft-max of |ac| with gradient via FFT cross-correlation."""
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs

    # grad_x[n] = sum_k grad_ac[k] * 2 * x[k-n] (valid range)
    # = 2 * cross-correlation(grad_ac, x) at lag n
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    grad_x = 2.0 * corr[:N]
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_init(rng, mode=0):
    if mode == 0:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(L_fft // 2 + 1)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=L_fft // 2 + 1)
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
    else:
        mag = np.ones(L_fft // 2 + 1)

    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    s = np.sum(x)
    if s < 0:
        x = -x
    return x

def _peak_targeted_polish(x0, rng, max_iters=2000, step_size=0.01):
    """Targeted perturbations to reduce peak |ac| lags. Uses symmetry: ac[k]=ac[2N-2-k]."""
    best_x = x0.copy()
    best_val = _objective_only(x0)
    ac = _autoconv_fft(best_x)
    abs_ac = np.abs(ac)

    for it in range(max_iters):
        if _check_deadline():
            break
        if it % 10 == 0:
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)

        # Use symmetry: only consider lags 0..N-1 (half), since ac[k]=ac[2N-2-k]
        # But the peak could be anywhere; use top lags from full array
        top_k = 12
        sorted_idx = np.argsort(-abs_ac)[:top_k]

        improved = False
        for k in sorted_idx:
            if _check_deadline():
                break
            n_lo = max(0, k - N + 1)
            n_hi = min(k, N - 1)
            if n_hi >= n_lo:
                idx_in_x = k - np.arange(n_lo, n_hi + 1)
                contrib_segment = 2.0 * best_x[idx_in_x]
                contrib = np.zeros(N)
                contrib[n_lo:n_hi+1] = contrib_segment
                norm_contrib = np.linalg.norm(contrib)
                if norm_contrib > 1e-10:
                    direction = -contrib / norm_contrib
                    for scale in [step_size, step_size * 0.5, step_size * 1.5]:
                        x_trial = best_x + scale * direction
                        val = _objective_only(x_trial)
                        if val < best_val:
                            best_val = val
                            best_x = x_trial.copy()
                            improved = True
                            break
        if improved:
            step_size = max(0.001, step_size * 0.93)
        else:
            step_size = min(0.08, step_size * 1.03)
    return best_x, best_val

def _epigraph_polish(x0, rng, max_iters=500):
    """Epigraph formulation: minimize t s.t. |ac[k]|/(s^2) <= t for all k.
    Use smooth constraint: sum exp(beta*(|ac[k]|/s^2 - t)) <= 1."""
    best_x = x0.copy()
    best_val = _objective_only(x0)
    beta = 80.0
    step_size = 0.005

    for it in range(max_iters):
        if _check_deadline():
            break
        ac = _autoconv_fft(best_x)
        s = np.sum(best_x)
        if abs(s) < 1e-10:
            break
        ac_norm = np.abs(ac) / (s * s)
        t = np.max(ac_norm)

        # Find violating lags (those near t)
        violations = ac_norm > t * 0.95
        if not np.any(violations):
            step_size = max(0.001, step_size * 0.9)
            continue

        violating_idx = np.where(violations)[0]
        # Pick the worst violator
        k = violating_idx[np.argmax(ac_norm[violating_idx])]

        n_lo = max(0, k - N + 1)
        n_hi = min(k, N - 1)
        if n_hi >= n_lo:
            idx_in_x = k - np.arange(n_lo, n_hi + 1)
            contrib_segment = 2.0 * best_x[idx_in_x]
            contrib = np.zeros(N)
            contrib[n_lo:n_hi+1] = contrib_segment
            norm_contrib = np.linalg.norm(contrib)
            if norm_contrib > 1e-10:
                direction = -contrib / norm_contrib
                x_trial = best_x + step_size * direction
                val = _objective_only(x_trial)
                if val < best_val:
                    best_val = val
                    best_x = x_trial.copy()
                    step_size = min(0.02, step_size * 1.1)
                else:
                    step_size = max(0.001, step_size * 0.85)
    return best_x, best_val

def _coordinate_descent(x0, rng, max_iters=50000, step_size=0.02):
    """Mixed random + structured coordinate descent."""
    best_x = x0.copy()
    best_val = _objective_only(x0)
    no_improve = 0
    t_arr = np.arange(N)

    while no_improve < max_iters and not _check_deadline():
        r = rng.random()
        if r < 0.15:
            idx = rng.integers(0, N)
            delta = step_size * rng.choice([-1, 1])
            x_trial = best_x.copy()
            x_trial[idx] += delta
        elif r < 0.35:
            n_perturb = rng.integers(2, 10)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
            x_trial = best_x + perturbation
        elif r < 0.6:
            freq = rng.integers(1, 10)
            phase = rng.uniform(0, 2 * np.pi)
            perturbation = step_size * np.sin(2 * np.pi * freq * t_arr / N + phase)
            x_trial = best_x + perturbation
        elif r < 0.8:
            freq1 = rng.integers(1, 5)
            freq2 = rng.integers(5, 20)
            phase1 = rng.uniform(0, 2 * np.pi)
            phase2 = rng.uniform(0, 2 * np.pi)
            perturbation = step_size * (0.7 * np.sin(2 * np.pi * freq1 * t_arr / N + phase1) +
                                        0.3 * np.sin(2 * np.pi * freq2 * t_arr / N + phase2))
            x_trial = best_x + perturbation
        else:
            n_perturb = rng.integers(5, 30)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size * 2.5, size=n_perturb)
            x_trial = best_x + perturbation

        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.0005, step_size * 0.96)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.1, step_size * 1.02)

        if no_improve > 2000 and rng.random() < 0.01:
            x_restart = best_x + rng.normal(0, 0.08, size=N)
            val_restart = _objective_only(x_restart)
            if val_restart < best_val:
                best_val = val_restart
                best_x = x_restart.copy()
                no_improve = 0
                step_size = 0.02

    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)

    if _check_deadline():
        return best_x

    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, N)

    # Focused initial points (fewer, higher quality)
    initial_points = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.0 * t),
        np.exp(-1.0 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5)**2,
        1.0 + 0.5 * np.sin(6 * np.pi * t) * np.cos(2 * np.pi * t),
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),
        1.0 + 0.4 * np.sin(8 * np.pi * t),
    ]

    # Spectral inits
    for mode in range(5):
        for _ in range(2):
            initial_points.append(_spectral_init(rng, mode))

    # A few random
    for _ in range(4):
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))

    # Fast L-BFGS on a subset of initial points (fewer, faster)
    betas = [50.0, 150.0]
    for x0 in initial_points[:18]:
        if _check_deadline():
            break
        x_start = x0 + rng.normal(0, 0.01, size=N)
        for beta in betas:
            if _check_deadline():
                break
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 150, 'gtol': 1e-8, 'ftol': 1e-11, 'maxls': 30})
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

    if _check_deadline():
        return best_x

    # Peak-targeted polish (primary workhorse)
    x_peak, val_peak = _peak_targeted_polish(best_x, rng, max_iters=1500, step_size=0.01)
    if val_peak < best_val:
        best_val = val_peak
        best_x = x_peak.copy()

    if _check_deadline():
        return best_x

    # Epigraph polish
    x_epi, val_epi = _epigraph_polish(best_x, rng, max_iters=400)
    if val_epi < best_val:
        best_val = val_epi
        best_x = x_epi.copy()

    if _check_deadline():
        return best_x

    # Coordinate descent (long phase)
    x_cd, val_cd = _coordinate_descent(best_x, rng, max_iters=40000, step_size=0.02)
    if val_cd < best_val:
        best_val = val_cd
        best_x = x_cd.copy()

    if _check_deadline():
        return best_x

    # Final high-precision L-BFGS polish
    for beta in [300.0, 800.0, 2000.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-13, 'maxls': 40})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass

    # Final peak-targeted micro-polish
    x_final, val_final = _peak_targeted_polish(best_x, rng, max_iters=500, step_size=0.005)
    if val_final < best_val:
        best_val = val_final
        best_x = x_final.copy()

    return best_x
# EVOLVE-BLOCK-END
