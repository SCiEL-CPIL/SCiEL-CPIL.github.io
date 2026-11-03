"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Fast FFT-based autoconvolution for speed.
2. Multi-start parametric initialization (cosine bumps, parabolic, cosine wave).
3. Simulated annealing peak-shaving (inspired by the better-scoring approach)
   with FFT-accelerated sensitivity computation.
4. Spectral-domain projected gradient refinement.
5. Multi-scale polish pass with varying thresholds and adaptive steps.
6. Strict deadline guard to stay within 150s.

Key improvements over parent:
- Simulated annealing acceptance for peak-shaving (explores local minima better)
- FFT-based sensitivity (much faster than O(N*|P|) direct computation)
- More diverse initializations
- Better time allocation: more time for peak-shaving (the proven winner)
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 145  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_autocorr(f)
    return np.max(conv) / (s * s)

def fft_autocorr(f):
    """Compute autoconvolution using FFT for speed."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)[:size]
    return conv

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    shifted = x - x_max
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max + log_sum

def cosine_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from cosine bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
    return result

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    f = cosine_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    conv = fft_autocorr(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def compute_sensitivity_fft(current_h, peak_indices):
    """Compute sensitivity using FFT cross-correlation."""
    n = len(current_h)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2

    peak_ind = np.zeros(fft_size)
    peak_ind[peak_indices] = 1.0

    f_padded = np.zeros(fft_size)
    f_padded[:n] = current_h

    F = np.fft.rfft(f_padded)
    IP = np.fft.rfft(peak_ind)

    S = np.fft.irfft(F * np.conj(IP), n=fft_size)

    return S[:n]

def initial_guess_parabolic():
    """Create a parabolic (radial basis) initial guess."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = np.maximum(1.0 - (x / 0.25) ** 2, 0.0)
    return f

def initial_guess_cosine_wave():
    """Create a cosine wave initial guess."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = 0.5 * (1.0 + np.cos(2 * np.pi * x))
    return f

def peak_shaving_sa(f, start_time, max_time, rng=None, T_init=0.08, cooling=0.997):
    """Simulated-annealing peak-shaving with FFT sensitivity."""
    if rng is None:
        rng = np.random.RandomState(42)

    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()

    T = T_init
    no_improve = 0
    max_no_improve = 200

    while time.time() - start_time < max_time:
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)

        conv = fft_autocorr(current_f)
        max_c = np.max(conv)
        current_val = max_c / (N * N)

        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break

        # Find peak lags
        threshold = max_c * 0.985
        peak_indices = np.where(conv >= threshold)[0]
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])

        # FFT sensitivity
        sensitivity = compute_sensitivity_fft(current_f, peak_indices)

        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std

        # Update with noise for SA
        noise = rng.normal(0, 0.05, size=N)
        update = -T * sens_norm + T * 0.3 * noise

        f_new = current_f * (1.0 + update)
        f_new = np.maximum(f_new, 0.0)

        s_new = np.sum(f_new)
        if s_new < 1e-12:
            T *= cooling
            continue
        f_new = f_new * (N / s_new)

        conv_new = fft_autocorr(f_new)
        val_new = np.max(conv_new) / (N * N)

        # Accept if better or with SA probability
        if val_new < current_val:
            current_f = f_new
            T = max(T * cooling, 1e-6)
        else:
            delta = val_new - current_val
            if delta > 0 and T > 1e-8:
                accept_prob = np.exp(-delta / (T + 1e-12))
                if rng.random() < accept_prob:
                    current_f = f_new
            T *= cooling

        if T < 1e-7:
            break

    return best_f, best_val

def spectral_refinement(h, start_time, max_time):
    """Spectral-domain refinement using projected gradient descent."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()

    s = np.sum(current_h)
    if s < 1e-12:
        return best_h, best_val
    current_h = current_h * (N / s)

    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    H = np.fft.rfft(current_h, n=fft_size)

    lr = 0.02
    min_lr = 1e-8
    no_improve_count = 0
    max_no_improve = 80

    while time.time() - start_time < max_time:
        h_recon = np.fft.irfft(H, n=fft_size)[:N]
        h_recon = np.maximum(h_recon, 0.0)

        s_recon = np.sum(h_recon)
        if s_recon < 1e-12:
            break
        h_recon = h_recon * (N / s_recon)

        conv = fft_autocorr(h_recon)
        max_c = np.max(conv)
        current_val = max_c / (N * N)

        if current_val < best_val:
            best_val = current_val
            best_h = h_recon.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break

        threshold = max_c * 0.97
        peak_indices = np.where(conv >= threshold)[0]
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])

        sensitivity = compute_sensitivity_fft(h_recon, peak_indices)
        S = np.fft.rfft(sensitivity, n=fft_size)

        H_new = H - lr * S
        h_new = np.fft.irfft(H_new, n=fft_size)[:N]
        h_new = np.maximum(h_new, 0.0)

        s_new = np.sum(h_new)
        if s_new < 1e-12:
            lr *= 0.5
            if lr < min_lr:
                break
            continue
        h_new = h_new * (N / s_new)

        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)

        if val_new < current_val * (1.0 - 1e-10):
            H = np.fft.rfft(h_new, n=fft_size)
            current_h = h_new
            lr = min(lr * 1.01, 0.2)
            no_improve_count = 0
        else:
            lr *= 0.93
            if lr < min_lr:
                break

    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Multi-scale polish pass with varying thresholds and adaptive step sizes."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()

    scales = [
        {'alpha': 0.008, 'threshold': 0.99, 'name': 'coarse'},
        {'alpha': 0.004, 'threshold': 0.995, 'name': 'medium'},
        {'alpha': 0.0015, 'threshold': 0.999, 'name': 'fine'},
        {'alpha': 0.0005, 'threshold': 0.9995, 'name': 'ultra_fine'},
    ]

    elapsed = time.time() - start_time
    remaining = max_time - elapsed
    if remaining < 1.0:
        return best_h, best_val

    time_weights = [0.15, 0.25, 0.35, 0.25]

    for scale_idx, scale in enumerate(scales):
        scale_time = time_weights[scale_idx] * remaining
        scale_deadline = time.time() + scale_time

        alpha = scale['alpha']
        threshold_frac = scale['threshold']
        min_alpha = 1e-7
        no_improve = 0
        max_no_improve = 40

        while time.time() < scale_deadline and time.time() - start_time < max_time:
            s = np.sum(current_h)
            if s < 1e-12:
                break
            current_h = current_h * (N / s)

            conv = fft_autocorr(current_h)
            max_c = np.max(conv)
            current_val = max_c / (N * N)

            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

            threshold = max_c * threshold_frac
            peak_indices = np.where(conv >= threshold)[0]
            if len(peak_indices) == 0:
                peak_indices = np.array([np.argmax(conv)])

            sensitivity = compute_sensitivity_fft(current_h, peak_indices)

            sens_mean = np.mean(sensitivity)
            sens_std = np.std(sensitivity) + 1e-12
            sens_norm = (sensitivity - sens_mean) / sens_std

            h_new = current_h * (1.0 - alpha * sens_norm)
            h_new = np.maximum(h_new, 0.0)

            s_new = np.sum(h_new)
            if s_new < 1e-12:
                break
            h_new = h_new * (N / s_new)

            conv_new = fft_autocorr(h_new)
            val_new = np.max(conv_new) / (N * N)

            if val_new < current_val * (1.0 - 1e-12):
                current_h = h_new
                alpha = min(alpha * 1.008, 0.04)
                no_improve = 0
            else:
                alpha *= 0.97
                if alpha < min_alpha:
                    break

    return best_h, best_val

def construct():
    start_time = time.time()
    rng = np.random.RandomState(7)

    # ---- Phase 1: Generate diverse initializations ----
    best_f = np.ones(N)
    best_score = compute_score(best_f)

    # Parabolic
    f_par = initial_guess_parabolic()
    sc = compute_score(f_par)
    if sc < best_score:
        best_f, best_score = f_par, sc

    # Cosine wave
    f_cos = initial_guess_cosine_wave()
    sc = compute_score(f_cos)
    if sc < best_score:
        best_f, best_score = f_cos, sc

    # Box functions
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    for w in [0.08, 0.12, 0.16, 0.2, 0.25, 0.3]:
        f_box = np.zeros(N)
        f_box[np.abs(x) <= w / 2.0] = 1.0
        sc = compute_score(f_box)
        if sc < best_score:
            best_f, best_score = f_box, sc

    # Parametric cosine bumps with Nelder-Mead
    if time.time() - start_time < TIME_LIMIT - 60:
        for n_c in [3, 4]:
            if time.time() - start_time > TIME_LIMIT - 50:
                break
            best_params = None
            best_obj = np.inf
            for beta in [10.0, 25.0]:
                if time.time() - start_time > TIME_LIMIT - 40:
                    break
                for restart in range(3):
                    if time.time() - start_time > TIME_LIMIT - 45:
                        break
                    if restart == 0:
                        centers = np.linspace(-0.15, 0.15, n_c)
                        widths = np.full(n_c, 0.15)
                        heights = np.full(n_c, 1.0)
                    else:
                        centers = rng.uniform(-0.2, 0.2, size=n_c)
                        widths = rng.uniform(0.04, 0.28, size=n_c)
                        heights = rng.uniform(0.3, 2.5, size=n_c)
                    x0 = np.concatenate([centers, widths, heights])
                    try:
                        result = minimize(
                            objective, x0, args=(n_c, beta),
                            method='Nelder-Mead',
                            options={'maxiter': 600, 'xatol': 1e-6,
                                     'fatol': 1e-8, 'adaptive': True}
                        )
                        if result.fun < best_obj:
                            best_obj = result.fun
                            best_params = result.x
                    except Exception:
                        pass
            if best_params is not None:
                f = parametric_to_step(best_params, n_c)
                actual = compute_score(f)
                if actual < best_score:
                    best_f, best_score = f, actual

    # ---- Phase 2: Simulated-annealing peak-shaving (the key improvement) ----
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 20.0:
        sa_time = min(remaining * 0.45, 55.0)
        sa_deadline = time.time() + sa_time
        sa_f, sa_score = peak_shaving_sa(
            best_f, start_time, sa_deadline, rng=rng,
            T_init=0.06, cooling=0.9975
        )
        if sa_score < best_score:
            best_f, best_score = sa_f, sa_score

    # ---- Phase 3: Spectral refinement ----
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 15.0:
        spec_time = min(remaining * 0.25, 30.0)
        spec_deadline = time.time() + spec_time
        spec_f, spec_score = spectral_refinement(
            best_f, start_time, spec_deadline
        )
        if spec_score < best_score:
            best_f, best_score = spec_f, spec_score

    # ---- Phase 4: Multi-scale polish ----
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 8.0:
        polish_deadline = time.time() + min(remaining * 0.30, 35.0)
        pol_f, pol_score = polish_pass(
            best_f, start_time, polish_deadline
        )
        if pol_score < best_score:
            best_f, best_score = pol_f, pol_score

    # ---- Phase 5: Final SA polish if time remains ----
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 12.0:
        final_deadline = time.time() + min(remaining * 0.40, 25.0)
        fin_f, fin_score = peak_shaving_sa(
            best_f, start_time, final_deadline, rng=rng,
            T_init=0.02, cooling=0.9985
        )
        if fin_score < best_score:
            best_f, best_score = fin_f, fin_score

    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
