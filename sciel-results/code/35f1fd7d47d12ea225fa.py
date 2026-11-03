"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Parametric initialization with cosine bumps (annealed smooth-max beta)
2. Spectral coordinate descent: optimize individual |F(k)| magnitudes to
   directly flatten the autoconvolution peak (key structural addition)
3. Block-level simulated annealing for structural changes
4. FFT-based sensitivity local polish for fine-tuning

The spectral coordinate descent is the critical addition: since the
autoconvolution is the inverse-FFT of |F(k)|^2, we can directly optimize
the magnitude spectrum to minimize the peak, which is a different
optimization path from time-domain methods and can escape local minima.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill
N_FFT = 2 * N

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    f_hat = np.fft.rfft(f, n=N_FFT)
    conv = np.fft.irfft(f_hat * f_hat, n=N_FFT)[:2*N-1]
    return np.max(conv) / (s * s)

def fft_convolve(f):
    """Fast autoconvolution via FFT."""
    f_hat = np.fft.rfft(f, n=N_FFT)
    conv = np.fft.irfft(f_hat * f_hat, n=N_FFT)[:2*N-1]
    return conv

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    shifted = x / x_max - 1.0
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

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
    return np.maximum(f, 0.0)

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_convolve(f)
    return smooth_max(conv, beta) / (s * s)

def spectral_coordinate_descent(h, start_time, max_time):
    """Spectral coordinate descent: optimize individual |F(k)| magnitudes
    to minimize the peak of the autoconvolution.
    
    Key insight: autoconvolution = IFFT(|F(k)|^2). By directly optimizing
    the magnitude spectrum, we can flatten the autoconvolution peak through
    a fundamentally different path than time-domain optimization.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    
    # Work with the magnitude spectrum
    f_hat = np.fft.rfft(h, n=N_FFT)
    mag = np.abs(f_hat).copy()
    phase = np.angle(f_hat)
    n_freqs = len(mag)
    
    # Ensure DC component is fixed (sum of f is preserved)
    dc_val = mag[0]
    
    # Coordinate descent: cycle through frequency bins, adjust each magnitude
    # to minimize the peak autoconvolution
    no_improve = 0
    max_no_improve = 50
    step_size = 0.05
    
    while time.time() - start_time < max_time:
        # Reconstruct time domain from current magnitude spectrum
        f_recon = mag * np.exp(1j * phase)
        f_time = np.fft.irfft(f_recon, n=N_FFT)[:N]
        f_time = np.maximum(f_time, 0.0)
        s = np.sum(f_time)
        if s < 1e-12:
            break
        f_time = f_time * (N / s)
        
        conv = fft_convolve(f_time)
        current_val = np.max(conv) / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = f_time.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            break
        
        # Compute sensitivity of peak to each frequency bin
        # d(conv_peak)/d(mag[k]) involves the phase and other magnitude bins
        # Use numerical gradient for a few key bins
        peak_idx = np.argmax(conv)
        peak_val = conv[peak_idx]
        
        # Try adjusting random subset of frequency bins
        n_adjust = min(20, n_freqs - 1)
        idx_candidates = np.random.randint(1, n_freqs, size=max(50, n_adjust * 3))
        
        improved_this_round = False
        for _ in range(n_adjust):
            if time.time() - start_time > max_time:
                break
            
            k = idx_candidates[np.random.randint(len(idx_candidates))]
            if k == 0:
                continue
            
            # Try small adjustments to mag[k]
            base_mag = mag[k]
            best_k_val = base_mag
            best_k_obj = current_val
            
            for delta in [step_size, -step_size, step_size * 0.5, -step_size * 0.5]:
                mag_trial = mag.copy()
                mag_trial[k] = max(0.0, base_mag + delta * base_mag)
                
                f_recon_trial = mag_trial * np.exp(1j * phase)
                f_time_trial = np.fft.irfft(f_recon_trial, n=N_FFT)[:N]
                f_time_trial = np.maximum(f_time_trial, 0.0)
                s_trial = np.sum(f_time_trial)
                if s_trial < 1e-12:
                    continue
                f_time_trial = f_time_trial * (N / s_trial)
                
                conv_trial = fft_convolve(f_time_trial)
                val_trial = np.max(conv_trial) / (N * N)
                
                if val_trial < best_k_obj:
                    best_k_obj = val_trial
                    best_k_val = mag_trial[k]
            
            if best_k_val != base_mag:
                mag[k] = best_k_val
                improved_this_round = True
        
        if not improved_this_round:
            # Reduce step size
            step_size *= 0.7
            if step_size < 0.001:
                break
        else:
            step_size = min(step_size * 1.05, 0.1)
    
    return best_h, best_val

def spectral_refinement(h, start_time, max_time):
    """Parametric spectral model optimization: fit Lorentzian bumps to
    log-magnitude spectrum, optimize to minimize peak autoconvolution."""
    best_h = h.copy()
    best_val = compute_score(h)
    
    f_hat = np.fft.rfft(h, n=N_FFT)
    mag = np.abs(f_hat)
    n_freqs = len(mag)
    
    def spectral_objective(params, n_bumps):
        centers = params[:n_bumps]
        widths = params[n_bumps:2*n_bumps]
        heights = params[2*n_bumps:3*n_bumps]
        
        widths = np.abs(widths) + 0.5
        heights = np.abs(heights)
        
        freqs = np.arange(n_freqs, dtype=float)
        log_mag = np.zeros(n_freqs)
        for c, w, hgt in zip(centers, widths, heights):
            log_mag += hgt / (1.0 + ((freqs - c) / w) ** 2)
        
        mag_model = np.exp(log_mag)
        
        # Preserve DC
        mag_model[0] = mag[0]
        
        f_recon = mag_model * np.exp(1j * np.angle(f_hat))
        f_time = np.fft.irfft(f_recon, n=N_FFT)[:N]
        f_time = np.maximum(f_time, 0.0)
        
        s = np.sum(f_time)
        if s < 1e-12:
            return 1e10
        f_time = f_time * (N / s)
        
        conv = fft_convolve(f_time)
        return np.max(conv) / (N * N)
    
    rng = np.random.RandomState(42)
    best_params = None
    best_obj = np.inf
    
    for n_bumps in [3, 5]:
        if time.time() > max_time:
            break
        
        for restart in range(3):
            if time.time() > max_time:
                break
            
            # Initialize from peaks in current magnitude
            peaks = []
            for i in range(2, n_freqs-1):
                if mag[i] > mag[i-1] and mag[i] > mag[i+1] and mag[i] > np.mean(mag) * 0.3:
                    peaks.append(i)
            
            if len(peaks) >= n_bumps:
                peak_vals = [mag[p] for p in peaks]
                top_idx = np.argsort(peak_vals)[-n_bumps:]
                centers = np.array([peaks[i] for i in top_idx], dtype=float)
            else:
                centers = rng.uniform(5, n_freqs-5, size=n_bumps)
            
            widths = rng.uniform(5.0, 30.0, size=n_bumps)
            heights = np.zeros(n_bumps)
            for i, c in enumerate(centers):
                c_int = int(np.clip(c, 0, n_freqs-1))
                heights[i] = np.log(mag[c_int] + 1e-10)
            
            x0 = np.concatenate([centers, widths, heights])
            
            try:
                result = minimize(
                    spectral_objective, x0, args=(n_bumps,),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 0.5, 'fatol': 1e-5}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass
    
    if best_params is not None:
        n_bumps = best_params.size // 3
        centers = best_params[:n_bumps]
        widths = np.abs(best_params[n_bumps:2*n_bumps]) + 0.5
        heights = np.abs(best_params[2*n_bumps:3*n_bumps])
        
        freqs = np.arange(n_freqs, dtype=float)
        log_mag = np.zeros(n_freqs)
        for c, w, hgt in zip(centers, widths, heights):
            log_mag += hgt / (1.0 + ((freqs - c) / w) ** 2)
        
        mag_model = np.exp(log_mag)
        mag_model[0] = mag[0]
        
        f_recon = mag_model * np.exp(1j * np.angle(f_hat))
        f_time = np.fft.irfft(f_recon, n=N_FFT)[:N]
        f_time = np.maximum(f_time, 0.0)
        
        s = np.sum(f_time)
        if s > 1e-12:
            f_time = f_time * (N / s)
            val = compute_score(f_time)
            if val < best_val:
                best_h = f_time
                best_val = val
    
    return best_h, best_val

def block_simulated_annealing(h, start_time, max_time):
    """Block-level simulated annealing for structural changes."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    
    T_start = 0.08
    T_end = 0.001
    T_current = T_start
    block_sizes = [4, 8, 16, 32]
    
    no_improve = 0
    max_no_improve = 150
    total_steps = 0
    
    while time.time() - start_time < max_time:
        total_steps += 1
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        bs = block_sizes[np.random.randint(len(block_sizes))]
        n_blocks = N // bs
        if n_blocks < 2:
            bs = 4
            n_blocks = N // bs
        
        block_idx = np.random.randint(0, n_blocks)
        start_idx = block_idx * bs
        end_idx = start_idx + bs
        
        mode = np.random.randint(3)
        h_new = current_h.copy()
        
        if mode == 0:
            scale = np.random.uniform(0.6, 1.4)
            h_new[start_idx:end_idx] *= scale
        elif mode == 1:
            half = bs // 2
            delta = np.random.uniform(-0.3, 0.3) * np.mean(current_h[start_idx:end_idx])
            h_new[start_idx:start_idx+half] -= delta
            h_new[start_idx+half:end_idx] += delta
        else:
            center = start_idx + bs // 2
            sigma = max(2, bs // 4)
            x = np.arange(start_idx, end_idx)
            bump = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
            amp = np.random.uniform(-0.2, 0.2) * np.mean(current_h)
            h_new[start_idx:end_idx] += amp * bump
        
        h_new = np.maximum(h_new, 0.0)
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            continue
        h_new = h_new * (N / s_new)
        
        conv_new = fft_convolve(h_new)
        val_new = np.max(conv_new) / (N * N)
        val_cur = np.max(fft_convolve(current_h)) / (N * N)
        
        if val_new < val_cur or np.random.random() < np.exp(-(val_new - val_cur) / max(T_current, 1e-10)):
            current_h = h_new
            no_improve = 0
        else:
            no_improve += 1
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            no_improve = 0
        
        cooling_rate = (T_end / T_start) ** (1.0 / max(800, total_steps))
        T_current = max(T_current * cooling_rate, T_end)
        
        if no_improve > max_no_improve:
            current_h = best_h.copy()
            T_current = T_start * 0.5
            no_improve = 0
    
    return best_h, best_val

def local_polish(h, start_time, max_time):
    """FFT-based sensitivity local polish for fine-tuning."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.03
    threshold_frac = 0.98
    
    no_improve_count = 0
    max_no_improve = 40
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        f_hat = np.fft.rfft(current_h, n=N_FFT)
        conv = np.fft.irfft(f_hat * f_hat, n=N_FFT)[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            break
        
        # Smooth peak mask using FFT
        lower_thresh = max_c * 0.93
        peak_mask = np.clip((conv - lower_thresh) / (max_c * (1 - threshold_frac) + 1e-15), 0, 1)
        peak_mask = peak_mask ** 2
        
        pm_hat = np.fft.rfft(peak_mask, n=N_FFT)
        ch_hat = np.fft.rfft(current_h, n=N_FFT)
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=N_FFT)[:N]
        
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        best_alpha = alpha
        best_h_candidate = None
        best_val_candidate = current_val
        
        for trial_alpha in [alpha, alpha * 0.5, alpha * 1.5]:
            h_new = current_h * (1.0 - trial_alpha * sens_norm)
            h_new = np.maximum(h_new, 0.0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                continue
            h_new = h_new * (N / s_new)
            
            f_new_hat = np.fft.rfft(h_new, n=N_FFT)
            conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=N_FFT)[:2*N-1]
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < best_val_candidate:
                best_val_candidate = val_new
                best_alpha = trial_alpha
                best_h_candidate = h_new
        
        if best_h_candidate is not None and best_val_candidate < current_val:
            current_h = best_h_candidate
            alpha = min(alpha * 1.1, 0.15)
        else:
            alpha *= 0.8
            if alpha < 1e-6:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Parametric initialization with cosine bumps
    # Budget: 12% of time
    param_budget_end = start_time + (TIME_LIMIT - 5) * 0.12
    
    for n_c in [2, 3, 4]:
        if time.time() > param_budget_end:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        for beta in [5.0, 10.0, 20.0]:
            if time.time() > param_budget_end:
                break
            
            for restart in range(2):
                if time.time() > param_budget_end:
                    break
                
                centers = rng.uniform(-0.15, 0.15, size=n_c)
                widths = rng.uniform(0.08, 0.25, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6}
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x
                except Exception:
                    pass
        
        if best_params is not None:
            f = parametric_to_step(best_params, n_c)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Phase 2: Spectral coordinate descent (key addition)
    # Budget: 25% of time
    spectral_cd_end = start_time + (TIME_LIMIT - 5) * 0.37
    if time.time() < spectral_cd_end:
        cd_f, cd_score = spectral_coordinate_descent(
            best_f, start_time, spectral_cd_end
        )
        if cd_score < best_score:
            best_f = cd_f
            best_score = cd_score
    
    # Phase 3: Parametric spectral model
    # Budget: 15% of time
    spectral_model_end = start_time + (TIME_LIMIT - 5) * 0.52
    if time.time() < spectral_model_end:
        sm_f, sm_score = spectral_refinement(
            best_f, start_time, spectral_model_end
        )
        if sm_score < best_score:
            best_f = sm_f
            best_score = sm_score
    
    # Phase 4: Block-level simulated annealing
    # Budget: 28% of time
    sa_end = start_time + (TIME_LIMIT - 5) * 0.80
    if time.time() < sa_end:
        sa_f, sa_score = block_simulated_annealing(
            best_f, start_time, sa_end
        )
        if sa_score < best_score:
            best_f = sa_f
            best_score = sa_score
    
    # Phase 5: Final local polish
    # Budget: remaining time
    if time.time() < TIME_LIMIT - 2:
        polished_f, polished_score = local_polish(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
