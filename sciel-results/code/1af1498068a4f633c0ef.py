"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _soft_max_grad(x, beta):
    """Soft maximum and its gradient."""
    m = np.max(x)
    if not np.isfinite(m):
        return m, np.zeros_like(x)
    exp_vals = np.exp(beta * (x - m))
    w = exp_vals / np.sum(exp_vals)
    sm = m + np.log(np.sum(exp_vals)) / beta
    grad = beta * w
    return sm, grad

def _objective_soft(h, beta):
    """
    Soft-max surrogate objective for autoconvolution peak.
    h is normalized internally so sum(h) = N.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def _objective_exact(h):
    """Exact objective using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses a random walk with negative drift to enforce zero-sum constraint,
    then optimizes the drift parameter and noise realization to minimize
    the peak of the zero-mean autocorrelation.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit

    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0

    # Phase 1: Quick parametric families to get a good starting point
    # Family 1: h[i] = 1 + a*cos(2*pi*m*i/N)
    for m in range(1, 21):
        if time.time() > deadline - 80:
            break
        for a in np.linspace(-0.8, 0.8, 17):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Family 2: Smoothed block
    for beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 80:
            break
        h = np.tanh(beta * (np.arange(N) - N / 2.0))
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        v = _eval_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Phase 2: Random walk with negative drift
    # h[i] = 1 + cumsum(noise) - drift * i
    # The drift is chosen so that sum(h) = N (zero-mean perturbation)
    # We optimize over noise realizations and drift to minimize the peak autocorrelation
    
    if time.time() < deadline - 60:
        rng = np.random.RandomState(42)
        
        # Try different noise scales and drift values
        for noise_scale in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            if time.time() > deadline - 60:
                break
            
            for trial in range(30):
                if time.time() > deadline - 60:
                    break
                
                # Generate random noise with occasional spikes
                noise = rng.randn(N) * noise_scale
                if trial % 5 == 0:
                    # Add a few spike perturbations
                    n_spikes = rng.randint(2, 6)
                    spike_idx = rng.choice(N, size=n_spikes, replace=False)
                    noise[spike_idx] += rng.choice([-1, 1], size=n_spikes) * noise_scale * 2
                
                # Compute cumulative sum (random walk)
                walk = np.cumsum(noise)
                
                # Subtract linear trend to make it zero-sum
                # We want sum(h) = N, so sum(walk - drift * i) = 0
                # sum(walk) - drift * sum(i) = 0
                # drift = sum(walk) / sum(i)
                sum_i = np.sum(np.arange(N))
                drift = np.sum(walk) / sum_i
                
                # Construct h
                h = 1.0 + walk - drift * np.arange(N)
                
                # Check validity
                if not np.all(np.isfinite(h)) or abs(np.sum(h) - N) > 1e-6:
                    continue
                
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Phase 3: DCT-domain optimization with soft-max surrogate
    # Convert best_h to DCT coefficients
    from scipy.fft import dct, idct
    
    def _h_to_dct(h):
        """Convert time-domain signal to DCT-II coefficients."""
        return dct(h, type=2, norm='ortho')
    
    def _dct_to_h(coeffs):
        """Convert DCT-II coefficients to time-domain signal."""
        return idct(coeffs, type=2, norm='ortho')
    
    c_best = _h_to_dct(best_h)

    # Fix DC component to maintain mean
    c_dc = c_best[0].copy()

    # Two-stage soft-max: low beta first, then high beta
    for beta in [1.0, 5.0, 20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - 25:
            break

        # Objective in DCT domain using soft-max
        def obj_dct(c_ac, beta=beta):
            c_full = np.concatenate([[c_dc], c_ac])
            h_recon = _dct_to_h(c_full)

            if not np.all(np.isfinite(h_recon)):
                return 1e10

            s = np.sum(h_recon)
            if abs(s) < 1e-10:
                return 1e10

            h_norm = h_recon * (N / s)
            conv = _convolve_fft(h_norm)
            sm = _soft_max(conv, beta)
            return abs(2.0 * N * sm / (N * N))

        # Try random restarts around best DCT coefficients
        rng = np.random.RandomState(42 + int(beta))
        c_ac_init = c_best[1:].copy()

        for trial in range(6):
            if time.time() > deadline - 25:
                break

            if trial == 0:
                c_ac_trial = c_ac_init.copy()
            elif trial == 1:
                c_ac_trial = c_ac_init + 0.05 * rng.randn(N-1)
            elif trial == 2:
                c_ac_trial = c_ac_init + 0.2 * rng.randn(N-1)
            elif trial == 3:
                # Perturb only low-frequency components
                c_ac_trial = c_ac_init.copy()
                n_low = N // 4
                c_ac_trial[:n_low] += 0.1 * rng.randn(n_low)
            elif trial == 4:
                # Perturb only high-frequency components
                c_ac_trial = c_ac_init.copy()
                n_high = N // 2
                c_ac_trial[n_high:] += 0.05 * rng.randn(N-1-n_high)
            else:
                c_ac_trial = c_ac_init + 0.15 * rng.randn(N-1)

            try:
                result = minimize(
                    obj_dct,
                    c_ac_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )

                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)

                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = _h_to_dct(best_h)
                        c_dc = c_best[0].copy()
                        c_ac_init = c_best[1:].copy()
            except Exception:
                continue

    # Phase 4: High-precision polish with very high beta in DCT domain
    if time.time() < deadline - 20:
        for beta in [200.0, 500.0, 1000.0, 2000.0]:
            if time.time() > deadline - 15:
                break

            def obj_polish(c_ac, beta=beta):
                c_full = np.concatenate([[c_dc], c_ac])
                h_recon = _dct_to_h(c_full)

                if not np.all(np.isfinite(h_recon)):
                    return 1e10

                s = np.sum(h_recon)
                if abs(s) < 1e-10:
                    return 1e10

                h_norm = h_recon * (N / s)
                conv = _convolve_fft(h_norm)
                sm = _soft_max(conv, beta)
                return abs(2.0 * N * sm / (N * N))

            # Try multiple starting points for polish
            rng_polish = np.random.RandomState(123 + int(beta))
            for trial in range(3):
                if time.time() > deadline - 15:
                    break
                
                if trial == 0:
                    c_ac_start = c_best[1:].copy()
                else:
                    c_ac_start = c_best[1:] + 0.01 * rng_polish.randn(N-1)

                try:
                    result = minimize(
                        obj_polish,
                        c_ac_start,
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                    )

                    c_ac_opt = result.x
                    c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                    h_opt = _dct_to_h(c_full_opt)

                    if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                        v_opt = _objective_exact(h_opt)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_opt.copy()
                            c_best = _h_to_dct(best_h)
                            c_dc = c_best[0].copy()
                except Exception:
                    continue

    # Phase 5: Final high-precision polish with coordinate descent
    if time.time() < deadline - 5:
        # First try DCT optimization with very tight tolerances
        for beta in [3000.0]:
            if time.time() > deadline - 3:
                break
            def obj_final(c_ac, beta=beta):
                c_full = np.concatenate([[c_dc], c_ac])
                h_recon = _dct_to_h(c_full)
                if not np.all(np.isfinite(h_recon)):
                    return 1e10
                s = np.sum(h_recon)
                if abs(s) < 1e-10:
                    return 1e10
                h_norm = h_recon * (N / s)
                conv = _convolve_fft(h_norm)
                sm = _soft_max(conv, beta)
                return abs(2.0 * N * sm / (N * N))
            
            try:
                result = minimize(
                    obj_final,
                    c_best[1:].copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
        
        # Coordinate descent polish: optimize individual h[i] values
        if time.time() < deadline - 2:
            h_cd = best_h.copy()
            rng_cd = np.random.RandomState(999)
            
            # Random order for coordinate descent
            order = rng_cd.permutation(N)
            n_cd_steps = min(500, int((deadline - 2 - time.time()) * 50))
            
            for step in range(n_cd_steps):
                if time.time() > deadline - 2:
                    break
                
                i = order[step % N]
                
                # Try small perturbations of h[i]
                base_val = h_cd[i]
                best_i_val = base_val
                best_i_obj = _eval_value(h_cd)
                
                for delta in [-0.01, -0.005, 0.005, 0.01, -0.02, 0.02]:
                    h_trial = h_cd.copy()
                    h_trial[i] = base_val + delta
                    
                    if not np.all(np.isfinite(h_trial)) or abs(np.sum(h_trial)) < 1e-10:
                        continue
                    
                    v_trial = _eval_value(h_trial)
                    if v_trial < best_i_obj:
                        best_i_obj = v_trial
                        best_i_val = base_val + delta
                
                h_cd[i] = best_i_val
            
            if _eval_value(h_cd) < best_val:
                best_val = _eval_value(h_cd)
                best_h = h_cd.copy()

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
