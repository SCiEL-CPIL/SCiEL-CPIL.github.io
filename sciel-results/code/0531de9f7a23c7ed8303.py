"""Combined program: spectral flattening + simulated annealing for autocorrelation optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft, dct, idct
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

def _h_to_dct(h):
    """Convert time-domain signal to DCT-II coefficients."""
    return dct(h, type=2, norm='ortho')

def _dct_to_h(coeffs):
    """Convert DCT-II coefficients to time-domain signal."""
    return idct(coeffs, type=2, norm='ortho')

def _spectral_flatness_obj(c_ac, c_dc, weight=0.5, beta=50.0):
    """
    Objective combining spectral flatness (L4 norm of DCT coeffs) 
    with a soft-max term to guide toward low peak.
    """
    c_full = np.concatenate([[c_dc], c_ac])
    h_recon = _dct_to_h(c_full)
    
    if not np.all(np.isfinite(h_recon)):
        return 1e10
    
    s = np.sum(h_recon)
    if abs(s) < 1e-10:
        return 1e10
    
    h_norm = h_recon * (N / s)
    c_norm = _h_to_dct(h_norm)
    
    l4_energy = np.sum(np.abs(c_norm[1:])**4)
    
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    peak_term = abs(2.0 * N * sm / (N * N))
    
    return weight * l4_energy / N**4 + (1.0 - weight) * peak_term

def _simulated_annealing(h_init, deadline):
    """
    Simulated annealing to minimize max autoconvolution.
    Uses binary-like perturbations to explore Golay-like sequences.
    """
    best_h = h_init.copy()
    best_val = _eval_value(h_init)
    current_h = h_init.copy()
    current_val = best_val
    
    # Initial temperature
    T_start = 0.5
    T_end = 0.001
    total_iters = 2000
    
    for it in range(total_iters):
        if time.time() > deadline:
            break
        
        # Cool temperature
        T = T_start * (T_end / T_start) ** (it / total_iters)
        
        # Generate perturbation: flip a random subset of elements
        # This creates binary-like changes
        n_flip = np.random.randint(1, 5)
        indices = np.random.choice(N, size=n_flip, replace=False)
        
        h_trial = current_h.copy()
        for idx in indices:
            # Random sign flip or magnitude change
            if np.random.random() < 0.5:
                h_trial[idx] = -h_trial[idx]
            else:
                h_trial[idx] += np.random.randn() * 0.1
        
        # Check validity
        if not np.all(np.isfinite(h_trial)) or abs(np.sum(h_trial)) < 1e-10:
            continue
        
        v_trial = _eval_value(h_trial)
        
        # Accept if better or with probability exp(-delta/T)
        delta = v_trial - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_h = h_trial
            current_val = v_trial
            
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
    
    return best_h, best_val

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Combines spectral flattening, DCT optimization, and simulated annealing.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s limit

    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0

    # Phase 1: Quick parametric families to get a good starting point
    for m in range(1, 21):
        if time.time() > deadline - 90:
            break
        for a in np.linspace(-0.8, 0.8, 17):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Family 2: Smoothed block
    for beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 90:
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
    if time.time() < deadline - 70:
        rng = np.random.RandomState(42)
        
        for noise_scale in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            if time.time() > deadline - 70:
                break
            
            for trial in range(20):
                if time.time() > deadline - 70:
                    break
                
                noise = rng.randn(N) * noise_scale
                walk = np.cumsum(noise)
                
                sum_i = np.sum(np.arange(N))
                drift = np.sum(walk) / sum_i
                
                h = 1.0 + walk - drift * np.arange(N)
                
                if not np.all(np.isfinite(h)) or abs(np.sum(h) - N) > 1e-6:
                    continue
                
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Phase 3: DCT-domain optimization with spectral flattening
    c_best = _h_to_dct(best_h)
    c_dc = c_best[0].copy()

    # Phase 3a: Spectral flattening - minimize L4 norm of DCT coefficients
    if time.time() < deadline - 50:
        c_dc_flat = c_best[0].copy()
        
        for weight in [0.8, 0.6, 0.4, 0.2]:
            if time.time() > deadline - 50:
                break
            
            def obj_flat(c_ac, weight=weight, c_dc=c_dc_flat):
                return _spectral_flatness_obj(c_ac, c_dc, weight=weight, beta=50.0)
            
            try:
                result = minimize(
                    obj_flat,
                    c_best[1:].copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc_flat], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)
                
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = _h_to_dct(best_h)
                        c_dc = c_best[0].copy()
                        c_dc_flat = c_best[0].copy()
            except Exception:
                continue

    # Phase 3b: Soft-max peak minimization in DCT domain
    for beta in [1.0, 5.0, 20.0, 50.0, 100.0]:
        if time.time() > deadline - 25:
            break

        def obj_dct(c_ac, beta=beta, c_dc=c_dc):
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
                obj_dct,
                c_best[1:].copy(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            c_ac_opt = result.x
            c_full_opt = np.concatenate([[c_dc], c_ac_opt])
            h_opt = _dct_to_h(c_full_opt)
            
            if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                v_opt = _eval_value(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
                    c_best = _h_to_dct(best_h)
                    c_dc = c_best[0].copy()
        except Exception:
            continue

    # Phase 4: Simulated annealing to find Golay-like sequences
    if time.time() < deadline - 10:
        np.random.seed(42)
        h_sa, v_sa = _simulated_annealing(best_h, deadline - 5)
        if v_sa < best_val:
            best_val = v_sa
            best_h = h_sa

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
