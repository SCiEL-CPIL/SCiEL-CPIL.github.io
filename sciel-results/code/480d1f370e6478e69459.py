"""Rewritten program for third autocorrelation inequality optimization."""
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

def _objective_energy_peak(c_ac, c_dc, energy_weight=0.9, beta=50.0):
    """
    Combined objective:
    1. Minimize L2 energy of the non-DC autocorrelation (spectral flatness).
    2. Minimize the soft-max of the autocorrelation peak (direct target).
    
    This structural change moves from 'chasing the peak' to 'flattening the spectrum',
    which is a more robust geometric property for reducing the max/mean ratio.
    """
    c_full = np.concatenate([[c_dc], c_ac])
    h_recon = _dct_to_h(c_full)
    
    if not np.all(np.isfinite(h_recon)):
        return 1e10
    
    s = np.sum(h_recon)
    if abs(s) < 1e-10:
        return 1e10
    
    # Normalize to have sum = N (so DC component is fixed, we optimize AC)
    # Note: The objective function in the task is scale-invariant if we normalize h.
    # value = abs(2 * N * max(conv(h)) / sum(h)^2)
    # If we scale h by k, conv scales by k^2, sum^2 scales by k^2. Ratio is invariant.
    # So we can fix sum(h) = N to simplify the optimization landscape.
    h_norm = h_recon * (N / s)
    
    conv = _convolve_fft(h_norm)
    
    # The DC term of the convolution is sum(h)^2 = N^2.
    # The "sidelobes" are conv[1:] and conv[-1] (which is conv[0] mirrored? No, full conv is symmetric).
    # conv has length 2N-1. conv[N-1] is the peak at lag 0.
    # We want to minimize the energy of the sidelobes.
    # Energy = sum(conv^2) - conv[N-1]^2 (if we want to exclude the main lobe)
    # Actually, the task cares about max(conv).
    # Minimizing L2 energy of the WHOLE conv (minus DC) forces the spectrum to be flat.
    # Let's compute the energy of the non-zero lags.
    # Lags 0 to N-1 are in conv[0:N]. Lag 0 is conv[N-1].
    # Wait, np.convolve full mode:
    # h = [h0, h1]
    # conv = [h0^2, h0h1+h1h0, h1^2]
    # Lag 0 is the middle.
    # For length N, lag 0 is index N-1.
    
    # Let's compute the sum of squares of the convolution, excluding the lag 0 term.
    # This is the total energy of the autocorrelation sidelobes.
    # By Parseval, sum(conv^2) = sum(|FFT(h)|^4) / N_fft ? No.
    # sum(conv[k]^2) = (1/2pi) integral |H(f)|^4 df.
    # Minimizing this flattens the power spectrum |H(f)|^2.
    
    # Exclude the center peak (lag 0) to focus on sidelobes.
    # conv is symmetric. Center is at N-1.
    sidelobe_energy = np.sum(conv**2) - conv[N-1]**2
    
    # Also keep a soft-max term to directly guide the peak down.
    sm = _soft_max(conv, beta)
    peak_term = abs(2.0 * N * sm / (N * N))
    
    # Normalize energy term to be comparable to peak_term (which is ~2.0 for seed)
    # Energy of seed (ones): conv is triangle. 
    # For N=400, conv[0]=1, conv[399]=400... 
    # Let's just scale it empirically.
    # Actually, let's just minimize the energy term primarily.
    
    obj = energy_weight * (sidelobe_energy / (N**4)) + (1.0 - energy_weight) * peak_term
    return obj

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
    Structural Change: Optimizes in DCT domain using Spectral Energy Flattening (L2 norm of autocorrelation sidelobes)
    combined with a soft-max peak penalty.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s limit

    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0

    # Phase 1: Quick parametric families to get a good starting point
    # Family 1: h[i] = 1 + a*cos(2*pi*m*i/N)
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
    
    # Phase 3: DCT-domain optimization with Spectral Energy Flattening
    c_best = _h_to_dct(best_h)
    c_dc = c_best[0].copy()

    # Phase 3a: Spectral Flattening - Minimize L2 Energy of Autocorrelation Sidelobes
    # This is the structural change: instead of just minimizing the peak, we flatten the spectrum.
    if time.time() < deadline - 50:
        c_dc_flat = c_best[0].copy()
        
        # We vary the weight of the energy term vs the peak term
        # Higher weight on energy -> flatter spectrum -> potentially lower peak
        for energy_weight in [0.95, 0.9, 0.8, 0.7]:
            if time.time() > deadline - 50:
                break
            
            def obj_flat(c_ac, ew=energy_weight, c_dc=c_dc_flat):
                return _objective_energy_peak(c_ac, c_dc, energy_weight=ew, beta=50.0)
            
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
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = _h_to_dct(best_h)
                        c_dc = c_best[0].copy()
                        c_dc_flat = c_best[0].copy()
            except Exception:
                continue

    # Phase 3b: Soft-max peak minimization in DCT domain (Refinement)
    # Now that we have a flatter spectrum, we can polish the peak directly.
    for beta in [10.0, 50.0, 100.0]:
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

        # Try random restarts around best DCT coefficients
        rng = np.random.RandomState(42 + int(beta))
        c_ac_init = c_best[1:].copy()

        for trial in range(4):
            if time.time() > deadline - 25:
                break

            if trial == 0:
                c_ac_trial = c_ac_init.copy()
            else:
                c_ac_trial = c_ac_init + 0.1 * rng.randn(N-1)

            try:
                result = minimize(
                    obj_dct,
                    c_ac_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
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
        for beta in [200.0, 500.0]:
            if time.time() > deadline - 15:
                break

            def obj_polish(c_ac, beta=beta, c_dc=c_dc):
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
                    obj_polish,
                    c_best[1:].copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
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

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
