"""Spectral peak minimization for third autocorrelation inequality."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft, rfft, irfft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    n_conv = 2 * N - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
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

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses spectral peak minimization in the frequency domain with projected
    gradient descent on the power spectrum, combined with DCT-domain polish.
    """
    deadline = time.time() + 145.0

    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0

    # Phase 1: Quick parametric families to get a good starting point
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

    # Phase 2: Spectral peak minimization via projected gradient on power spectrum
    # The autoconvolution c = ifft(|H|^2) (up to indexing)
    # We want to minimize max(c) subject to sum(h) = N (DC fixed)
    # In frequency domain: H[0] = N (fixed), minimize peak of ifft(|H|^2)
    
    if time.time() < deadline - 60:
        # Work with real FFT since h is real
        # h has N elements, use rfft of length 2N-1 for linear convolution
        n_conv = 2 * N - 1
        n_fft = 1
        while n_fft < n_conv:
            n_fft *= 2
        
        # Initialize from best_h
        H_init = fft(best_h, n=n_fft)
        # Power spectrum: |H|^2, but we work with the magnitude squared
        # Constraint: H[0] = N (real, since sum(h) = N)
        
        # We'll optimize the phase and magnitude of H[1:] and H[-1:-1-//2:-1]
        # Actually, let's work with the full complex H but fix H[0]
        
        # Projected gradient descent on H (complex)
        # Objective: soft_max(ifft(|H|^2)) with H[0] fixed
        
        def spectral_obj_and_grad(H, beta=50.0):
            """
            Objective: soft_max of autoconvolution, gradient w.r.t. H
            H[0] is fixed, so gradient[0] = 0
            """
            # Compute autoconvolution from H
            conv = ifft(H * H, n=n_fft).real[:n_conv]
            sm, grad_conv = _soft_max_grad(conv, beta)
            
            # Gradient of conv w.r.t. H
            # conv = ifft(H * H), so d conv / d H = ifft of something?
            # Actually, let's use: conv_k = sum_i h_i * h_{k-i}
            # In frequency: C = H * H (element-wise), conv = ifft(C)
            # d conv / d H_j = ifft( (dC/dH_j) ) where dC/dH_j has H at position j
            # This is complex; let's use a simpler approach:
            # grad of soft_max(conv) w.r.t. h, then transform to H
            # Actually, let's just do gradient in time domain and FFT
            
            # Gradient of soft_max(conv) w.r.t. h:
            # conv = convolve(h, h), d conv_k / d h_i = h_{k-i} (with bounds)
            # grad_h = convolve(grad_conv, h) + convolve(h, grad_conv) (approximately)
            # More precisely: d conv / d h_i = sum_k grad_conv_k * h_{k-i}
            # This is convolve(grad_conv, h) but with appropriate indexing
            
            # For full convolution: conv[k] = sum_i h[i] * h[k-i]
            # d conv[k] / d h[i] = h[k-i]
            # So grad_h[i] = sum_k grad_conv[k] * h[k-i] = (grad_conv * h)[i] (cross-correlation)
            # Actually it's: grad_h = np.convolve(grad_conv, h, mode='same') + ... 
            # Let me think more carefully.
            
            # conv = np.convolve(h, h, mode='full'), length 2N-1
            # d conv[k] / d h[i] = h[k-i] if 0 <= k-i < N, else 0
            # grad_h[i] = sum_k grad_conv[k] * h[k-i]
            # This is the cross-correlation of grad_conv and h, evaluated at lag i
            
            # Use FFT for efficiency:
            # grad_h = ifft(fft(grad_conv, n=n_fft) * fft(h, n=n_fft)) but this gives circular
            # Let's just do it directly for clarity, N=400 is small
            
            # Actually, let's use the fact that:
            # grad_h = np.correlate(grad_conv, h, mode='full') but with proper indexing
            # Hmm, this is getting complicated. Let me use a different approach.
            
            # Alternative: use the chain rule through the FFT
            # C = H * H (element-wise product in freq domain)
            # conv = ifft(C)
            # d obj / d H = fft(grad_conv) * H (by chain rule, since C = H*H)
            # Wait, d C / d H_j = H at position j (and 0 elsewhere), so
            # d obj / d H_j = sum_k grad_conv_k * d conv_k / d C_k * d C_k / d H_j
            # d conv / d C = fft(grad_conv) (since conv = ifft(C), so d conv_k / d C_m = delta_{k,m})
            # Actually conv = ifft(C), so d conv / d C_k = 1 for all k (in the sense that C_k contributes directly)
            # No: conv_k = (1/n) sum_m C_m exp(2pi i k m / n), so d conv_k / d C_m = (1/n) exp(2pi i k m / n)
            # So d obj / d C_m = sum_k grad_conv_k * (1/n) exp(2pi i k m / n) = (1/n) fft(grad_conv)[m]
            # And d C_m / d H_j = H_m if j == m, else 0 (since C_m = H_m^2, dC_m/dH_m = 2*H_m)
            # Wait, C_m = H_m * H_m, so dC_m/dH_m = 2*H_m
            # So d obj / d H_m = d obj / d C_m * d C_m / d H_m = (1/n) fft(grad_conv)[m] * 2 * H_m
            
            # But we also need to account for the fact that h is real, so H is conjugate symmetric
            # For now, let's just compute the gradient in time domain
            
            # Simpler: compute grad_h directly
            # grad_h[i] = sum_k grad_conv[k] * h[k-i] where the sum is over k such that 0 <= k-i < N
            # This is: for each i, sum over k from i to i+N-1 of grad_conv[k] * h[k-i]
            # = sum_j grad_conv[i+j] * h[j] for j=0..N-1, with grad_conv padded
            
            # Use FFT:
            # grad_h = ifft(fft(grad_conv, n=n_fft) * np.conj(fft(h, n=n_fft))) ... no
            # Let's just do it with np.convolve
            # grad_conv has length 2N-1, h has length N
            # We want: for i in 0..N-1, grad_h[i] = sum_{j=0}^{N-1} grad_conv[i+j] * h[j]
            # This is like a cross-correlation
            
            # Pad grad_conv to length n_fft
            gc_padded = np.zeros(n_fft)
            gc_padded[:len(grad_conv)] = grad_conv
            h_padded = np.zeros(n_fft)
            h_padded[:N] = h
            
            # The cross-correlation can be computed as:
            # grad_h[i] = sum_j gc[i+j] * h[j] = (gc * h)[i] in the sense of correlation
            # Using FFT: correlation = ifft(fft(gc) * conj(fft(h)))
            # But this gives circular correlation. For linear, we need to be careful.
            
            # Actually, let's just use the direct approach since N=400:
            grad_h = np.zeros(N)
            for i in range(N):
                # grad_h[i] = sum_j grad_conv[i+j] * h[j] for j=0..N-1
                # i+j ranges from i to i+N-1, which is within 0..2N-2
                grad_h[i] = np.dot(grad_conv[i:i+N], h)
            
            # Now transform to frequency domain
            # The gradient w.r.t. H is related to grad_h by:
            # h = ifft(H), so d h / d H = ifft of identity
            # d obj / d H = fft(grad_h) (up to scaling)
            # Actually, obj depends on h, h = ifft(H), so
            # d obj / d H_k = sum_i grad_h[i] * d h_i / d H_k
            # h_i = (1/n) sum_k H_k exp(2pi i i k / n)
            # d h_i / d H_k = (1/n) exp(2pi i i k / n)
            # d obj / d H_k = sum_i grad_h[i] * (1/n) exp(2pi i i k / n) = (1/n) fft(grad_h)[k]
            
            grad_H = fft(grad_h, n=n_fft) / n_fft
            
            # Fix H[0]
            grad_H[0] = 0
            
            return sm, grad_H, conv
        
        # Projected gradient descent on H
        # Constraint: H[0] = N (real), H is conjugate symmetric (for real h)
        # We'll work with the first half + DC + Nyquist
        
        # Initialize
        H = fft(best_h, n=n_fft).copy()
        H[0] = float(N)  # Fix DC
        
        # Ensure conjugate symmetry
        for k in range(1, n_fft // 2 + 1):
            if k < n_fft - k:
                H[n_fft - k] = np.conj(H[k])
        
        best_spectral_val = best_val
        best_spectral_h = best_h.copy()
        
        # Projected gradient descent
        lr = 0.01
        beta = 20.0
        max_iters = 500
        patience = 50
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 60:
                break
            
            sm, grad_H, conv = spectral_obj_and_grad(H, beta)
            
            # Gradient step
            H_new = H - lr * grad_H
            
            # Project: fix H[0] = N
            H_new[0] = float(N)
            
            # Enforce conjugate symmetry
            for k in range(1, n_fft // 2 + 1):
                if k < n_fft - k:
                    H_new[n_fft - k] = np.conj(H_new[k])
            
            # Convert back to time domain
            h_new = ifft(H_new, n=n_fft).real[:N]
            
            # Check validity
            if not np.all(np.isfinite(h_new)) or abs(np.sum(h_new)) < 1e-10:
                H = H_new  # Keep H even if h is invalid, try next step
                continue
            
            v_new = _eval_value(h_new)
            
            if v_new < best_spectral_val:
                best_spectral_val = v_new
                best_spectral_h = h_new.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            H = H_new
            
            # Adaptive learning rate
            if no_improve > patience:
                lr *= 0.5
                no_improve = 0
                if lr < 1e-8:
                    break
        
        if best_spectral_val < best_val:
            best_val = best_spectral_val
            best_h = best_spectral_h.copy()

    # Phase 3: DCT-domain optimization with soft-max surrogate (polish)
    from scipy.fft import dct, idct
    
    def _h_to_dct(h):
        return dct(h, type=2, norm='ortho')
    
    def _dct_to_h(coeffs):
        return idct(coeffs, type=2, norm='ortho')
    
    c_best = _h_to_dct(best_h)
    c_dc = c_best[0].copy()
    
    for beta in [10.0, 50.0, 200.0]:
        if time.time() > deadline - 25:
            break

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

        rng = np.random.RandomState(42 + int(beta))
        c_ac_init = c_best[1:].copy()

        for trial in range(3):
            if time.time() > deadline - 25:
                break

            if trial == 0:
                c_ac_trial = c_ac_init.copy()
            else:
                c_ac_trial = c_ac_init + 0.05 * rng.randn(N-1)

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
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = _h_to_dct(best_h)
                        c_dc = c_best[0].copy()
                        c_ac_init = c_best[1:].copy()
            except Exception:
                continue

    # Phase 4: Final polish with very high beta
    if time.time() < deadline - 10:
        beta = 500.0
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
        
        try:
            result = minimize(
                obj_polish,
                c_best[1:].copy(),
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c_ac_opt = result.x
            c_full_opt = np.concatenate([[c_dc], c_ac_opt])
            h_opt = _dct_to_h(c_full_opt)
            if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                v_opt = _eval_value(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
