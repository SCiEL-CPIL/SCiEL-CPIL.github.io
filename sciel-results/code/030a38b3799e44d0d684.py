"""Minimize the normalized maximum of the autoconvolution of a nonnegative
step function on 600 equal intervals over [-1/4, 1/4].

Strategy: Direct iterative optimization of the 600 heights. Uses a custom
algorithm that iteratively reduces the peak of the autoconvolution by
redistributing mass from locations that contribute to the peak. Also tries
several structured initializations (uniform, cosine-modulated, etc.) and
polishes with projected subgradient descent.
"""

import time
import math

import numpy as np

N = 600
DELTA = 0.5 / N
DEADLINE = 145.0  # safety margin before 150s kill

_t0 = time.time()


def _time_left():
    return DEADLINE - (time.time() - _t0)


def _midpoints():
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA / 2.0
    return x


def _autoconv_fft(h):
    """Compute full autoconvolution via FFT for speed."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    H = np.fft.rfft(h, size)
    C = np.fft.irfft(H * H, size)
    return C[:2 * n - 1]


def _evaluate_step_heights(h):
    """Compute the evaluator's score: max(autoconv) / (sum h)^2."""
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0.0, None)
    s = float(np.sum(h))
    if s < 1e-12:
        return float('inf')
    
    # Compute autoconvolution via FFT
    c = _autoconv_fft(h)
    peak = float(np.max(c))
    
    # L1-based normalization: peak / (sum h)^2
    return peak / (s * s)


def _iterative_peak_reduction(h, max_iters=200, step_scale=0.05):
    """Iteratively reduce the autoconvolution peak by redistributing mass.
    
    At each iteration:
    1. Compute autoconvolution c
    2. Find the peak location k*
    3. The gradient of c(k*) w.r.t. h(i) is h(k*-i) for valid i
    4. Reduce h at positions that contribute most to the peak
    5. Add mass back at positions that don't contribute as much
    """
    h = np.array(h, dtype=float)
    h = np.clip(h, 0.0, None)
    s0 = float(np.sum(h))
    if s0 < 1e-12:
        return h
    
    best_h = h.copy()
    best_score = _evaluate_step_heights(h)
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        c = _autoconv_fft(h)
        peak_idx = int(np.argmax(c))
        peak_val = c[peak_idx]
        
        if peak_val < 1e-10:
            break
        
        # Gradient of c(peak_idx) w.r.t. h(i) is h(peak_idx - i) for valid indices
        # c(k) = sum_i h(i) * h(k-i)
        # d c(k*) / d h(i) = h(k* - i) + h(i - k*) [but for autoconv, it's 2*h(k*-i) if symmetric]
        # Actually for autoconv: c(k) = sum_i h(i)h(k-i), so d c(k*)/d h(i) = h(k*-i) + h(i-k*)
        # But since we're dealing with discrete and k* can be negative offset, let's be careful
        
        # For the full autoconvolution indexed from -(N-1) to (N-1):
        # c_full[m] for m in [-(N-1), N-1]
        # Our c array is indexed 0 to 2N-2, where c[j] corresponds to lag j-(N-1)
        
        lag = peak_idx - (N - 1)
        
        # Gradient contribution: for each i, d c(lag) / d h[i] = h[(i - lag) mod N] approximately
        # More precisely: c(lag) = sum_i h[i] * h[i - lag] (with appropriate bounds)
        
        # Let's compute the gradient numerically or analytically
        # d c(lag) / d h[i] = h[i - lag] + h[i + lag] (for the two terms in the sum)
        # Wait, c(lag) = sum_{i} h[i] * h[i + lag] where i ranges so both indices valid
        
        # Let's just compute it directly
        grad = np.zeros(N)
        for i in range(N):
            # c(lag) includes term h[i]*h[i+lag] if i+lag in [0,N)
            # and term h[i+lag]*h[i] which is the same
            # Actually c(lag) = sum_{i: 0<=i<N, 0<=i+lag<N} h[i]*h[i+lag]
            # If lag >= 0: i from 0 to N-1-lag
            # If lag < 0: i from -lag to N-1
            
            if lag >= 0:
                if i < N - lag:
                    # h[i]*h[i+lag], derivative w.r.t h[i] is h[i+lag]
                    # Also h[i+lag]*h[i], derivative w.r.t h[i] is h[i+lag] (same term counted once in sum)
                    # Actually in the sum, each pair appears once, so d/dh[i] of h[i]*h[i+lag] = h[i+lag]
                    grad[i] += h[i + lag]
            else:
                # lag < 0, let lag2 = -lag
                lag2 = -lag
                if i >= lag2 and i < N:
                    # h[i]*h[i+lag] = h[i]*h[i-lag2]
                    if i - lag2 >= 0:
                        grad[i] += h[i - lag2]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Step: reduce h where gradient is positive (contributing to peak)
        # Add mass where gradient is negative or zero
        # Use a simple approach: h_new = h - step * grad, then project to nonnegative and renormalize
        
        step = step_scale * s0 / (grad_norm * N)  # scale appropriately
        
        h_new = h - step * grad
        h_new = np.clip(h_new, 0.0, None)
        
        # Renormalize to preserve sum
        s_new = float(np.sum(h_new))
        if s_new > 1e-12:
            h_new = h_new * (s0 / s_new)
        else:
            break
        
        score_new = _evaluate_step_heights(h_new)
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
            h = h_new.copy()
        else:
            # Reduce step size
            step_scale *= 0.7
            if step_scale < 1e-8:
                break
    
    return best_h


def _structured_initializations(x_mid):
    """Generate several structured initializations."""
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Cosine modulated: h(x) = 1 + a*cos(2*pi*x/0.5)
    for a in [0.2, 0.4, 0.6, -0.2, -0.4, -0.6]:
        h = 1.0 + a * np.cos(2 * np.pi * x_mid / 0.5)
        h = np.clip(h, 0.0, None)
        inits.append(h)
    
    # Second harmonic
    for a in [0.2, 0.4, -0.2, -0.4]:
        h = 1.0 + a * np.cos(4 * np.pi * x_mid / 0.5)
        h = np.clip(h, 0.0, None)
        inits.append(h)
    
    # Triangular: h(x) = 1 - 2*|x|/0.5 = 1 - 4|2x|... wait, on [-1/4, 1/4]
    # h(x) = 1 - 4*|x| gives 1 at x=0, 0 at x=+-1/4
    h_tri = 1.0 - 4.0 * np.abs(x_mid)
    h_tri = np.clip(h_tri, 0.0, None)
    inits.append(h_tri)
    
    # Inverse triangular: high at edges
    h_inv_tri = 4.0 * np.abs(x_mid)
    h_inv_tri = np.clip(h_inv_tri, 0.0, None)
    inits.append(h_inv_tri)
    
    # Bump: h(x) = exp(-x^2/sigma^2) for various sigma
    for sigma in [0.05, 0.1, 0.15]:
        h_bump = np.exp(-x_mid**2 / (2 * sigma**2))
        inits.append(h_bump)
    
    # Flat-top with rounded edges
    for w in [0.3, 0.5, 0.7]:
        h_flat = np.ones(N)
        mask = np.abs(x_mid) > w * 0.25
        h_flat[mask] = 0.5
        inits.append(h_flat)
    
    return inits


def construct():
    global _t0
    _t0 = time.time()
    
    x_mid = _midpoints()
    
    # Try all structured initializations
    inits = _structured_initializations(x_mid)
    
    best_h = inits[0].copy()
    best_score = _evaluate_step_heights(best_h)
    
    for init in inits:
        if _time_left() < 10.0:
            break
        
        score = _evaluate_step_heights(init)
        if score < best_score:
            best_score = score
            best_h = init.copy()
    
    # Iterative peak reduction from the best initialization
    if _time_left() > 15.0:
        h_improved = _iterative_peak_reduction(best_h, max_iters=300, step_scale=0.1)
        score_improved = _evaluate_step_heights(h_improved)
        if score_improved < best_score:
            best_score = score_improved
            best_h = h_improved.copy()
    
    # Try iterative reduction from several other initializations
    if _time_left() > 10.0:
        for init in inits[1:4]:  # try a few more
            if _time_left() < 5.0:
                break
            h_try = _iterative_peak_reduction(init, max_iters=100, step_scale=0.05)
            score_try = _evaluate_step_heights(h_try)
            if score_try < best_score:
                best_score = score_try
                best_h = h_try.copy()
    
    # Final: try a few more iterations on the best
    if _time_left() > 5.0:
        h_final = _iterative_peak_reduction(best_h, max_iters=100, step_scale=0.02)
        score_final = _evaluate_step_heights(h_final)
        if score_final < best_score:
            best_score = score_final
            best_h = h_final.copy()
    
    # Ensure nonnegative
    best_h = np.clip(best_h, 0.0, None)
    
    return [float(v) for v in best_h]


_t0 = time.time()
