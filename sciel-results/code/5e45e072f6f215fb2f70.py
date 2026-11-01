"""Optimized program for third autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin before 150s kill

def _autoconv(h):
    """Compute autoconvolution of h using FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:n]
    return conv

def _objective_and_grad(x, logsumexp_beta):
    """
    x: continuous parameters, h = tanh(x)
    Returns objective value and gradient.
    Objective: max|conv| / (sum|h|)^2, smoothed with log-sum-exp.
    """
    h = np.tanh(x)
    abs_h = np.abs(h)
    S = np.sum(abs_h)
    if S < 1e-10:
        S = 1e-10
    
    conv = _autoconv(h)
    abs_conv = np.abs(conv)
    
    # Smooth max via log-sum-exp
    m = np.max(abs_conv)
    if m < 1e-15:
        LSE = m
    else:
        # log-sum-exp of abs_conv / m, then * m
        shifted = abs_conv / m
        LSE = m * np.log(np.sum(np.exp(logsumexp_beta * (shifted - np.max(shifted))))) / logsumexp_beta
        # Actually simpler: LSE = (1/beta) * log(sum(exp(beta * abs_conv)))
        # But for numerical stability:
        max_ac = np.max(abs_conv)
        if max_ac > 1e-15:
            LSE = max_ac + (1.0 / logsumexp_beta) * np.log(np.sum(np.exp(logsumexp_beta * (abs_conv - max_ac))))
        else:
            LSE = 0.0
    
    val = LSE / (S * S)
    
    # Gradient via finite differences would be slow; use analytic chain rule
    # d/dx_i [LSE / S^2]
    # LSE = max_ac + (1/beta) log(sum(exp(beta*(conv_j - max_ac))))
    # dLSE/dconv_j = exp(beta*(conv_j - max_ac)) / sum(exp(beta*(conv_k - max_ac)))
    # For the max element, this is ~1; for others, smaller.
    
    # dLSE/dh_i = sum_j (dLSE/dconv_j) * (dconv_j/dh_i)
    # conv_j = sum_k h_k * h_{k+j}
    # dconv_j/dh_i = h_{i-j} + h_{i+j} (with appropriate bounds)
    
    # This is complex; let's use a simpler approach: use the unsmoothed max with subgradient
    # Or use a moderate beta and finite difference for gradient (expensive)
    
    # Alternative: use the analytic gradient of the smooth surrogate
    # Let's compute it properly
    
    # weights for log-sum-exp
    if max_ac > 1e-15:
        exponents = logsumexp_beta * (abs_conv - max_ac)
        # Clip for numerical stability
        exponents = np.clip(exponents, -50, 50)
        weights = np.exp(exponents)
        weights /= np.sum(weights)
    else:
        weights = np.ones(N) / N
    
    # dLSE/dconv_j = weights_j * sign(conv_j)  (since abs_conv_j = |conv_j|)
    # Actually d|conv_j|/dconv_j = sign(conv_j)
    dLSE_dconv = weights * np.sign(conv)
    
    # dconv_j/dh_i: conv_j = sum_k h_k * h_{k+j}
    # For a given i, which conv_j does h_i contribute to?
    # conv_j = sum_k h_k h_{k+j}
    # h_i appears when k=i (h_i * h_{i+j}) or k+j=i i.e. k=i-j (h_{i-j} * h_i)
    # So dconv_j/dh_i = h_{i+j} + h_{i-j} (if in bounds)
    
    # We need dLSE/dh_i = sum_j dLSE_dconv_j * dconv_j/dh_i
    # = sum_j dLSE_dconv_j * (h_{i+j} + h_{i-j})
    # = sum_j dLSE_dconv_j * h_{i+j} + sum_j dLSE_dconv_j * h_{i-j}
    
    # Let w_j = dLSE_dconv_j
    # sum_j w_j * h_{i+j} = (w * h)[i] via cross-correlation
    # sum_j w_j * h_{i-j} = (w * h)[i] via auto-correlation... 
    
    # Actually: sum_j w_j h_{i+j} is the cross-correlation of w and h evaluated at i
    # sum_j w_j h_{i-j} is also a cross-correlation
    
    # Use FFT for efficiency
    size = 2 * N
    W = np.fft.rfft(w, size)
    H = np.fft.rfft(h, size)
    
    # cross_corr[i] = sum_j w_j * h_{i+j}
    # This is like (w flipped) * h
    W_flip = np.conj(W)  # for real signals, flipping in freq domain
    # Actually, cross-correlation: sum_j w_j h_{i+j}
    # = sum_j w_j h_{i+j}
    # Let's use: np.correlate(w, h, mode='full') but that's O(N^2)
    # FFT approach: 
    # sum_j w_j h_{i+j} = [w * h_rev][i + N - 1] where h_rev is reversed
    # Simpler: use ifft(fft(w) * conj(fft(h))) gives cross-correlation
    
    # cross_corr[i] = sum_j w_{i-j} h_j = sum_j w_j h_{i-j+2j}... let me just do it directly
    # For N=400, O(N^2) = 160000 operations, which is fast enough
    
    # dLSE/dh_i = sum_j w_j (h_{i+j} + h_{i-j})
    # = sum_j w_j h_{i+j} + sum_j w_j h_{i-j}
    
    # Compute via direct summation (N=400, so N^2 = 160k, fast)
    dLSE_dh = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j in range(N):
            if 0 <= i + j < N:
                s += w[j] * h[i + j]
            if 0 <= i - j < N:
                s += w[j] * h[i - j]
        dLSE_dh[i] = s
    
    # Now dval/dh_i = dLSE_dh_i / S^2 - 2 * LSE * dS/dh_i / S^3
    # dS/dh_i = sign(h_i) (since S = sum |h_i|)
    dS_dh = np.sign(h)
    
    dval_dh = dLSE_dh / (S * S) - 2.0 * LSE * dS_dh / (S * S * S)
    
    # dval/dx_i = dval/dh_i * dh/dx_i = dval/dh_i * (1 - h_i^2)
    dval_dx = dval_dh * (1.0 - h * h)
    
    return val, dval_dx


def _hill_climb(h, max_iters=2000, time_budget=5.0):
    """Discrete sign-flip hill climbing."""
    h = h.copy()
    best_val = _raw_value(h)
    start = time.time()
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        improved = False
        # Try flipping each sign
        for i in range(N):
            h[i] = -h[i]
            val = _raw_value(h)
            if val < best_val - 1e-12:
                best_val = val
                improved = True
            else:
                h[i] = -h[i]  # revert
            if improved:
                break
        if not improved:
            break
    
    return h, best_val


def _raw_value(h):
    """Compute the raw C3 upper bound value."""
    conv = _autoconv(h)
    max_conv = np.max(np.abs(conv))
    S = np.sum(np.abs(h))
    if S < 1e-10:
        return np.inf
    return max_conv / (S * S)


def _make_initialization(seed):
    """Create diverse initializations."""
    rng = np.random.RandomState(seed)
    init_type = seed % 5
    
    if init_type == 0:
        # Random
        x = rng.randn(N) * 0.5
    elif init_type == 1:
        # Block pattern
        block_size = 40
        x = np.zeros(N)
        for b in range(N // block_size):
            if rng.rand() > 0.5:
                x[b*block_size:(b+1)*block_size] = 1.0
            else:
                x[b*block_size:(b+1)*block_size] = -1.0
    elif init_type == 2:
        # Alternating
        x = np.zeros(N)
        x[::2] = 1.0
        x[1::2] = -1.0
        x += rng.randn(N) * 0.3
    elif init_type == 3:
        # Sparse blocks
        x = np.zeros(N)
        n_blocks = 5
        for _ in range(n_blocks):
            start = rng.randint(0, N - 20)
            length = rng.randint(10, 30)
            if rng.rand() > 0.5:
                x[start:start+length] = 1.0
            else:
                x[start:start+length] = -1.0
    else:
        # Low frequency sinusoid
        freq = rng.choice([1, 2, 3, 4])
        phase = rng.uniform(0, 2 * np.pi)
        x = np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
    
    return x


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_h = np.ones(N)
    best_val = _raw_value(best_h)
    
    beta = 20.0  # log-sum-exp temperature
    
    # Multiple restarts with diverse initializations
    n_restarts = 8
    for r in range(n_restarts):
        elapsed = time.time() - _START_TIME
        if elapsed > TIME_LIMIT - 20:  # leave 20s for final polish
            break
        
        x0 = _make_initialization(r * 137 + 42)
        
        # L-BFGS-B optimization
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_opt = res.x
        except Exception:
            x_opt = x0
        
        h = np.tanh(x_opt)
        val = _raw_value(h)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Hill climbing polish
        elapsed = time.time() - _START_TIME
        if elapsed > TIME_LIMIT - 15:
            break
        
        h_hc, val_hc = _hill_climb(h, max_iters=1000, time_budget=5.0)
        if val_hc < best_val:
            best_val = val_hc
            best_h = h_hc.copy()
    
    # Final hill climbing on best
    elapsed = time.time() - _START_TIME
    if elapsed < TIME_LIMIT - 10:
        best_h, best_val = _hill_climb(best_h, max_iters=3000, time_budget=10.0)
    
    # Ensure we return within deadline
    if time.time() - _START_TIME > TIME_LIMIT - 5:
        pass  # just return what we have
    
    return best_h
