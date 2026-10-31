"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from numpy.fft import fft, ifft
import time


def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]


def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # Initialize with a smooth sinusoidal profile to distribute energy
    # We want to minimize peak autoconv. A sine wave has autoconv that is also sinusoidal (lower peaks relative to sum^2 than impulses).
    # x[i] = sin(2*pi*k*i/N)
    # Sum is 0, so we need a DC offset or a non-zero mean.
    # Let's try x[i] = 1 + a*sin(2*pi*k*i/N) where a is small enough to keep x in [-1, 1].
    # Actually, let's just use a base of 0.5 + 0.5*sin(...) or similar, or just optimize from a random smooth start.
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    def try_candidate(x):
        nonlocal best_x, best_score
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    # Strategy: Continuous Optimization
    # We define a loss function L(x) = max(|A|) / S^2
    # Since max is non-differentiable, we use a smooth approximation:
    # L_approx(x) = (sum(|A|^p))^(1/p) / S^2  for large p, or just sum(A^2)/S^2 (variance-like)
    # Minimizing sum(A^2)/S^2 is equivalent to minimizing the L2 norm of the autoconvolution relative to the integral.
    # By Parseval, sum(A^2) = sum(|F|^4) / N_fft.
    # So we minimize sum(|F|^4) / (sum(x))^2.
    # This is a smooth function!
    
    def _loss_smooth(x):
        s = np.sum(x)
        if abs(s) < 1e-12:
            return np.inf, np.inf, 0
        n = len(x)
        size = 2 * n - 1
        size2 = 1
        while size2 < size:
            size2 *= 2
        F = fft(x, size2)
        # sum(|F|^4)
        sum_f4 = np.sum(np.abs(F)**4)
        val = sum_f4 / (s * s)
        # Gradient of val w.r.t x
        # val = sum(|F|^4) * S^-2
        # d(val)/dx = (d(sum(|F|^4))/dx) * S^-2 - 2 * sum(|F|^4) * S^-3 * dS/dx
        # dS/dx = 1
        # d(sum(|F|^4))/dx:
        # |F|^2 = F * conj(F)
        # d(|F|^4)/dx = 4 * Re(|F|^2 * conj(dF/dx))
        # F = fft(x). dF_k/dx_j = exp(-2*pi*i*j*k/size2)
        # This is getting complex. Let's use the autoconvolution gradient directly.
        # A = x * x (linear)
        # sum(A^2) = sum_k (sum_j x_j x_{k-j})^2
        # This is equivalent to the 4-point correlation.
        # Gradient of sum(A^2) w.r.t x is 4 * (x * (x * x)) ? No.
        # Let's use the property that grad(sum(A^2)) = 4 * (A * x) where * is circular?
        # For linear convolution, the gradient of sum(A^2) is 4 * (x * A) where * is linear convolution?
        # Let's verify with small N.
        # Actually, simpler: use FFT gradient.
        # f(x) = sum_k |F_k|^4
        # df/dx_j = sum_k 4 Re(|F_k|^2 conj(F_k) conj(dF_k/dx_j))
        # dF_k/dx_j = exp(-2*pi*i*j*k/size2)
        # conj(dF_k/dx_j) = exp(2*pi*i*j*k/size2)
        # So df/dx_j = sum_k 4 Re(|F_k|^2 F_k^* exp(2*pi*i*j*k/size2))
        # Note that |F_k|^2 F_k^* = F_k^2? No. |F|^2 F^* = F F^* F^* = F (F^*)^2.
        # Let G_k = F_k^2. Then |F_k|^2 F_k^* = F_k^* F_k F_k^* = F_k (F_k^*)^2.
        # Wait, |F|^2 = F F^*. So |F|^2 F^* = F (F^*)^2.
        # We want sum_k 4 Re( F_k (F_k^*)^2 exp(2*pi*i*j*k/size2) )
        # Let H_k = F_k (F_k^*)^2.
        # The term is Re( sum_k H_k exp(2*pi*i*j*k/size2) )
        # This is the real part of the Inverse FFT of H_k!
        # ifft(H)_j = (1/size2) sum_k H_k exp(2*pi*i*j*k/size2)
        # So df/dx_j = 4 * size2 * Re( ifft(H)_j )
        
        H = F * (np.conj(F))**2
        grad_f = 4.0 * size2 * np.real(ifft(H))[:n]
        
        # Now combine with S^-2
        # val = f * S^-2
        # dval/dx = grad_f * S^-2 - 2 * f * S^-3
        # grad_val = grad_f / (s*s) - 2 * val / s
        
        grad_val = grad_f / (s * s) - 2.0 * val / s
        
        return val, grad_val, s
    
    # We will perform gradient descent on the smooth proxy L2-autoconv / S^2
    # Then project to [-1, 1] and maybe threshold to +-1 at the end?
    # The task asks for "signed step heights". It doesn't explicitly say they must be +-1.
    # "sequence of signed step heights".
    # If real values are allowed, we can get a much lower score.
    # If they must be +-1, we should threshold.
    # Let's assume real values are allowed as per "signed step heights" (unlike "binary sequence").
    # However, the seed program used +-1. The task description says "signed equal-width step functions".
    # Usually step functions imply constant values on intervals.
    # Let's optimize the real-valued version and return that. If the evaluator expects +-1, it might fail or score differently.
    # But "heights" suggests values.
    
    # Let's start from a smooth initialization
    x = np.zeros(N)
    # Initialize with a low frequency cosine to have a large sum and smooth shape
    # x = cos(2*pi*i/N) has sum 0.
    # x = 0.5 + 0.5*cos(2*pi*i/N) has sum N*0.5. Values in [0, 1].
    # To make it signed, maybe x = cos(2*pi*i/N) + 0.1?
    # Let's try x = 1.0 initially, then optimize.
    x = np.ones(N)
    
    # Gradient Descent
    lr = 0.1
    for step in range(5000):
        if time.time() > deadline:
            break
        
        val, grad, s = _loss_smooth(x)
        if not np.isfinite(val):
            break
            
        # Gradient step
        x_new = x - lr * grad
        
        # Project to [-1, 1]
        x_new = np.clip(x_new, -1.0, 1.0)
        
        # Check if improved on the ACTUAL score (max abs autoconv)
        # The smooth proxy might not perfectly correlate with max abs.
        # So we evaluate the real score periodically or on every step?
        # Real score is expensive (FFT + max).
        # Let's evaluate real score every 10 steps.
        
        if step % 10 == 0:
            real_sc = _score(x_new)
            if real_sc < best_score:
                best_score = real_sc
                best_x = x_new.copy()
            x = x_new
            
        else:
            x = x_new
            
        # Adaptive learning rate?
        # If val increases, reduce lr?
        # For now, keep it simple.
        
    # After gradient descent, we might be stuck in a local minimum of the L2 proxy.
    # Let's try to sharpen the solution towards +-1 if that helps?
    # Or just return the continuous solution.
    
    # One more check: try thresholding the best continuous solution to +-1
    if time.time() < deadline - 5.0:
        x_thresh = np.where(best_x > 0, 1.0, -1.0)
        try_candidate(x_thresh)
        
        # Also try the negative
        x_thresh_neg = -x_thresh
        try_candidate(x_thresh_neg)

    # Ensure we return a valid array
    if best_score == np.inf:
        return np.ones(N)
        
    return best_x

# EVOLVE-BLOCK-END
