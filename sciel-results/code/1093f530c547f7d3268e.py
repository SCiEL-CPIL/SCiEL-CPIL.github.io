"""
Strategy: ADMM-based convex optimization to flatten the full autoconvolution profile.

Instead of using structured sequences (Zadoff-Chu, Golay) and trying to suppress sidelobes,
we directly solve the convex problem:
    min ||conv(h,h) - C*1||_1 + (rho/2)*||h - 1/N||^2
    subject to sum(h) = 1

where C is chosen to be close to the target (we want the autoconvolution to be as flat as
possible, ideally C = sum(h)^2 / (2*N+1) for perfect flatness, but we let ADMM find the
optimal C implicitly through the l1 penalty).

The ADMM update rules are:
    h-update: minimize (rho/2)||h - z||^2 + (rho/2)||h - 1/N||^2  =>  h = (z + 1/N)/2
    z-update (soft-thresholding): z = soft_threshold(conv(h,h) + u, 1/rho)
    u-update: u += conv(h,h) - z

We alternate between these updates. The key insight is that by using FFT for the convolution
in each iteration, we can run many ADMM iterations within the time budget.

We also add a final polishing step with coordinate descent to fine-tune individual h values.
"""

import numpy as np
import time

N = 400
_DEADLINE = 145.0  # wall-clock limit with safety margin


def _fft_convolve(h, h2):
    """Compute full convolution using FFT, padded to avoid circular effects."""
    n = len(h)
    n2 = len(h2)
    size = n + n2 - 1
    # Next power of 2 for efficient FFT
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    H2 = np.fft.rfft(h2, fsize)
    conv = np.fft.irfft(H * H2, fsize)[:size]
    return conv


def _autoconv(h):
    """Compute autoconvolution h * h using FFT."""
    return _fft_convolve(h, h)


def _value(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _admm_flatten(h0, rho=1.0, max_iters=2000, tol=1e-8, deadline=None):
    """
    ADMM algorithm to minimize ||conv(h,h) - C*1||_1 + (rho/2)||h - h0||^2
    subject to sum(h) = sum(h0).
    
    We use the formulation:
        min_h,z  ||conv(h,h) - z||_1 + (rho/2)||h - h0||^2
        s.t. sum(h) = sum(h0)
    
    The ADMM updates (with the sum constraint handled by projection):
        h^{k+1} = argmin (rho/2)||h - (z^k - u^k)||^2 + (rho/2)||h - h0||^2  s.t. sum(h)=sum(h0)
                 => h^{k+1} = (z^k - u^k + h0) / 2, then project to sum constraint
        
        z^{k+1} = soft_threshold(conv(h^{k+1}, h^{k+1}) + u^k, 1/rho)
        
        u^{k+1} = u^k + conv(h^{k+1}, h^{k+1}) - z^{k+1}
    
    The soft-thresholding operation drives z toward a constant value, which in turn
    drives conv(h,h) toward a constant, flattening the autocorrelation profile.
    """
    n = len(h0)
    h = h0.copy().astype(float)
    z = _autoconv(h).copy()
    u = np.zeros_like(z)
    
    target_sum = np.sum(h0)
    
    best_val = _value(h)
    best_h = h.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # h-update: h = (z - u + h0) / 2, then project to sum constraint
        h_candidate = (z - u + h0) / 2.0
        # Project to sum constraint: adjust so sum(h_candidate) = target_sum
        diff = target_sum - np.sum(h_candidate)
        h_candidate += diff / n
        
        # z-update: soft-thresholding
        conv_h = _autoconv(h_candidate)
        z = np.sign(conv_h + u) * np.maximum(np.abs(conv_h + u) - 1.0 / rho, 0.0)
        
        # u-update
        u = u + conv_h - z
        
        # Evaluate
        val = _value(h_candidate)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_h = h_candidate.copy()
        
        h = h_candidate
    
    return best_h, best_val


def _coordinate_descent(h, max_steps=50000, deadline=None, step_size=0.01):
    """
    Coordinate descent polishing step.
    For each index, try small perturbations and keep if they improve the value.
    """
    n = len(h)
    best_val = _value(h)
    best_h = h.copy()
    rng = np.random.default_rng(12345)
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.integers(0, n)
        delta = rng.normal(0, step_size)
        h_new = best_h.copy()
        h_new[idx] += delta
        
        val_new = _value(h_new)
        if np.isfinite(val_new) and val_new < best_val - 1e-12:
            best_val = val_new
            best_h = h_new
    
    return best_h, best_val


def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initialize with ones (seed)
    h_init = np.ones(N)
    best_h = h_init.copy()
    best_val = _value(best_h)  # Should be 2.0
    
    # Phase 1: ADMM with different rho values
    rho_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    for rho in rho_values:
        if time.time() > deadline - 30:
            break
        
        # Run ADMM from the current best
        h_admm, val_admm = _admm_flatten(
            best_h, 
            rho=rho, 
            max_iters=1500, 
            tol=1e-9,
            deadline=deadline - 10
        )
        
        if np.isfinite(val_admm) and val_admm < best_val:
            best_val = val_admm
            best_h = h_admm.copy()
    
    # Phase 2: Multi-start ADMM from different initializations
    if time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        
        # Try a few different initializations
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            
            # Different initializations:
            if trial == 0:
                h_init = np.ones(N)
            elif trial == 1:
                # Smooth bump
                x = np.linspace(-1, 1, N)
                h_init = np.exp(-x**2 * 5)
            elif trial == 2:
                # Cosine window
                x = np.linspace(0, np.pi, N)
                h_init = np.cos(x) + 0.5
            elif trial == 3:
                # Random with small noise
                h_init = np.ones(N) + rng.normal(0, 0.1, N)
            else:
                # Two bumps
                x = np.linspace(-1, 1, N)
                h_init = np.exp(-(x-0.3)**2 * 10) + np.exp(-(x+0.3)**2 * 10)
            
            # Normalize to sum = 1
            s = np.sum(h_init)
            if abs(s) > 1e-10:
                h_init = h_init / s
            
            h_admm, val_admm = _admm_flatten(
                h_init, 
                rho=1.0, 
                max_iters=1000, 
                tol=1e-9,
                deadline=deadline - 10
            )
            
            if np.isfinite(val_admm) and val_admm < best_val:
                best_val = val_admm
                best_h = h_admm.copy()
    
    # Phase 3: Coordinate descent polishing
    if time.time() < deadline - 5:
        h_cd, val_cd = _coordinate_descent(
            best_h, 
            max_steps=100000, 
            deadline=deadline - 3,
            step_size=0.005
        )
        
        if np.isfinite(val_cd) and val_cd < best_val:
            best_val = val_cd
            best_h = h_cd.copy()
    
    # Final validation
    final_val = _value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
    
    return best_h
