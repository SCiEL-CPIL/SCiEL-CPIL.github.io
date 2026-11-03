"""Improved program using spectral parameterization for Erdos minimum overlap."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _overlap_values(h):
    """Compute all overlap values O_k for shifts k=1..N-1."""
    n = len(h)
    overlaps = np.zeros(n - 1)
    for k in range(1, n):
        # O_k = sum_{i=0}^{n-1-k} h[i] * (1 - h[i+k])
        overlaps[k-1] = np.dot(h[:n-k], 1 - h[k:])
    return overlaps

def _max_overlap(h):
    """Return the maximum overlap value."""
    return np.max(_overlap_values(h))

def _spectral_to_h(coeffs):
    """
    Convert spectral coefficients to height sequence h in [0,1].
    coeffs[0] is the DC offset (fixed to 0.5 for sum=N/2).
    coeffs[1:] are cosine/sine coefficients for low frequencies.
    
    We use: h[i] = 0.5 + sum_{m=1}^{M} (a_m * cos(2*pi*m*i/N) + b_m * sin(2*pi*m*i/N))
    Then clip to [0,1] and adjust to ensure sum = N/2.
    """
    n = len(coeffs) // 2 + 1  # number of frequency modes
    i = np.arange(N)
    h = np.full(N, 0.5)  # DC component ensures mean = 0.5
    
    for m in range(1, n):
        a = coeffs[2*m - 2]
        b = coeffs[2*m - 1]
        h += a * np.cos(2 * np.pi * m * i / N) + b * np.sin(2 * np.pi * m * i / N)
    
    # Clip to [0, 1]
    h = np.clip(h, 0, 1)
    
    # Adjust to ensure sum is exactly N/2
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    
    if abs(diff) > 1e-10:
        # Distribute the correction uniformly, then re-clip
        h += diff / N
        h = np.clip(h, 0, 1)
        
        # If still not exact, adjust the largest element
        residual = target_sum - np.sum(h)
        if abs(residual) > 1e-12:
            # Find element with most room to absorb residual
            if residual > 0:
                idx = np.argmin(1 - h)
            else:
                idx = np.argmax(h)
            h[idx] += residual
            h[idx] = np.clip(h[idx], 0, 1)
    
    return h

def _objective(coeffs):
    """Objective function: maximize overlap to minimize."""
    h = _spectral_to_h(coeffs)
    return _max_overlap(h)

def construct():
    """
    Construct step heights h[0..N-1] in [0,1] with sum = N/2
    to minimize the maximum overlap of h with shifted (1-h).
    
    Uses spectral parameterization with low-frequency cosine/sine terms.
    The DC component is fixed at 0.5 to naturally satisfy the sum constraint.
    Optimizes a small number of spectral coefficients via SLSQP.
    """
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Start with the known good solution: step function
    h_step = np.zeros(N)
    h_step[N//2:] = 1.0
    # Adjust for exact sum
    if N % 2 == 1:
        h_step[N//2] = 0.5
    
    # Try spectral optimization with different numbers of modes
    best_h = h_step.copy()
    best_score = _max_overlap(best_h)
    
    # Number of frequency modes to try
    max_modes = 5
    
    for n_modes in range(1, max_modes + 1):
        if time.time() > deadline:
            break
            
        # Initialize coefficients: try to match the step function spectrum
        coeffs = np.zeros(2 * n_modes)
        
        # Compute DFT of the step function to initialize
        H = np.fft.fft(h_step)
        for m in range(1, n_modes + 1):
            coeffs[2*m - 2] = 2 * np.real(H[m]) / N
            coeffs[2*m - 1] = -2 * np.imag(H[m]) / N
        
        # Optimize
        result = minimize(
            _objective,
            coeffs,
            method='SLSQP',
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if result.success:
            h_opt = _spectral_to_h(result.x)
            score_opt = _max_overlap(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
        
        # Also try random initializations
        for trial in range(3):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + trial + n_modes * 100)
            coeffs_rand = np.random.uniform(-0.3, 0.3, size=2 * n_modes)
            
            result = minimize(
                _objective,
                coeffs_rand,
                method='SLSQP',
                options={'maxiter': 150, 'ftol': 1e-8}
            )
            
            if result.success:
                h_opt = _spectral_to_h(result.x)
                score_opt = _max_overlap(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt
    
    # Final local polish: coordinate descent on the best spectral solution
    if time.time() < deadline - 5:
        # Extract spectral coefficients from best_h
        H = np.fft.fft(best_h)
        n_modes = 3
        coeffs = np.zeros(2 * n_modes)
        for m in range(1, n_modes + 1):
            coeffs[2*m - 2] = 2 * np.real(H[m]) / N
            coeffs[2*m - 1] = -2 * np.imag(H[m]) / N
        
        # Polish with Nelder-Mead
        result = minimize(
            _objective,
            coeffs,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if result.success:
            h_polished = _spectral_to_h(result.x)
            score_polished = _max_overlap(h_polished)
            
            if score_polished < best_score:
                best_h = h_polished
                best_score = score_polished
    
    return best_h

# EVOLVE-BLOCK-END
