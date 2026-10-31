import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter1d
import time

N = 600
dx = 1.0 / (2 * N)  # interval width = (1/4 - (-1/4)) / N = 1/2 / 600

def compute_autoconvolution(heights):
    """Compute autoconvolution of a step function."""
    # Use FFT-based convolution with oversampling for accuracy
    fine_N = N * 2
    fine_dx = dx / fine_N
    fine_heights = np.repeat(heights, fine_N)
    conv_fine = fftconvolve(fine_heights, fine_heights, mode='full') * fine_dx
    return conv_fine

def compute_score(heights):
    """Compute normalized maximum of autoconvolution."""
    heights = np.maximum(heights, 0)
    
    # L2 norm squared = integral of f^2 = sum(h[i]^2) * dx
    l2_sq = np.sum(heights ** 2) * dx
    
    if l2_sq <= 0:
        return float('inf')
    
    autoconv = compute_autoconvolution(heights)
    max_autoconv = np.max(autoconv)
    
    return max_autoconv / l2_sq

def soft_max(values, temperature=0.01):
    """Approximate maximum using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    max_val = np.max(values)
    if max_val == 0:
        return 0.0
    return max_val + temperature * np.log(np.sum(np.exp((values - max_val) / temperature)))

def objective(heights):
    """Objective: soft-max of normalized autoconvolution."""
    heights = np.maximum(heights, 0)
    l2_sq = np.sum(heights ** 2) * dx
    if l2_sq <= 1e-15:
        return 1e10
    
    autoconv = compute_autoconvolution(heights)
    normalized = autoconv / l2_sq
    return soft_max(normalized, temperature=0.005)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Try multiple initial shapes
    x = np.linspace(0, 1, N)
    
    shapes = [
        np.ones(N),                          # Constant
        np.maximum(0, 1 - 2 * np.abs(x - 0.5)),  # Triangular
        np.maximum(0, 1 - 4 * (x - 0.5) ** 2),   # Parabolic
        np.maximum(0, np.sin(np.pi * x)),         # Sine
        np.exp(-8 * (x - 0.5) ** 2),              # Gaussian
    ]
    
    best_heights = None
    best_score = float('inf')
    
    for shape in shapes:
        score = compute_score(shape)
        if score < best_score:
            best_score = score
            best_heights = shape.copy()
    
    if best_heights is None:
        return [1.0] * N
    
    # Optimize from best initial shape
    try:
        result = minimize(
            objective,
            best_heights,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7, 'maxfun': 8000}
        )
        if result.fun < best_score:
            candidate = np.maximum(result.x, 0)
            score = compute_score(candidate)
            if score < best_score:
                best_score = score
                best_heights = candidate
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 15:
        return list(best_heights)
    
    # Additional restarts with random smooth initializations
    rng = np.random.RandomState(123)
    for restart in range(8):
        if time.time() > deadline - 5:
            break
        
        # Generate smooth random shape
        base = rng.randn(N)
        smooth = gaussian_filter1d(base, sigma=30)
        heights_init = np.maximum(smooth, 0)
        
        # Ensure some mass
        if np.sum(heights_init) < 1e-6:
            heights_init = np.ones(N) * 0.1
        
        try:
            result = minimize(
                objective,
                heights_init,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 4000}
            )
            if result.fun < best_score:
                candidate = np.maximum(result.x, 0)
                score = compute_score(candidate)
                if score < best_score:
                    best_score = score
                    best_heights = candidate
        except Exception:
            pass
    
    return list(best_heights)

