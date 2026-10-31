"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 600
L = 0.5  # length of interval [-1/4, 1/4]
dx = L / N  # width of each interval

def _autoconvolution_peak(heights):
    """Compute the maximum value of the autoconvolution of a step function."""
    # heights: array of length N, nonnegative
    # The step function is f(x) = heights[i] for x in [-1/4 + i*dx, -1/4 + (i+1)*dx)
    # Autoconvolution (f*f)(t) = ∫ f(x) f(t-x) dx
    # For a step function, this can be computed via FFT.
    # We need to be careful with the normalization.
    
    # Pad to at least 2*N to avoid circular convolution issues
    # Actually, the support of f*f is [-1/2, 1/2] if f is supported on [-1/4, 1/4]
    # We'll compute the linear convolution via FFT
    
    # Use numpy's convolve or FFT
    # For efficiency, use FFT
    size = 2 * N
    # Zero-pad the signal
    f_padded = np.zeros(size)
    f_padded[:N] = heights
    
    # FFT, multiply, inverse FFT
    F = np.fft.fft(f_padded)
    conv = np.fft.ifft(F * F).real
    
    # The convolution is on a grid with spacing dx
    # The result conv[k] corresponds to (f*f)(k*dx) approximately
    # But we need to be careful: the FFT convolution gives us the values at the grid points
    # The maximum should be near the center (t=0)
    
    # The linear convolution of two length-N signals has length 2N-1
    # With our padding, conv has length 2N
    # The peak is around index N-1 or N
    # Let's just take the max of the whole thing, but it should be near the center
    
    return np.max(conv) * dx  # Multiply by dx because the integral approximation

def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin: stop at 120s, evaluator kills at 150s
    
    # Start with a reasonable initial guess
    # A Gaussian-like shape centered at 0
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    x = x + dx/2  # center of each interval
    
    # Initial guess: a smooth bump
    sigma = 0.1
    heights = np.exp(-x**2 / (2*sigma**2))
    heights = np.maximum(heights, 0)
    
    # Normalize to have a fixed L1 norm (or L2 norm) to avoid trivial scaling
    # The problem says "minimize the normalized maximum of its autoconvolution"
    # Normalization: likely L1 norm = 1 or L2 norm = 1
    # Let's assume L1 norm = 1 (probability density normalization)
    # This is a common normalization for such problems
    
    heights = heights / np.sum(heights) * N  # so that sum(heights)*dx = 1, i.e., L1 norm = 1
    
    best_heights = heights.copy()
    best_score = _autoconvolution_peak(best_heights)
    
    # Use differential evolution or simple gradient-based optimization
    # Since we have a smooth objective, let's try a simple approach:
    # 1. Start with a Gaussian
    # 2. Optimize the Gaussian parameters (center, width, amplitude)
    # 3. Then do local refinement
    
    # Actually, let's use scipy's minimize if available, with a time guard
    
    from scipy.optimize import minimize
    
    def objective(params):
        # params: [center, log_sigma, log_amplitude]
        center = params[0]
        log_sigma = params[1]
        log_amp = params[2]
        
        sigma = np.exp(log_sigma)
        amp = np.exp(log_amp)
        
        h = amp * np.exp(-(x - center)**2 / (2*sigma**2))
        h = np.maximum(h, 0)
        
        # Normalize to L1 norm = 1
        norm = np.sum(h)
        if norm < 1e-10:
            return 1e10
        h = h / norm * N
        
        return _autoconvolution_peak(h)
    
    # Initial parameters for the Gaussian
    x0 = [0.0, np.log(0.1), np.log(1.0)]
    
    bounds = [(-0.1, 0.1), (np.log(0.01), np.log(0.25)), (-5, 5)]
    
    # Try to optimize with time limit
    elapsed = time.time() - start_time
    if elapsed < 100:  # leave 20s margin
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                # Reconstruct the heights
                center = result.x[0]
                sigma = np.exp(result.x[1])
                amp = np.exp(result.x[2])
                h = amp * np.exp(-(x - center)**2 / (2*sigma**2))
                h = np.maximum(h, 0)
                norm = np.sum(h)
                if norm > 1e-10:
                    best_heights = h / norm * N
        except Exception:
            pass
    
    # Now do local refinement using gradient descent on the full vector
    elapsed = time.time() - start_time
    if elapsed < 110:
        try:
            # Use a simple projected gradient method
            lr = 0.01
            for step in range(200):
                if time.time() - start_time > 115:
                    break
                
                # Compute gradient numerically
                eps = 1e-4
                grad = np.zeros(N)
                for i in range(0, N, 10):  # subsample for speed
                    h_plus = best_heights.copy()
                    h_plus[i] += eps
                    h_plus = h_plus / np.sum(h_plus) * N
                    h_minus = best_heights.copy()
                    h_minus[i] -= eps
                    h_minus = np.maximum(h_minus, 0)
                    h_minus = h_minus / np.sum(h_minus) * N
                    
                    grad[i] = (_autoconvolution_peak(h_plus) - _autoconvolution_peak(h_minus)) / (2*eps)
                    # Scale to all elements in the subsample
                    for j in range(i, min(i+10, N)):
                        grad[j] = grad[i]
                
                # Update
                best_heights = best_heights - lr * grad
                best_heights = np.maximum(best_heights, 0)
                
                # Normalize
                norm = np.sum(best_heights)
                if norm > 1e-10:
                    best_heights = best_heights / norm * N
                
                # Check improvement
                current_score = _autoconvolution_peak(best_heights)
                if current_score < best_score:
                    best_score = current_score
                else:
                    # Revert if no improvement
                    best_heights = best_heights + lr * grad
                    best_heights = np.maximum(best_heights, 0)
                    norm = np.sum(best_heights)
                    if norm > 1e-10:
                        best_heights = best_heights / norm * N
        except Exception:
            pass
    
    # Ensure nonnegativity and normalization
    best_heights = np.maximum(best_heights, 0)
    norm = np.sum(best_heights)
    if norm > 1e-10:
        best_heights = best_heights / norm * N
    else:
        best_heights = np.ones(N)
    
    return list(best_heights)
