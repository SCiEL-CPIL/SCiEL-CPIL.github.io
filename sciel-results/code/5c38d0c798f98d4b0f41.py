import numpy as np
from scipy.signal import windows
import time

N = 400

def _get_max_sidelobe(f):
    """Compute the maximum absolute value of the autoconvolution sidelobes."""
    # Use FFT for efficient autoconvolution
    f_fft = np.fft.fft(f, n=2*N)
    ac = np.fft.ifft(f_fft * f_fft).real
    center = N - 1
    sidelobes = np.delete(ac, center)
    return np.max(np.abs(sidelobes))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Parameterize the step function as a smooth window that we optimize.
    # We use a sum of Gaussians or a parametric family.
    # Let's try optimizing the coefficients of a basis expansion, or directly
    # optimize a parameterized window like a generalized Hamming window or
    # a linear combination of a few windows.
    
    # A powerful approach: parameterize f as a window with adjustable "shape".
    # One good family is the generalized cosine windows, or simply optimizing
    # the coefficients of a few basis functions (e.g., Gaussians of different widths).
    
    # Let's try a different approach: use scipy.optimize.minimize to directly
    # optimize the vector f subject to norm constraint, or parameterize f.
    # Since N=400, direct optimization of 400 variables might be slow but feasible
    # if we use a good gradient. However, the objective is non-smooth (max).
    # We can smooth it with a soft-max.
    
    # Alternative: Use a parametric model.
    # Let f(x) = exp(-alpha * x^2) * (1 + beta * x^4) or similar.
    # Or use a window defined by a few parameters.
    
    # Let's try optimizing a set of Gaussians.
    # f = sum_i w_i * exp(-a_i * (x - mu_i)^2)
    # With symmetry, mu_i = -mu_j.
    
    # Simpler: Optimize a generalized Kaiser window? No, let's use a different family.
    # How about a linear combination of a few orthogonal polynomials (like Chebyshev)
    # multiplied by a base window?
    
    # Let's try a direct optimization of the parameters of a multi-Gaussian mixture.
    # To keep it fast, we'll use a small number of components.
    
    # Actually, a very effective method for this specific problem (minimizing sidelobes)
    # is the "Dolphin-Chebyshev" window or similar, but those are fixed.
    # We can optimize a window defined by:
    # f[i] = exp(-p * (i/(N-1) - 0.5)**2 * 4) * (1 + q * (i/(N-1) - 0.5)**4)
    
    # Let's use scipy.optimize to find best p, q.
    
    def objective(params):
        p, q = params
        if p < 0:
            p = 0
        x = np.linspace(-0.5, 0.5, N)
        f = np.exp(-p * x**2 * 4) * (1 + q * x**4)
        norm = np.linalg.norm(f)
        if norm == 0:
            return 1e10
        f = f / norm
        return _get_max_sidelobe(f)
    
    best_val = float('inf')
    best_f = None
    
    # Grid search over p and q to find a good starting point
    for p in np.arange(0, 50, 1):
        if time.time() > deadline - 30:
            break
        for q in np.arange(-1, 1, 0.2):
            x = np.linspace(-0.5, 0.5, N)
            f = np.exp(-p * x**2 * 4) * (1 + q * x**4)
            norm = np.linalg.norm(f)
            if norm == 0:
                continue
            f = f / norm
            val = _get_max_sidelobe(f)
            if val < best_val:
                best_val = val
                best_f = f.copy()
                
    # Incorporate ideas from inspiration: try narrow boxcars and Gaussians
    # These represent extreme cases of the parametric family (very high p, or discrete support)
    if time.time() < deadline - 10:
        # Try narrow boxcars (widths 1, 2, 3, 5, 10)
        for width in [1, 2, 3, 5, 10, 20, 50, 100, 200, 400]:
            if time.time() > deadline - 10:
                break
            f = np.zeros(N)
            # Center the boxcar
            start_idx = (N - width) // 2
            end_idx = start_idx + width
            f[start_idx:end_idx] = 1.0
            norm = np.linalg.norm(f)
            if norm == 0:
                continue
            f = f / norm
            val = _get_max_sidelobe(f)
            if val < best_val:
                best_val = val
                best_f = f.copy()
                
        # Try a narrow Gaussian
        for sigma in [0.5, 1.0, 2.0, 5.0]:
            if time.time() > deadline - 10:
                break
            x = np.linspace(-N/2, N/2, N)
            f = np.exp(-x**2 / (2 * sigma**2))
            norm = np.linalg.norm(f)
            if norm == 0:
                continue
            f = f / norm
            val = _get_max_sidelobe(f)
            if val < best_val:
                best_val = val
                best_f = f.copy()

    # Refine with local optimization if we have time
    if best_f is not None and time.time() < deadline - 10:
        # We need to extract p, q from best_f or just restart optimization from a random point near best.
        # Let's just do a few random local optimizations from the grid search result.
        # Since we don't have the exact p, q, let's just try a few random p, q combinations 
        # in a finer grid or use a random restart on the parameters.
        
        # Actually, let's just try a few random p, q values to see if we can beat the grid.
        np.random.seed(42)
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            p = np.random.uniform(0, 50)
            q = np.random.uniform(-1, 1)
            x = np.linspace(-0.5, 0.5, N)
            f = np.exp(-p * x**2 * 4) * (1 + q * x**4)
            norm = np.linalg.norm(f)
            if norm == 0:
                continue
            f = f / norm
            val = _get_max_sidelobe(f)
            if val < best_val:
                best_val = val
                best_f = f.copy()
                
    if best_f is None:
        # Fallback to Kaiser
        f = windows.kaiser(N, 14)
        norm = np.linalg.norm(f)
        if norm == 0:
            return np.ones(N)
        return f / norm
        
    return best_f
