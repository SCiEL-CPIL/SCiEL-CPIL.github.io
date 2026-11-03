"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.optimize as opt
import time
from scipy.fft import fft, ifft, fftfreq

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    
    The score is the upper bound on C1, computed as:
    max(autoconvolution) / (||f||_inf * ||f||_1)
    
    Strategy: Use a smooth, symmetric, nonnegative base function and optimize
    a few shape parameters (e.g., width, height profile) using local optimization.
    We represent f as a sum of a few basis functions that maintain non-negativity.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Interval setup
    # 600 intervals on [-1/4, 1/4]
    # dx = (1/2) / 600 = 1/1200
    dx = 0.25 / N
    # x values at interval centers
    x = np.linspace(-0.25 + dx/2, 0.25 - dx/2, N)
    
    # Autoconvolution in discrete form:
    # (f * f)[k] = sum_j f[j] f[k-j] * dx^2 (approximately)
    # But for step functions on equal intervals, the autoconvolution at bin m is:
    # sum_j f[j] * f[m-j] where indices wrap or are zero-padded
    
    # Since f is supported on [-1/4, 1/4], the autoconvolution is supported on [-1/2, 1/2]
    # With N=600 bins on [-1/4, 1/4], the autoconvolution has up to 2N-1 = 1199 bins
    
    # For efficient computation, use FFT
    def autoconvolve_peak(f_vals):
        """Compute the normalized peak autoconvolution value."""
        if not np.all(f_vals >= 0):
            # Penalty for non-negativity violation
            return 1e6
        f = np.asarray(f_vals, dtype=float)
        # Zero-pad to length >= 2*N-1 for linear convolution
        n_fft = 1 << int(np.ceil(np.log2(2 * N - 1)))
        F = fft(f, n=n_fft)
        conv = ifft(F * F).real[:2*N-1]
        # The autoconvolution sum: sum_j f[j] f[k-j] dx^2
        # But we need to be careful about normalization
        # The raw discrete convolution is: sum_j f[j] * f[k-j]
        # The continuous autoconvolution is approximately: dx * sum_j f[j] * f[k-j] * dx = dx^2 * sum
        # ||f||_inf = max(f)
        # ||f||_1 = sum(f) * dx
        # So normalized peak = (dx^2 * max(conv)) / (max(f) * sum(f) * dx) = dx * max(conv) / (max(f) * sum(f))
        
        max_f = np.max(f)
        if max_f < 1e-12:
            return 1e6
        sum_f = np.sum(f)
        if sum_f < 1e-12:
            return 1e6
        
        peak_conv = np.max(conv)
        normalized_peak = dx * peak_conv / (max_f * sum_f)
        return normalized_peak
    
    # Parameterize f using a smooth, symmetric, nonnegative profile
    # Use a raised cosine or Gaussian-like shape parameterized by a few coefficients
    # f[x] = max(0, 1 - (x/sigma)^2)^alpha * amp
    
    # Better: use a polynomial basis that is symmetric and nonnegative
    # Let u = 2|x| - 0.25, so u in [0, 0.25] for x in [0, 0.25]
    # Actually, let's use t = 4|x| - 1, so t in [-1, 1]
    # For x in [-1/4, 1/4], |x| in [0, 1/4], so 4|x| in [0, 1], t = 4|x| - 1 in [-1, 0]
    # Let s = 1 - 4|x|, so s in [0, 1]
    # f[x] = amp * g(s) where g is a nonnegative function on [0,1]
    
    s = 1 - 4 * np.abs(x)  # s in [0, 1], s=1 at x=0, s=0 at x=±1/4
    
    # Use a polynomial: g(s) = a0 + a1*s + a2*s^2 + a3*s^3
    # Subject to g(s) >= 0 for s in [0,1]
    # Symmetry is built in
    
    def f_from_params(params):
        amp = params[0]
        a0 = params[1]
        a1 = params[2]
        a2 = params[3]
        a3 = params[4]
        g = a0 + a1 * s + a2 * s**2 + a3 * s**3
        # Ensure non-negativity: clip negative values
        g = np.maximum(g, 0)
        return amp * g
    
    def objective(params):
        if time.time() > deadline:
            return 1e6
        f = f_from_params(params)
        return autoconvolve_peak(f)
    
    # Try multiple initializations and pick the best
    best_score = 1e6
    best_f = None
    
    # Several random initializations
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        
        # Random positive parameters
        amp = np.random.uniform(0.5, 2.0)
        a0 = np.random.uniform(0.1, 1.0)
        a1 = np.random.uniform(-0.5, 0.5)
        a2 = np.random.uniform(-0.5, 0.5)
        a3 = np.random.uniform(-0.3, 0.3)
        
        params0 = np.array([amp, a0, a1, a2, a3])
        
        # Local optimization using L-BFGS-B
        try:
            result = opt.minimize(objective, params0, method='L-BFGS-B',
                                   options={'maxiter': 200, 'ftol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_f = f_from_params(result.x)
        except Exception:
            pass
    
    # Also try some structured initializations
    structured_inits = [
        # Triangle: g(s) = s
        [1.0, 0.0, 1.0, 0.0, 0.0],
        # Parabolic: g(s) = s^2
        [1.0, 0.0, 0.0, 1.0, 0.0],
        # Cubic: g(s) = s^3
        [1.0, 0.0, 0.0, 0.0, 1.0],
        # Gaussian-like: g(s) = 1 - (1-s)^2
        [1.0, -1.0, 1.0, 0.0, 0.0],
        # Smooth bump: g(s) = 4*s*(1-s)
        [1.0, 0.0, 4.0, 0.0, 0.0],
    ]
    
    for params0 in structured_inits:
        if time.time() > deadline - 20:
            break
        try:
            result = opt.minimize(objective, np.array(params0, dtype=float), method='L-BFGS-B',
                                   options={'maxiter': 300, 'ftol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_f = f_from_params(result.x)
        except Exception:
            pass
    
    # If best_f is still None or score is bad, use a default
    if best_f is None or best_score > 1.0:
        # Use a simple triangle: f[x] = 1 - 4|x|
        best_f = 1.0 - 4.0 * np.abs(x)
        best_score = autoconvolve_peak(best_f)
    
    # Final polish: try a few more local optimizations from the best found
    if best_score < 1e5 and time.time() < deadline - 30:
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            # Perturb the current best parameters
            # We need to recover params from best_f... this is hard.
            # Instead, just do a small random search around structured points
            pass
    
    # Ensure non-negativity
    best_f = np.maximum(best_f, 0.0)
    
    # Verify the score
    final_score = autoconvolve_peak(best_f)
    
    return list(best_f)
# EVOLVE-BLOCK-END
