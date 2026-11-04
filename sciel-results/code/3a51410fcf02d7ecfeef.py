"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    return float(np.max(np.abs(conv[:2 * n - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _soft_max_abs(x, beta=20.0):
    """
    Smooth approximation of max |autoconv| using log-sum-exp.
    This is differentiable with respect to x.
    """
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)[:2 * n - 1]
    abs_conv = np.abs(conv)
    m = np.max(abs_conv)
    if m < 1e-12:
        return 0.0
    # log-sum-exp trick for stability
    return m + np.log(np.sum(np.exp((abs_conv - m) * beta))) / beta

def _objective_and_grad(theta, beta=20.0):
    """
    Objective function for optimization.
    theta: array of phases (real values)
    We reconstruct x from theta.
    """
    # Reconstruct x from phases
    # We fix the amplitude to 1 for simplicity (or optimize amplitudes too?)
    # Let's stick to phases for now, magnitude 1.
    # x_k = exp(i * theta_k)
    # But we need real x. So we use the real part? Or optimize complex?
    # The problem allows signed step heights. 
    # If we restrict to x_k = cos(theta_k), we are in [-1, 1].
    # If we use x_k = exp(i theta_k), x is complex.
    # The autoconvolution of a real signal is real.
    # Let's use x = real(FFT^{-1}(exp(i*theta)))? No, that changes the spectrum.
    
    # Better approach: Parameterize x directly in the time domain?
    # The current strategy uses flips. The new strategy should use continuous heights.
    # Let's optimize the heights h_i in [-1, 1] directly.
    # Objective: max |conv(h)| / (sum h)^2
    
    # To make it smooth, we use soft max.
    # Variable: h (N,)
    # Bounds: [-1, 1]
    
    pass # This function is a placeholder, we will define the actual objective for L-BFGS-B

def _construct_via_optimization(deadline, rng):
    """
    Use L-BFGS-B to optimize continuous heights in [-1, 1].
    """
    # Initial guess: random continuous values
    x0 = rng.uniform(-1, 1, size=N)
    
    # Define objective for L-BFGS-B
    # We want to minimize f(x) = soft_max_abs_conv(x) / (sum(x))^2
    # L-BFGS-B requires gradient. We can use numerical gradient or analytic.
    # Analytic gradient of soft max is complex. Let's use numerical gradient first to be safe,
    # or use a subgradient method if L-BFGS-B is too slow/complex.
    # Given 120s limit, L-BFGS-B with numerical grad might be slow for N=400.
    # Let's try a custom subgradient/gradient descent with momentum.
    
    # Actually, let's try to use the spectral insight.
    # Autoconvolution in frequency domain: |FFT(x)|^2.
    # Max abs of IFFT(|FFT(x)|^2).
    
    # Let's stick to time-domain continuous optimization with a simple gradient step.
    # Gradient of soft_max is computable.
    
    def _soft_max_grad(x, beta=20.0):
        n = len(x)
        size = 1
        while size < 2 * n - 1:
            size *= 2
        fx = np.fft.rfft(x, size)
        conv = np.fft.irfft(fx * fx, size)[:2 * n - 1]
        abs_conv = np.abs(conv)
        m = np.max(abs_conv)
        if m < 1e-12:
            return np.zeros_like(x)
        
        # Gradient of log-sum-exp of abs_conv
        # d/dx log(sum(exp(beta * abs_conv))) = sum(exp(beta * abs_conv) * beta * sign(conv) * d(conv)/dx) / sum(exp(beta * abs_conv))
        
        weights = np.exp((abs_conv - m) * beta)
        weights /= np.sum(weights)
        
        # d(conv)/dx is the convolution kernel (symmetric)
        # conv[k] = sum_i x[i] x[k-i]
        # d conv[k] / d x[j] = x[k-j] + x[j-k] (with appropriate bounds)
        # This is essentially 2 * cross_corr(x, delta) but structured.
        # Actually, d/dx_j sum_k w_k |conv[k]| is not just w_k * d conv[k]/dx_j because of the abs.
        # grad_abs_conv[k] = sign(conv[k]) * d conv[k]/dx
        # d conv[k]/dx_j = x[k-j] + x[j-k]
        
        # We need to sum over k: w_k * sign(conv[k]) * (x[k-j] + x[j-k])
        # This is a convolution of (w_k * sign(conv[k])) with x.
        
        g_conv = weights * np.sign(conv)
        # g_conv is the gradient of the soft max w.r.t conv
        # Now we need gradient w.r.t x.
        # L = sum_k g_conv[k] * conv[k] (ignoring the log for a moment, it's a linear functional of conv)
        # d L / d x_j = sum_k g_conv[k] * (x[k-j] + x[j-k])
        # This is 2 * conv(g_conv, x) (with appropriate padding)
        
        # Let's compute this via FFT for efficiency
        # g_conv is defined on indices 0..2N-2
        # x is defined on 0..N-1
        
        # Pad g_conv to size
        g_padded = np.zeros(size)
        g_padded[:len(conv)] = g_conv
        
        # The gradient is the convolution of g_padded and x?
        # Let's verify with a small example or use numerical diff for the gradient step?
        # For speed, let's use the analytical gradient.
        # The term is sum_{k} g_k (x_{k-j} + x_{j-k})
        # This is equivalent to the cross-correlation of g and x, doubled.
        
        # Using FFT:
        # Convolution theorem: IFFT(FFT(g) * FFT(x))
        # But we need the sum over k.
        
        # Let's just use numerical gradient for the first few steps to ensure correctness,
        # or a simpler heuristic: coordinate descent.
        
        # Given the complexity, let's try a simpler approach:
        # Random restarts of continuous random vectors, then local search with small perturbations.
        # This is similar to the current strategy but in continuous space.
        
        return np.zeros_like(x) # Placeholder

    # Since analytical gradient is tricky, let's use a "greedy" continuous local search.
    # 1. Start with random continuous x.
    # 2. For each step, pick a random subset of indices, perturb them slightly (Gaussian noise).
    # 3. If metric improves, keep it.
    # This is essentially simulated annealing / hill climbing in continuous space.
    
    best_x = x0.copy()
    best_val = _metric(best_x)
    
    # Initial random search
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        cand = rng.uniform(-1, 1, size=N)
        v = _metric(cand)
        if v < best_val:
            best_val = v
            best_x = cand.copy()
    
    # Local search
    current_x = best_x.copy()
    current_val = best_val
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        # Pick a few random indices to perturb
        k = rng.integers(1, 5)
        indices = rng.choice(N, size=k, replace=False)
        
        # Perturb with Gaussian noise
        noise = rng.normal(0, step_size, size=k)
        cand = current_x.copy()
        cand[indices] += noise
        cand[indices] = np.clip(cand[indices], -1, 1)
        
        v = _metric(cand)
        if v < current_val:
            current_val = v
            current_x = cand
            no_improve_count = 0
            if v < best_val:
                best_val = v
                best_x = cand.copy()
        else:
            no_improve_count += 1
            # Anneal: reduce step size if stuck
            step_size *= 0.95
            if step_size < 0.001:
                break
                
    return best_x, best_val

def construct():
    # Deadline guard: return with a safety margin before the 150 s kill.
    deadline = time.time() + 130.0

    rng = np.random.default_rng(2024)

    best_x = np.ones(N)
    best_val = _metric(best_x)

    # Try continuous optimization
    if time.time() < deadline - 10:
        x_cont, val_cont = _construct_via_optimization(deadline, rng)
        if val_cont < best_val:
            best_val = val_cont
            best_x = x_cont

    # Try discrete search as fallback/alternative
    # (Keep the original discrete search logic but maybe reduce time if continuous was good)
    # For now, just return the best found.
    
    return best_x
# EVOLVE-BLOCK-END
