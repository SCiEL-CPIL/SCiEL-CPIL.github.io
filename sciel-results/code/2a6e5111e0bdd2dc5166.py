import numpy as np
import time
from scipy.optimize import minimize
import scipy.fft as fft

N = 400

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT."""
    n = 2 * N - 1
    H = fft.rfft(h, n=n)
    c = fft.irfft(H * H, n=n)
    return c

def _soft_max(x, beta=100.0):
    """Differentiable approximation to max."""
    m = np.max(x)
    # Avoid overflow in exp
    x_shifted = x - m
    # If beta is large, this is close to max
    # log(sum(exp(beta*x_shifted))/len) / beta + m
    # Using log-sum-exp trick for stability
    max_x = np.max(x_shifted)
    sum_exp = np.sum(np.exp(beta * (x_shifted - max_x)))
    if sum_exp == 0:
        return m
    return m + max_x / 1.0 + np.log(sum_exp) / beta # Wait, the formula is:
    # max(x) + (1/beta) * log( sum( exp(beta*(x_i - max(x))) ) )
    # Let's re-verify.
    # log(sum exp(beta x_i)) = beta max(x) + log(sum exp(beta(x_i - max(x))))
    # (1/beta) log(sum exp(beta x_i)) = max(x) + (1/beta) log(sum exp(beta(x_i - max(x))))
    # So:
    return m + (1.0 / beta) * np.log(sum_exp)

def _objective(params):
    """
    params: [mu, A1, phi1, A2, phi2, ...]
    We fix sum(h) = 1 by construction? 
    Let's just optimize h directly but project to sum=1?
    Or better, optimize h directly with a penalty for sum(h) != 1?
    Actually, scale invariance means we can just minimize the ratio.
    Let's minimize V(h) directly using soft-max.
    """
    h = params
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta=200.0)
    
    val = 2.0 * N * sm / (s * s)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = 2.0
    
    def try_h(h):
        nonlocal best_h, best_val
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        c = _autoconv_fft(h)
        val = 2.0 * N * np.max(c) / (s * s)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        return val

    # Phase 1: Initial Grid Search with FFT
    # Try DC + 1 sinusoidal mode
    dc_values = [1.0, 2.0, 5.0, 10.0]
    freqs = [1, 2, 3, 5, 7, 10, 15, 20]
    amp_fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    
    t = np.arange(N)
    
    for dc in dc_values:
        for freq in freqs:
            for amp_frac in amp_fracs:
                if time.time() > deadline - 30.0:
                    break
                amp = dc * amp_frac
                # Antisymmetric mode
                a = amp * np.sin(2 * np.pi * freq * (t - (N - 1) / 2.0) / N)
                a = (a - a[::-1]) / 2.0
                h = dc * np.ones(N) + a
                try_h(h)
                
                # Symmetric mode
                a_sym = amp * np.cos(2 * np.pi * freq * (t - (N - 1) / 2.0) / N)
                a_sym = (a_sym + a_sym[::-1]) / 2.0
                h_sym = dc * np.ones(N) + a_sym
                try_h(h_sym)

    # Phase 2: Local Optimization with L-BFGS
    # Use soft-max objective for differentiability
    def objective_scipy(h_flat):
        return _objective(h_flat)

    # Start from the best found in Phase 1
    h0 = best_h.copy()
    
    # Normalize h0 to sum=1 to help optimizer? 
    # The objective is scale invariant, so it doesn't matter, 
    # but gradients might be better conditioned if scale is ~1.
    s0 = np.sum(h0)
    if abs(s0) > 1e-10:
        h0 = h0 / s0
        # Adjust best_val? No, best_val is the raw value.
        # The optimizer will find a direction.
        # We need to map back to original scale or just let it run.
        # Since it's scale invariant, the value is the same.
        
    try:
        # Run L-BFGS
        # We need to be careful with the soft-max beta.
        # Start with lower beta, increase?
        # Or just use a fixed moderate beta.
        
        for restart in range(10):
            if time.time() > deadline - 20.0:
                break
            
            if restart == 0:
                x0 = h0.flatten()
            else:
                # Perturb
                noise_scale = 0.1 * np.std(h0) * (1 + 0.5 * restart)
                x0 = h0.flatten() + rng.standard_normal(N) * noise_scale
            
            try:
                result = minimize(
                    objective_scipy,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_opt = result.x
                val_opt = _evaluate_exact(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    except Exception:
        pass

    # Phase 3: Final Polish with exact max
    # Do a few steps of coordinate descent or small random steps
    # to refine the exact max location.
    
    for _ in range(500):
        if time.time() > deadline - 5.0:
            break
        h_new = best_h + rng.standard_normal(N) * 0.01 * np.std(best_h)
        val_new = _evaluate_exact(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new

    return best_h

def _evaluate_exact(h):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    return abs(2.0 * N * np.max(c) / (s * s))

# Note: I defined _evaluate_exact after construct, but construct uses it.
# In Python, functions are looked up at call time, so this is fine as long as 
# construct is called after the module is loaded.
# However, it's cleaner to define it before.
# Let's move it up in the final output.
