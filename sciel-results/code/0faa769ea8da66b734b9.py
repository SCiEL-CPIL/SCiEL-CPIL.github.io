"""
Optimized solution for the third autocorrelation inequality task.

Strategy: Represent h as a uniform background plus sparse spikes.
The autoconvolution peak is dominated by self-correlation of large elements.
We parameterize only spike locations and amplitudes for a low-dimensional search,
then refine with gradient-based methods.
"""
import numpy as np
import time
import scipy.fft as fft
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
DEADLINE = START_TIME + 140.0  # Safety margin before 150s kill

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = 2 * N - 1
    H = fft.rfft(h, n=n)
    c = fft.irfft(H * H, n=n)
    return c

def _evaluate(h):
    """Evaluate the objective: abs(2*N*max(autoconv) / sum(h)^2)"""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max(x, beta=200.0):
    """Differentiable approximation to max using log-sum-exp."""
    m = np.max(x)
    x_shifted = x - m
    # Use log-sum-exp trick for numerical stability
    log_sum_exp = m + np.log(np.sum(np.exp(beta * x_shifted))) - np.log(len(x))
    return log_sum_exp / beta

def _objective_smooth(h_flat):
    """Smooth objective for gradient-based optimization."""
    h = h_flat.reshape(N)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    sm = _soft_max(conv, beta=150.0)
    return abs(2.0 * N * sm / (s * s))

def _gradient_objective(h_flat):
    """Compute objective and approximate gradient via finite differences."""
    h = h_flat.reshape(N)
    val = _evaluate(h)
    
    # Central differences for gradient
    eps = 1e-7 * max(1.0, np.std(h))
    grad = np.zeros(N)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        val_plus = _evaluate(h_plus)
        
        h_minus = h.copy()
        h_minus[i] -= eps
        val_minus = _evaluate(h_minus)
        
        grad[i] = (val_plus - val_minus) / (2 * eps)
    
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_val = 1e10
    
    def try_h(h):
        nonlocal best_h, best_val
        if time.time() > deadline - 5.0:
            return
        val = _evaluate(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        return val
    
    # Phase 1: Sparse spike representation
    # h = dc * ones + sum of spikes at random locations
    # Spikes have large positive and negative amplitudes to cancel in sum but create small autoconv
    
    # Start with DC offset
    dc_values = [1.0, 2.0, 5.0, 10.0]
    
    # Try different numbers of spikes
    n_spikes_options = [4, 8, 12, 16, 20]
    
    for dc in dc_values:
        for n_spikes in n_spikes_options:
            if time.time() > deadline - 60.0:
                break
            
            # Multiple random spike configurations
            for trial in range(20):
                if time.time() > deadline - 55.0:
                    break
                
                h = dc * np.ones(N)
                
                # Place spikes at random locations, alternating signs
                locations = rng.choice(N, size=n_spikes, replace=False)
                for i, loc in enumerate(locations):
                    # Alternate signs to make sum of spikes ~0
                    sign = 1.0 if i % 2 == 0 else -1.0
                    amp = sign * dc * rng.uniform(0.5, 2.0)
                    h[loc] += amp
                
                try_h(h)
    
    # Phase 2: Gradient-based refinement from best found
    if best_h is None:
        best_h = np.ones(N)
        best_val = _evaluate(best_h)
    
    # Use L-BFGS-B with smooth surrogate
    if time.time() < deadline - 40.0:
        h0 = best_h.copy()
        
        try:
            result = minimize(
                _objective_smooth,
                h0.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = result.x.reshape(N)
            val_opt = _evaluate(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Local coordinate descent with adaptive step sizes
    if time.time() < deadline - 30.0:
        h_curr = best_h.copy()
        lr = 0.05 * np.std(h_curr)
        
        for coord_iter in range(300):
            if time.time() > deadline - 25.0:
                break
            
            # Pick a random coordinate
            j = rng.integers(0, N)
            old_val = h_curr[j]
            
            # Try several step sizes
            for step_factor in [1.0, 0.5, 0.25, -0.25, -0.5, -1.0]:
                h_curr[j] = old_val + step_factor * lr
                val_new = _evaluate(h_curr)
                
                if val_new < best_val:
                    best_val = val_new
                    best_h = h_curr.copy()
                    break
                else:
                    h_curr[j] = old_val
            else:
                # No improvement, decay learning rate
                lr *= 0.95
                if lr < 1e-8:
                    break
    
    # Phase 4: Final random perturbations with small steps
    if time.time() < deadline - 10.0:
        for _ in range(500):
            if time.time() > deadline - 8.0:
                break
            
            scale = 0.01 * np.std(best_h)
            h_new = best_h + rng.standard_normal(N) * scale
            val_new = _evaluate(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
    
    # Safety check
    if best_h is None or abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _evaluate(best_h)
    
    return best_h
