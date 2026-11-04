"""Parametric Chirp Design: Minimize autoconvolution peak via continuous Zadoff-Chu-like sequences."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _chirp_sequence(beta):
    n = np.arange(N)
    return np.cos(2 * np.pi * beta * n * n / N)

def _chirp_metric(beta):
    x = _chirp_sequence(beta)
    return _metric(x)

def _try_chirp_optimization(deadline):
    """
    Optimize the chirp parameter beta to minimize the autoconvolution metric.
    The chirp sequence has constant envelope (magnitude 1), which theoretically
    minimizes the autoconvolution sidelobes relative to the main lobe.
    """
    # The metric is periodic in beta with period N/2 (since n^2 mod N).
    # We search in a fundamental domain.
    # Note: beta is dimensionless.
    
    best_beta = 0.0
    best_val = float('inf')
    best_x = None
    
    # Coarse scan to find good regions
    n_coarse = 200
    betas = np.linspace(0.1, N/2 - 0.1, n_coarse)
    
    for b in betas:
        if time.time() > deadline - 5:
            break
        val = _chirp_metric(b)
        if val < best_val:
            best_val = val
            best_beta = b
            best_x = _chirp_sequence(b)
            
    # Fine optimization using Brent's method in the neighborhood of the best coarse point
    # We need to handle periodicity carefully.
    # Let's try a few local minima refinement steps.
    
    # Identify a few local minima from the coarse scan
    vals = []
    for b in betas:
        vals.append(_chirp_metric(b))
    vals = np.array(vals)
    
    # Find indices of local minima
    local_min_indices = []
    for i in range(1, n_coarse - 1):
        if vals[i] < vals[i-1] and vals[i] < vals[i+1]:
            local_min_indices.append(i)
    
    # Sort by value and take top 5
    local_min_indices.sort(key=lambda i: vals[i])
    top_indices = local_min_indices[:5]
    
    for idx in top_indices:
        if time.time() > deadline - 2:
            break
        b_center = betas[idx]
        b_left = betas[max(0, idx-1)]
        b_right = betas[min(n_coarse-1, idx+1)]
        
        # Ensure bounds are valid
        if b_left >= b_right:
            continue
            
        res = minimize_scalar(
            _chirp_metric, 
            bounds=(b_left, b_right), 
            method='bounded', 
            options={'xatol': 1e-4}
        )
        
        if res.success:
            b_opt = res.x
            val_opt = res.fun
            if val_opt < best_val:
                best_val = val_opt
                best_beta = b_opt
                best_x = _chirp_sequence(b_opt)
                
    return best_x, best_val

def _try_mixed_chirp_optimization(deadline, rng):
    """
    Try a mixed signal: x_n = cos(2*pi*beta*n^2/N) + alpha * noise
    Or perhaps just scale the chirp? No, scale cancels out in metric.
    Let's try adding a small linear phase component or amplitude modulation?
    The task asks for "signed step heights".
    
    Let's try a different parametric family:
    x_n = cos(2*pi*beta*n^2/N + phi)
    """
    best_val = float('inf')
    best_x = None
    
    def _metric_2d(theta):
        beta, phi = theta
        n = np.arange(N)
        x = np.cos(2 * np.pi * beta * n * n / N + phi)
        return _metric(x)
    
    # Start from the best chirp found
    # We need to pass in the best beta from previous step? 
    # Let's just run a global optimization on (beta, phi)
    
    # Since beta is periodic, we restrict to [0, N/2]
    # phi is [0, 2*pi]
    
    # Use differential evolution or just a grid search + local refine?
    # Grid search is safer within time limits.
    
    n_beta = 50
    n_phi = 20
    betas = np.linspace(0.1, N/2 - 0.1, n_beta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    for b in betas:
        if time.time() > deadline - 5:
            break
        for p in phis:
            val = _metric_2d([b, p])
            if val < best_val:
                best_val = val
                best_x = np.cos(2 * np.pi * b * np.arange(N)**2 / N + p)
                
    # Refine best point
    if best_x is not None:
        # We need to know which b, p it was. 
        # Let's just redo the best one found in the loop above properly?
        # Actually, the loop above already finds the best.
        # We can do a local refinement around the best (b, p).
        # But we lost track of the exact b, p in the loop above easily.
        # Let's just return the best found.
        pass
        
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0
    
    # Phase 1: Pure Chirp Optimization
    x_chirp, val_chirp = _try_chirp_optimization(deadline - 30)
    
    best_x = x_chirp
    best_val = val_chirp
    
    # Phase 2: Mixed Chirp (Chirp + Phase)
    if time.time() < deadline - 10:
        rng = np.random.default_rng(2024)
        x_mixed, val_mixed = _try_mixed_chirp_optimization(deadline - 10, rng)
        if x_mixed is not None and val_mixed < best_val:
            best_x = x_mixed
            best_val = val_mixed
            
    # Phase 3: Local Search Refinement (optional, if time permits)
    # Given the parametric nature, the solutions are already quite good.
    # A simple local search on the time domain might help if the chirp isn't perfect.
    if time.time() < deadline - 2 and best_x is not None:
        # Simple local search
        x_curr = best_x.copy()
        v_curr = best_val
        step = 0.1
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            idx = np.random.randint(0, N)
            x_test = x_curr.copy()
            x_test[idx] += np.random.normal(0, step)
            v_test = _metric(x_test)
            if v_test < v_curr:
                x_curr = x_test
                v_curr = v_test
                step *= 0.9
            else:
                step *= 0.95
        if v_curr < best_val:
            best_x = x_curr
            best_val = v_curr

    if best_x is None:
        return np.ones(N)
        
    return best_x
# EVOLVE-BLOCK-END
