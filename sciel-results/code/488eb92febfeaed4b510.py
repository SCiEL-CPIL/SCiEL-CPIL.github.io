"""
Strategy: Sparse-spike parameterization with L-BFGS-B optimization.
We model h as a sum of Gaussian spikes to allow for sharper features
and optimize the spike parameters directly.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import rfft, irfft

N = 400
_FFT_N = 2 * N - 1
_X_GRID = np.linspace(-0.25, 0.25, N, endpoint=False)

# EVOLVE-BLOCK-START

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    H = rfft(h, n=_FFT_N)
    c = irfft(H * H, n=_FFT_N)
    return c

def _soft_max(x, beta):
    """Differentiable approximation to max using log-sum-exp."""
    m = np.max(x)
    x_shifted = x - m
    exp_vals = np.exp(beta * x_shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return m
    return m + (1.0 / beta) * np.log(sum_exp)

def _evaluate_exact(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _sparse_to_h(params, n_spikes):
    """
    Convert sparse parameters to full height vector.
    params layout: [baseline (1), amplitudes (n_spikes), centers (n_spikes), widths (n_spikes)]
    """
    baseline = params[0]
    amps = params[1:1+n_spikes]
    centers = params[1+n_spikes:1+2*n_spikes]
    widths = params[1+2*n_spikes:1+3*n_spikes]
    
    # Ensure widths are positive
    widths = np.abs(widths) + 1e-6
    
    h = np.full(N, baseline)
    for i in range(n_spikes):
        h += amps[i] * np.exp(-0.5 * ((_X_GRID - centers[i]) / widths[i]) ** 2)
        
    return h

def _objective_sparse(params, n_spikes, beta):
    """
    Objective function for sparse parameter optimization.
    """
    h = _sparse_to_h(params, n_spikes)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    val = 2.0 * N * sm / (s * s)
    return val

def _coordinate_descent_polish(h, max_iters=2000, step_sizes=None, deadline=None):
    """
    Adaptive coordinate descent on exact objective.
    """
    if step_sizes is None:
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    best_h = h.copy()
    best_val = _evaluate_exact(best_h)
    rng = np.random.default_rng(123)
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for _ in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            idx = rng.integers(0, N)
            for delta in step_sizes:
                found = False
                for sign in [1, -1]:
                    h_new = best_h.copy()
                    h_new[idx] += sign * delta
                    s_new = np.sum(h_new)
                    if abs(s_new) < 1e-10:
                        continue
                    val_new = _evaluate_exact(h_new)
                    if val_new < best_val - 1e-14:
                        best_val = val_new
                        best_h = h_new
                        improved = True
                        found = True
                        break
                if found:
                    break
        if not improved:
            step_sizes = [s * 0.5 for s in step_sizes]
            if step_sizes[0] < 1e-7:
                break
    
    return best_h, best_val

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin: evaluator kills at 150s

    rng = np.random.default_rng(42)

    best_h = np.ones(N)
    best_val = _evaluate_exact(best_h)  # should be 2.0

    def try_h(h):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        val = _evaluate_exact(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        return val

    # === Phase 1: Sparse Spike Optimization ===
    # We try different numbers of spikes and different initializations
    n_spikes_list = [10, 20, 30]
    betas = [50.0, 100.0, 200.0]
    
    for n_spikes in n_spikes_list:
        if time.time() > deadline - 20.0:
            break
            
        # Number of parameters: 1 (baseline) + 3 * n_spikes
        n_params = 1 + 3 * n_spikes
        
        # Try multiple random restarts
        for restart in range(5):
            if time.time() > deadline - 20.0:
                break
                
            # Initialize parameters
            # Baseline: positive to ensure sum > 0 initially
            baseline = rng.uniform(0.5, 2.0)
            amps = rng.standard_normal(n_spikes) * rng.uniform(1.0, 3.0)
            centers = rng.uniform(-0.25, 0.25, n_spikes)
            widths = rng.uniform(0.02, 0.1, n_spikes)
            
            x0 = np.concatenate([[baseline], amps, centers, widths])
            
            for beta in betas:
                if time.time() > deadline - 20.0:
                    break
                    
                try:
                    res = minimize(
                        lambda x: _objective_sparse(x, n_spikes, beta),
                        x0,
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    h_opt = _sparse_to_h(res.x, n_spikes)
                    val_opt = _evaluate_exact(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass

    # === Phase 2: Coordinate Descent Polish ===
    # Refine the best sparse solution with coordinate descent on the full vector
    if time.time() < deadline - 5.0:
        best_h, best_val = _coordinate_descent_polish(
            best_h,
            max_iters=2000,
            step_sizes=[0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
            deadline=deadline - 5.0
        )

    # Safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
