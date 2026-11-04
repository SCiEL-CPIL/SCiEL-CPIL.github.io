"""Spectral design: minimize off-DC spectral energy to suppress autoconvolution lobes, then refine peak."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

# Precompute indices for real FFT
_M = _FFT_SIZE // 2 + 1


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


def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _soft_max_abs(v, sigma=0.1):
    """Smooth approximation of max(|v|)."""
    return float(np.logaddexp(np.log(np.sum(np.exp((v / sigma) ** 2))), np.log(np.sum(np.exp((-v / sigma) ** 2))))) * sigma / 2


def _smooth_metric(x, sigma=0.05):
    """Smooth surrogate for the true metric."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    conv = _autoconv_values(x)
    # Use soft max of absolute values
    soft_max = _soft_max_abs(conv, sigma)
    return soft_max / (s * s)


def _epigraph_objective(theta):
    """
    Objective for epigraph formulation.
    theta = [x (N), t]
    Minimize t + lambda * ||x||^2 (regularization to keep x bounded)
    Subject to: |conv[k]| <= t * (sum(x))^2 for all k
    We use a soft penalty for the constraints.
    """
    x = theta[:N]
    t = theta[N]
    
    s = _integral(x)
    if abs(s) < 1e-6:
        return 1e10 + t
    
    conv = _autoconv_values(x)
    # Penalty for violating |conv[k]| <= t * s^2
    violation = np.maximum(0, np.abs(conv) - t * s * s)
    penalty = np.sum(violation ** 2)
    
    # Regularization
    reg = 1e-4 * np.sum(x ** 2)
    
    return t + penalty + reg


def _epigraph_gradient(theta):
    """Gradient of epigraph objective via finite differences (slow but reliable)."""
    eps = 1e-5
    grad = np.zeros_like(theta)
    f0 = _epigraph_objective(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        grad[i] = (_epigraph_objective(theta_plus) - f0) / eps
    return grad


def _spectral_design_optimize(x_init, deadline, rng):
    """
    Optimize the spectral magnitudes to minimize off-DC energy,
    then use the resulting phase structure as a starting point
    for direct optimization of the autoconvolution peak.
    
    Key insight: minimizing sum_{k>=1} |FX[k]|^4 (or equivalently
    minimizing the L4 norm of the spectrum) leads to flat spectra,
    which in turn leads to low autoconvolution sidelobes.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # Strategy 1: Direct optimization of a smoothed version of the metric
    # using scipy.optimize.minimize with BFGS
    
    x0 = x_init.copy()
    s0 = _integral(x0)
    if abs(s0) > 1e-12:
        x0 = x0 / s0  # Normalize to sum=1
    
    def _obj(x):
        s = _integral(x)
        if abs(s) < 1e-12:
            return 1e10
        conv = _autoconv_values(x)
        # Smooth max using log-sum-exp
        sigma = 0.02
        max_abs = _soft_max_abs(conv, sigma)
        return max_abs / (s * s)
    
    def _grad(x):
        # Numerical gradient
        eps = 1e-6
        g = np.zeros(N)
        f0 = _obj(x)
        for i in range(N):
            x_plus = x.copy()
            x_plus[i] += eps
            g[i] = (_obj(x_plus) - f0) / eps
        return g
    
    # Run a few optimization rounds with different initializations
    n_rounds = 5
    for r in range(n_rounds):
        if time.time() > deadline:
            break
        
        if r == 0:
            x_start = x0.copy()
        else:
            # Perturb the best solution found so far
            x_start = best_x.copy()
            s_b = _integral(x_start)
            if abs(s_b) > 1e-12:
                x_start = x_start / s_b
            x_start += rng.normal(0, 0.01, size=N)
        
        # Use L-BFGS-B with a time-limited evaluation
        result = minimize(
            _obj,
            x_start,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        x_opt = result.x
        s_opt = _integral(x_opt)
        if abs(s_opt) > 1e-12:
            x_opt = x_opt / s_opt
        
        val_opt = _metric(x_opt)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt.copy()
    
    return best_x, best_val


def _try_local_search_fast(x, deadline, rng, max_rounds=100000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.2
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 150:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        k = int(rng.integers(1, 5))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(2):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.03, 0.03
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(8):
                if time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x1 = best_x.copy()
                x1[idx] += m1
                v1 = _metric(x1)
                
                x2 = best_x.copy()
                x2[idx] += m2
                v2 = _metric(x2)
                
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
                
                if v1 < best_v:
                    best_v = v1
                    best_delta = m1
                if v2 < best_v:
                    best_v = v2
                    best_delta = m2
            
            if best_delta != 0.0 and best_v < best_val:
                best_x[idx] += best_delta
                best_val = best_v
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0

    # Start with a structured initial guess
    best_x = np.ones(N)
    best_val = _metric(best_x)

    rng = np.random.default_rng(2024)

    # Try several structured initializations
    inits = [
        np.ones(N),
        np.cos(np.linspace(0, 2 * np.pi, N, endpoint=False)),
        np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False)),
        np.ones(N) + 0.1 * np.random.default_rng(42).normal(size=N),
    ]

    for init in inits:
        if time.time() > deadline - 10:
            break
        v = _metric(init)
        if v < best_val:
            best_val = v
            best_x = init.copy()

    # Phase 1: Spectral design optimization (new strategy)
    if time.time() < deadline - 60:
        x_spectral, val_spectral = _spectral_design_optimize(best_x, deadline - 60, rng)
        if val_spectral < best_val:
            best_val = val_spectral
            best_x = x_spectral

    # Phase 2: Local search
    if time.time() < deadline - 20:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 20, rng, max_rounds=50000)

    # Phase 3: Final polish
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)

    return best_x
# EVOLVE-BLOCK-END
