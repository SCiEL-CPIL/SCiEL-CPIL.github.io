"""
Zadoff-Chu chirp parameterization + smooth optimization polish.

Key insight: Zadoff-Chu sequences have constant amplitude and zero periodic autocorrelation
in the complex domain. By taking the real part of a chirped Zadoff-Chu sequence with
continuously optimized phase parameters, we exploit the inherent low-autocorrelation
structure to minimize off-diagonal autoconvolution peaks.

Strategy:
1. Generate candidates using 1D chirp parameter search (fast, avoids high-dim local minima).
2. Use smooth objective (soft-max) + L-BFGS-B to refine top candidates.
3. Final gradient descent polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    if len(c) > 1:
        max_val = np.max(np.abs(c[1:]))
    else:
        max_val = 0.0
    return max_val / (s * s)

def _objective_smooth(x, temp=1.0):
    """Smooth approximation of the off-diagonal metric for optimization."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv(x)
    if len(c) > 1:
        abs_c = np.abs(c[1:])
    else:
        return 0.0
    max_c = np.max(abs_c)
    if max_c < 1e-15:
        return 0.0
    soft_max = max_c + temp * np.log(np.sum(np.exp((abs_c - max_c) / temp)))
    return soft_max / (s * s)

def _chirp_sequence(beta, gamma=0.0, delta=0.0):
    """
    Generate a real-valued sequence from a generalized Zadoff-Chu chirp.
    
    Parameters:
    - beta: chirp rate parameter (primary optimization variable)
    - gamma: quadratic phase offset
    - delta: linear phase offset
    
    The complex sequence is z[n] = exp(i * (pi * beta * n^2 / N + gamma * n + delta))
    We return the real part, which has the structure of a cosine wave with chirped frequency.
    """
    n = np.arange(N, dtype=float)
    # Use a chirp that wraps appropriately for N-point sequences
    phase = np.pi * beta * (n ** 2) / N + gamma * n + delta
    z = np.exp(1j * phase)
    # Take real part for a real-valued sequence
    x = np.real(z)
    return x

def _chirp_objective(params):
    """
    Objective function for chirp parameter optimization.
    params: [beta, gamma, delta]
    """
    beta, gamma, delta = params
    x = _chirp_sequence(beta, gamma, delta)
    return _metric_offdiag(x)

def _search_chirp_parameters(deadline=None, n_grid=200):
    """
    1D grid search over chirp parameter beta, with secondary optimization of gamma, delta.
    Returns best sequence and its metric value.
    """
    best_x = None
    best_val = float('inf')
    
    # Grid search over beta
    beta_values = np.linspace(-2.0, 2.0, n_grid)
    
    for beta in beta_values:
        if deadline is not None and time.time() > deadline:
            break
        
        # For each beta, do a quick 2D search over gamma, delta
        # Use a small grid for speed
        gamma_vals = np.linspace(-np.pi, np.pi, 10)
        delta_vals = np.linspace(-np.pi, np.pi, 10)
        
        for gamma in gamma_vals:
            for delta in delta_vals:
                if deadline is not None and time.time() > deadline:
                    break
                x = _chirp_sequence(beta, gamma, delta)
                val = _metric_offdiag(x)
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
    
    return best_x, best_val

def _gradient_descent_polish(x0, max_iter=500, initial_lr=0.01, deadline=None):
    """Projected gradient descent with annealed temperature for final polish."""
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    # Normalize sum to 1
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    lr = initial_lr
    temp = 0.5
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        c = _autoconv(x)
        if len(c) <= 1:
            break
        abs_c = np.abs(c[1:])
        
        max_c = np.max(abs_c)
        if max_c < 1e-15:
            break
        
        exp_vals = np.exp((abs_c - max_c) / temp)
        sum_exp = np.sum(exp_vals)
        soft_max = max_c + temp * np.log(sum_exp)
        
        grad_abs_c = exp_vals / sum_exp
        c_off = c[1:]
        sign_c = np.sign(c_off)
        grad_c = grad_abs_c * sign_c
        
        w = np.zeros(2 * N - 1)
        w[1:] = grad_c
        
        fsize = 1
        while fsize < 2 * N:
            fsize *= 2
        X = np.fft.rfft(x, fsize)
        W = np.fft.rfft(w, fsize)
        grad_x = np.fft.irfft(W * np.conj(X), fsize)[:N]
        
        s = np.sum(x)
        metric = soft_max / (s * s)
        
        grad_metric = grad_x / (s * s) - 2.0 * soft_max * np.ones(N) / (s ** 3)
        
        x_new = x - lr * grad_metric
        
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x + 1e-6 * np.random.randn(N)
            s_new = np.sum(x_new)
        x_new = x_new / s_new
        
        val = _metric_offdiag(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        x = x_new
        lr *= 0.995
        temp *= 0.999
        
    return best_x

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. Chirp parameter search (primary strategy)
    if time.time() < deadline - 80:
        x_chirp, val_chirp = _search_chirp_parameters(deadline=deadline - 80, n_grid=100)
        if x_chirp is not None and val_chirp < best_val:
            best_val = val_chirp
            best_x = x_chirp.copy()
        if x_chirp is not None:
            candidates.append(x_chirp)
    
    # 3. Fine-grained chirp search around best beta
    if time.time() < deadline - 60:
        # Do a finer grid around the region that gave best results
        for beta_center in np.linspace(-1.5, 1.5, 8):
            if time.time() > deadline - 60:
                break
            beta_fine = np.linspace(beta_center - 0.1, beta_center + 0.1, 20)
            for beta in beta_fine:
                if time.time() > deadline - 60:
                    break
                for gamma in np.linspace(-np.pi, np.pi, 5):
                    for delta in np.linspace(-np.pi, np.pi, 5):
                        x = _chirp_sequence(beta, gamma, delta)
                        val = _metric_offdiag(x)
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            candidates.append(x)
    
    # 4. Random binary sequences with non-zero mean
    for _ in range(5):
        if time.time() > deadline - 30:
            break
        r = rng.random(N)
        x = np.where(r > 0.35, 1.0, -1.0)
        if abs(np.sum(x)) > 5:
            candidates.append(x)
    
    # 5. Block patterns
    for block_size in [10, 20, 50]:
        if time.time() > deadline - 20:
            break
        pattern = np.zeros(N)
        for i in range(N):
            if (i // block_size) % 2 == 0:
                pattern[i] = 1.0
            else:
                pattern[i] = -1.0
        pattern[0] += 0.5
        candidates.append(pattern)
    
    # Optimize top candidates with L-BFGS-B on smooth objective
    # Sort by current metric to prioritize best
    scored = [(c, _metric_offdiag(c)) for c in candidates]
    scored.sort(key=lambda t: t[1])
    
    for x0, cur_val in scored[:8]:
        if time.time() > deadline - 25:
            break
        
        x0 = x0.copy()
        s0 = np.sum(x0)
        if abs(s0) < 1e-6:
            continue
        x0 = x0 / s0
        
        try:
            # First pass with higher temperature
            res1 = minimize(
                lambda x: _objective_smooth(x, temp=1.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 60, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_mid = res1.x
            
            # Second pass with lower temperature
            res2 = minimize(
                lambda x: _objective_smooth(x, temp=0.1),
                x_mid,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_opt = res2.x
            
            val = _metric_offdiag(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            polished = _gradient_descent_polish(
                best_x,
                max_iter=600,
                initial_lr=0.005,
                deadline=deadline - 5
            )
            val = _metric_offdiag(polished)
            if val < best_val:
                best_val = val
                best_x = polished
        except Exception:
            pass
    
    return best_x
