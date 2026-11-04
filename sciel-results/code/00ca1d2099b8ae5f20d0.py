"""Sparse polynomial representation with differential evolution for signed autocorrelation minimization."""
import numpy as np
from scipy.optimize import differential_evolution
import time

N = 400
DEADLINE = 140.0  # Safety margin before 150s kill


def _autoconv_full(h):
    """Compute full linear autoconvolution of h using FFT."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, n=size)
    c = np.fft.irfft(f * f.conj(), n=size)
    return c


def _metric(h):
    """Compute max|c_k| / S^2."""
    S = np.sum(h)
    if abs(S) < 1e-12:
        return np.inf
    c = _autoconv_full(h)
    return np.max(np.abs(c)) / (S * S)


def _sparse_to_dense(positions, magnitudes, N):
    """Convert sparse representation to dense array."""
    h = np.zeros(N, dtype=np.float64)
    for pos, mag in zip(positions, magnitudes):
        idx = int(round(pos)) % N
        h[idx] += mag
    return h


def construct():
    """
    Returns signed step heights that minimize the maximum absolute autoconvolution
    divided by squared integral.
    
    Uses sparse polynomial representation with differential evolution to find
    signals with spectral nulls that minimize autoconvolution peaks.
    """
    t0 = time.time()
    
    # Baseline: constant sequence
    h_const = np.full(N, 1.0, dtype=np.float64)
    best_h = h_const.copy()
    best_metric = _metric(h_const)
    
    # Sparse representation: K non-zero terms
    # Each term has a position (0 to N-1) and a magnitude (-1 to 1)
    K = 20  # Number of non-zero terms
    
    def _sparse_objective(params):
        """Objective function for differential evolution on sparse params."""
        positions = params[:K]
        magnitudes = params[K:]
        
        # Ensure sum is non-zero by adding a small constant if needed
        h = _sparse_to_dense(positions, magnitudes, N)
        S = np.sum(h)
        if abs(S) < 0.1:
            # Add small bias to ensure non-zero sum
            h[0] += 0.1
            S = np.sum(h)
            if abs(S) < 1e-12:
                return 1e6
        
        c = _autoconv_full(h)
        return np.max(np.abs(c)) / (S * S)
    
    # Differential evolution bounds
    bounds = [(0, N - 1)] * K + [(-1.0, 1.0)] * K
    
    # Run differential evolution with time limit
    try:
        def _timeout_callback(xk, funcval, convergence):
            return (time.time() - t0) > DEADLINE - 10
        
        result = differential_evolution(
            _sparse_objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            callback=_timeout_callback,
            init='sobol'
        )
        
        if result.success or result.fun < best_metric:
            h_de = _sparse_to_dense(result.x[:K], result.x[K:], N)
            m_de = _metric(h_de)
            if m_de < best_metric:
                best_metric = m_de
                best_h = h_de
    except Exception:
        pass
    
    # Try different K values if time permits
    for K_try in [10, 30, 50]:
        if (time.time() - t0) > DEADLINE - 20:
            break
        
        def _sparse_objective_k(params, K=K_try):
            positions = params[:K]
            magnitudes = params[K:]
            h = _sparse_to_dense(positions, magnitudes, N)
            S = np.sum(h)
            if abs(S) < 0.1:
                h[0] += 0.1
                S = np.sum(h)
                if abs(S) < 1e-12:
                    return 1e6
            c = _autoconv_full(h)
            return np.max(np.abs(c)) / (S * S)
        
        bounds_k = [(0, N - 1)] * K_try + [(-1.0, 1.0)] * K_try
        
        try:
            result_k = differential_evolution(
                _sparse_objective_k,
                bounds_k,
                maxiter=50,
                popsize=10,
                tol=1e-5,
                seed=42 + K_try,
                polish=False,
                callback=_timeout_callback,
                init='sobol'
            )
            
            if result_k.success or result_k.fun < best_metric:
                h_k = _sparse_to_dense(result_k.x[:K_try], result_k.x[K_try:], N)
                m_k = _metric(h_k)
                if m_k < best_metric:
                    best_metric = m_k
                    best_h = h_k
        except Exception:
            pass
    
    # Final polish with local search if time permits
    if (time.time() - t0) < DEADLINE - 30:
        def local_search_polish(h, max_time=20.0):
            t0_ls = time.time()
            current = h.copy()
            current_metric = _metric(current)
            improved = True
            while improved and (time.time() - t0_ls) < max_time and (time.time() - t0) < DEADLINE - 5:
                improved = False
                n = len(current)
                for i in range(n):
                    old_val = current[i]
                    current[i] = -old_val
                    new_metric = _metric(current)
                    if new_metric < current_metric:
                        current_metric = new_metric
                        improved = True
                    else:
                        current[i] = old_val
            return current, current_metric
        
        h_polished, m_polished = local_search_polish(best_h, max_time=25.0)
        if m_polished < best_metric:
            best_metric = m_polished
            best_h = h_polished
    
    return best_h
