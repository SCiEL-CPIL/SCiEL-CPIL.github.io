"""Combined approach: Chebyshev basis + differential evolution for autocorrelation minimization."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_chebyt

N = 400
_t0 = time.time()
_DEADLINE = 140.0  # safety margin under 150s limit

def _value(h):
    """Compute the objective value (lower is better)."""
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))


def _cheb_basis(k, n_samples=N):
    """
    Generate Chebyshev polynomial basis vectors on [-1, 1].
    T_0 = 1, T_1 = x, T_k = 2x*T_{k-1} - T_{k-2}
    """
    x = np.linspace(-1, 1, n_samples)
    basis = np.zeros((k, n_samples))
    basis[0] = 1.0
    if k > 1:
        basis[1] = x
    for j in range(2, k):
        basis[j] = 2.0 * x * basis[j-1] - basis[j-2]
    return basis


def _make_h_from_cheb(coeffs, K):
    """
    Construct h as linear combination of Chebyshev polynomials.
    h[n] = sum_{j=0}^{K-1} coeffs[j] * T_j(x_n)
    where x_n = linspace(-1, 1, N)
    """
    x = np.linspace(-1, 1, N)
    h = np.zeros(N)
    # T_0 = 1
    h += coeffs[0] * 1.0
    if K > 1:
        h += coeffs[1] * x
    for j in range(2, K):
        # Compute T_j recursively for efficiency
        pass
    
    # More efficient: precompute basis
    basis = _cheb_basis(K, N)
    h = basis.T @ coeffs
    return h


def _obj_cheb(coeffs, K):
    """Objective function for Chebyshev coefficient optimization."""
    h = _make_h_from_cheb(coeffs, K)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12
    c = np.convolve(h, h)
    max_c = np.max(c)
    val = 2.0 * N * max_c / (s * s)
    return abs(val)


def construct():
    """Construct step heights minimizing the objective value."""
    start = time.time()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    def _update(h):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Precompute Chebyshev basis for various K values
    # Try different numbers of basis functions
    for K in [3, 5, 7, 10]:
        if time.time() - start > _DEADLINE:
            break
        
        # Initial guess: constant (T_0 only)
        init_coeffs = np.zeros(K)
        init_coeffs[0] = 1.0  # constant term
        
        # Bounds for coefficients
        # T_0 coefficient should be dominant to keep sum(h) nonzero
        bounds = [(0.1, 10.0)] + [(-5.0, 5.0)] * (K - 1)
        
        dim = K
        # Differential evolution parameters
        maxiter = max(50, int((_DEADLINE - (time.time() - start)) * 2))
        popsize = max(10, min(50, 30 - K))
        
        try:
            res = differential_evolution(
                _obj_cheb, bounds, args=(K,),
                maxiter=maxiter, popsize=popsize,
                tol=0.01, mutation=(0.5, 1.0), recombination=0.7,
                seed=42, polish=False, init='sobol' if dim <= 20 else 'latinhypercube'
            )
            if res.x is not None:
                h = _make_h_from_cheb(res.x, K)
                _update(h)
                
                # Local polish
                if time.time() - start < _DEADLINE - 10:
                    try:
                        r2 = minimize(_obj_cheb, res.x, args=(K,), method='Nelder-Mead',
                                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                        if r2.x is not None:
                            h2 = _make_h_from_cheb(r2.x, K)
                            _update(h2)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Also try a direct low-dimensional Fourier approach as backup
    if time.time() - start < _DEADLINE - 5:
        n = np.arange(N)
        def _fourier_h(coeffs, M):
            h = np.full(N, 1.0)
            for j in range(1, M + 1):
                a = coeffs[2*(j-1)]
                b = coeffs[2*(j-1)+1]
                h += a * np.cos(2*np.pi*j*n/N) + b * np.sin(2*np.pi*j*n/N)
            return h
        
        def _fourier_obj(coeffs, M):
            h = _fourier_h(coeffs, M)
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e12
            c = np.convolve(h, h)
            return abs(2.0 * N * np.max(c) / (s * s))
        
        for M in [2, 4, 6]:
            if time.time() - start > _DEADLINE:
                break
            dim = 2 * M
            bounds = [(-1.0, 1.0)] * dim
            try:
                res_f = differential_evolution(
                    _fourier_obj, bounds, args=(M,),
                    maxiter=100, popsize=20,
                    tol=0.01, seed=123, polish=False
                )
                if res_f.x is not None:
                    h_f = _fourier_h(res_f.x, M)
                    _update(h_f)
            except Exception:
                pass
    
    return best_h
