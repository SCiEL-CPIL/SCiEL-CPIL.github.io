"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _value(h):
    """Compute the evaluation metric."""
    s = np.sum(h)
    if s == 0:
        return np.inf
    ac = np.convolve(h, h)
    return abs(2 * N * np.max(ac) / (s ** 2))

def _objective(x):
    """
    Objective: minimize the peak-to-average autocorrelation ratio.
    We use a smooth surrogate via log-sum-exp for the max operation.
    x is the parameter vector for a smooth basis.
    """
    # Reconstruct h from parameters
    h = _reconstruct(x)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10  # Penalize zero sum
    
    # Compute autocorrelation
    ac = np.convolve(h, h)
    
    # Smooth max using log-sum-exp
    # Scale for numerical stability
    ac_scaled = ac / 100.0
    max_ac_scaled = np.max(ac_scaled)
    log_sum_exp = max_ac_scaled + np.log(np.sum(np.exp(ac_scaled - max_ac_scaled)))
    max_ac_smooth = log_sum_exp * 100.0
    
    return abs(2 * N * max_ac_smooth / (s ** 2))

def _reconstruct(x):
    """
    Reconstruct h from parameter vector x.
    Use a low-dimensional smooth basis: Fourier cosine basis.
    x = [a0, a1, ..., a_{K-1}, b1, ..., b_{K-1}]
    h[n] = a0 + sum_{k=1}^{K-1} (a_k * cos(2*pi*k*n/N) + b_k * sin(2*pi*k*n/N))
    """
    K = 20  # Number of basis functions
    n = np.arange(N)
    h = np.zeros(N)
    
    # DC component
    h += x[0]
    
    # Cosine and sine terms
    for k in range(1, K):
        h += x[k] * np.cos(2 * np.pi * k * n / N)
        h += x[K + k - 1] * np.sin(2 * np.pi * k * n / N)
    
    return h

def _gradient(x):
    """
    Numerical gradient computation for the smooth objective.
    """
    eps = 1e-7
    grad = np.zeros_like(x)
    f0 = _objective(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (_objective(x_plus) - f0) / eps
    return grad

def construct():
    """
    Construct the optimal step heights using optimization.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    K = 20  # Number of basis functions
    dim = 2 * K  # Dimension of parameter space
    
    # Multiple random starts
    best_x = None
    best_val = np.inf
    
    # Seed start: all ones corresponds to a0 = 1, others = 0
    x_seed = np.zeros(dim)
    x_seed[0] = 1.0
    
    starts = [x_seed]
    
    # Add random starts
    np.random.seed(42)
    for _ in range(10):
        x_rand = np.random.randn(dim) * 0.5
        x_rand[0] += 1.0  # Bias toward positive DC
        starts.append(x_rand)
    
    for x0 in starts:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_val:
                h_candidate = _reconstruct(res.x)
                val = _value(h_candidate)
                if val < best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if best_x is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                _objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_final.fun < best_val:
                best_x = res_final.x
                best_val = _value(_reconstruct(best_x))
        except Exception:
            pass
    
    if best_x is not None:
        return _reconstruct(best_x)
    
    # Fallback to seed
    return np.ones(N)

# EVOLVE-BLOCK-END
