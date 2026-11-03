"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _autoconv_sup_norm(f, smooth_eps=1e-6):
    """Compute a smooth approximation of the sup norm of autoconvolution."""
    # Compute autoconvolution
    n = len(f)
    conv = np.convolve(f, f, mode='full')
    # Smooth max using log-sum-exp
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0
    # Use scaled log-sum-exp for numerical stability
    scaled = conv / smooth_eps
    log_sum_exp = smooth_eps * (np.log(np.sum(np.exp(scaled - np.max(scaled)))) + np.max(scaled))
    return log_sum_exp

def _objective(f, smooth_eps=1e-6):
    """Negative of the ratio we want to maximize."""
    f = np.maximum(f, 0)  # enforce nonnegativity
    l1 = np.sum(f)
    if l1 < 1e-12:
        return 1e12
    l2_sq = np.sum(f**2)
    conv_sup = _autoconv_sup_norm(f, smooth_eps)
    if conv_sup < 1e-12:
        return 1e12
    ratio = l2_sq / (l1 * conv_sup)
    return -ratio

def _objective_exact(f):
    """Exact objective for final evaluation."""
    f = np.maximum(f, 0)
    l1 = np.sum(f)
    if l1 < 1e-12:
        return 1e12
    l2_sq = np.sum(f**2)
    conv = np.convolve(f, f, mode='full')
    conv_sup = np.max(conv)
    if conv_sup < 1e-12:
        return 1e12
    ratio = l2_sq / (l1 * conv_sup)
    return -ratio

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_f = np.ones(N) / N  # Start with uniform
    best_val = -_objective_exact(best_f)
    
    # Try several initializations
    rng = np.random.default_rng(42)
    
    # Different initialization strategies
    inits = []
    
    # Uniform
    inits.append(np.ones(N) / N)
    
    # Sparse: few nonzero elements
    for k in [1, 2, 3, 5, 10]:
        idx = rng.choice(N, size=k, replace=False)
        f = np.zeros(N)
        f[idx] = 1.0 / k
        inits.append(f)
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    for sigma in [0.1, 0.3, 0.5, 1.0]:
        f = np.exp(-x**2 / (2 * sigma**2))
        f = np.maximum(f, 0)
        f = f / np.sum(f)
        inits.append(f)
    
    # Random sparse
    for _ in range(10):
        f = rng.random(N)
        # Make it sparse
        mask = rng.random(N) < 0.3
        f = f * mask
        if np.sum(f) < 1e-12:
            f = np.ones(N) / N
        else:
            f = f / np.sum(f)
        inits.append(f)
    
    # Random dense
    for _ in range(10):
        f = rng.random(N)
        f = f / np.sum(f)
        inits.append(f)
    
    # Triangle / tent functions
    for center in [N//4, N//2, 3*N//4]:
        f = np.zeros(N)
        for i in range(N):
            f[i] = max(0, 1 - abs(i - center) / (N/4))
        if np.sum(f) > 0:
            f = f / np.sum(f)
            inits.append(f)
    
    # Cosine bump
    x = np.linspace(-1, 1, N)
    f = (1 + np.cos(np.pi * x)) / 2
    f = np.maximum(f, 0)
    f = f / np.sum(f)
    inits.append(f)
    
    # Optimize from each initialization
    for init in inits:
        if time.time() > deadline:
            break
        
        # Use L-BFGS-B with bounds [0, 1]
        try:
            result = minimize(
                _objective,
                init,
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            f_opt = np.maximum(result.x, 0)
            if np.sum(f_opt) > 1e-12:
                f_opt = f_opt / np.sum(f_opt)
            
            val = -_objective_exact(f_opt)
            if val > best_val:
                best_val = val
                best_f = f_opt.copy()
                
        except Exception:
            continue
    
    # Final polish: try a few more iterations on the best solution
    if time.time() < deadline - 10:
        for _ in range(3):
            try:
                result = minimize(
                    _objective,
                    best_f,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * N,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                f_opt = np.maximum(result.x, 0)
                if np.sum(f_opt) > 1e-12:
                    f_opt = f_opt / np.sum(f_opt)
                val = -_objective_exact(f_opt)
                if val > best_val:
                    best_val = val
                    best_f = f_opt.copy()
            except Exception:
                break
    
    # Convert to list of floats
    return [float(x) for x in best_f]
# EVOLVE-BLOCK-END
