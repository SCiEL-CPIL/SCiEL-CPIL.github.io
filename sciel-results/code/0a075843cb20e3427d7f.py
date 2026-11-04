import numpy as np
from scipy.optimize import minimize
import time

N = 400
_DEADLINE = 150.0
_START_TIME = None

def _objective(x):
    """
    Compute the value to minimize: 2 * N * max(conv(h,h)) / sum(h)^2
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    # np.convolve with mode='full' is efficient
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    # The value is defined as abs(...), but since max(conv) can be negative, 
    # we take the absolute value of the whole expression as per task description.
    # However, for optimization, we want to minimize the magnitude.
    # Note: The task says "value = abs(2 * N * np.max(...) / np.sum(h)**2)"
    # If max(conv) is negative, the value is positive.
    # We want to minimize this value.
    val = abs(2 * N * m / (s * s))
    return val

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_val = 2.0  # Seed value
    best_h = np.ones(N)
    
    deadline = _START_TIME + _DEADLINE - 5.0  # 5 second safety margin
    
    # Generate diverse initializations
    inits = []
    
    # 1. Random initializations
    for seed in range(20):
        r = np.random.RandomState(seed)
        h = r.randn(N)
        # Ensure sum is not zero
        if abs(np.sum(h)) < 1e-6:
            h[0] += 1.0
        inits.append(h)
        
    # 2. Structured initializations
    # Linear ramps
    inits.append(np.linspace(-1, 1, N))
    inits.append(np.linspace(1, -1, N))
    
    # Step functions
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        h = np.zeros(N)
        h[:int(frac*N)] = 1.0
        h[int(frac*N):] = -1.0
        inits.append(h)
        
    # Waves
    t = np.linspace(0, 4*np.pi, N)
    inits.append(np.cos(t))
    inits.append(np.sin(t))
    inits.append(np.cos(2*t))
    inits.append(np.sin(3*t))
    
    # Polynomials
    x = np.linspace(-1, 1, N)
    inits.append(x**2 - 1/3)
    inits.append(x**3)
    inits.append(x**4 - 1/5)
    
    # Sparse-like: mostly zero with some non-zeros
    for k in [10, 20, 50]:
        h = np.zeros(N)
        idx = np.random.choice(N, k, replace=False)
        h[idx] = np.random.choice([-1, 1], k)
        if abs(np.sum(h)) < 1e-6:
            h[0] += 1.0
        inits.append(h)

    # Optimize each initialization
    for init_h in inits:
        if time.time() > deadline:
            break
            
        h = init_h.astype(float).copy()
        
        # Quick check
        val = _objective(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        try:
            # Use Nelder-Mead as it is robust for non-smooth functions
            # max() is non-smooth, so gradient-based methods require approximation
            res = minimize(
                _objective,
                h,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_h = res.x.copy()
                
        except Exception:
            pass

    # Final refinement on the best found
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective,
                best_h,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if res.fun < best_val:
                best_val = res.fun
                best_h = res.x.copy()
        except Exception:
            pass
            
    return best_h
