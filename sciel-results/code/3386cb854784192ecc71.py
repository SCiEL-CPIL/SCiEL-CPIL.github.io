"""Program for the suite task; maximizes squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 5.0  # seconds before the deadline we must stop

def _score(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0:
        return 0.0
    # Autoconvolution (linear convolution of h with itself)
    conv = np.convolve(h, h, mode='full')
    sup_norm = float(np.max(conv))
    if sup_norm <= 0:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def _objective(x):
    """Minimize negative score. x is a vector of log-heights to enforce nonnegativity."""
    h = np.exp(x)
    return -_score(h)

def construct():
    start_time = time.time()
    
    # Initial guess: uniform
    best_score = 0.0
    best_h = np.ones(N, dtype=np.float64)
    best_x = np.zeros(N, dtype=np.float64)
    
    # Try several initial conditions
    initial_guesses = []
    
    # 1. Uniform
    initial_guesses.append(np.zeros(N))
    
    # 2. Peak at center
    x2 = np.full(N, -10.0)
    x2[N // 2] = 0.0
    initial_guesses.append(x2)
    
    # 3. Triangle (linear rise and fall)
    idx = np.arange(N)
    center = (N - 1) / 2.0
    x3 = -np.abs(idx - center) / center * 5.0
    x3 -= np.max(x3)  # normalize so max is 0
    initial_guesses.append(x3)
    
    # 4. Gaussian-like
    x4 = -((idx - center) ** 2) / (2 * (N / 4) ** 2)
    x4 -= np.max(x4)
    initial_guesses.append(x4)
    
    # 5. Random with seed
    np.random.seed(42)
    for _ in range(5):
        x5 = np.random.randn(N) * 0.5
        x5 -= np.max(x5)
        initial_guesses.append(x5)
    
    # Optimize from each initial guess
    for i, x0 in enumerate(initial_guesses):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - SAFETY_MARGIN:
            break
        
        # Evaluate initial point
        h0 = np.exp(x0)
        s0 = _score(h0)
        if s0 > best_score:
            best_score = s0
            best_h = h0.copy()
            best_x = x0.copy()
        
        # Run L-BFGS-B optimization
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if result.success or result.fun < -best_score:
                h_res = np.exp(result.x)
                s_res = _score(h_res)
                if s_res > best_score:
                    best_score = s_res
                    best_h = h_res.copy()
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Final polish: use the best found as starting point for more precise optimization
    elapsed = time.time() - start_time
    if elapsed < DEADLINE_SECONDS - SAFETY_MARGIN and best_x is not None:
        try:
            result = minimize(
                _objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 100}
            )
            h_res = np.exp(result.x)
            s_res = _score(h_res)
            if s_res > best_score:
                best_score = s_res
                best_h = h_res.copy()
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Return as list of floats
    return [float(v) for v in best_h]
