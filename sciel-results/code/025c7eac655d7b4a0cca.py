import numpy as np
from scipy.optimize import minimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def objective(h):
        s = np.sum(h)
        if s < 1e-6: return 1e9
        size = 2*N
        H = np.fft.fft(h, size=size)
        conv = np.fft.ifft(H * H).real
        max_val = np.max(conv)
        return max_val / (s * s)
        
    best_h = np.zeros(N)
    best_score = 1e9
    
    # Try a few initial configurations
    configs = [
        # Single wide bump
        lambda: np.ones(N) * 0.1,
        # 3 bumps
        lambda: np.array([1 if i%200 < 10 else 0 for i in range(N)], dtype=float),
        # 5 bumps
        lambda: np.array([1 if i%120 < 5 else 0 for i in range(N)], dtype=float)
    ]
    
    for i, gen in enumerate(configs):
        if time.time() > deadline - 30: break
        h0 = gen()
        # Scale to sum 1 for stability? No, objective is scale invariant.
        # But L-BFGS-B might behave better with normalized scale.
        # Let's keep it as is.
        
        try:
            res = minimize(objective, h0, method='L-BFGS-B', bounds=[(0, None)]*N, options={'maxiter': 200, 'ftol': 1e-5})
            if res.success or res.fun < best_score:
                best_h = res.x
                best_score = res.fun
        except Exception:
            pass
            
    return best_h.tolist()
