import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _objective(h):
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr)

def _construct_with_deadline(deadline):
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)
    
    start = time.time()
    rng = np.random.default_rng(42)
    
    for restart in range(50):
        if time.time() > deadline - 3:
            break
        
        # Generate diverse initializations
        if restart == 0:
            h0 = np.full(N, 0.5)
        elif restart % 3 == 1:
            # Block-like
            h0 = np.zeros(N)
            ones = rng.choice(N, N//2, replace=False)
            h0[ones] = 1.0
        elif restart % 3 == 2:
            # Smooth random
            h0 = np.cumsum(rng.normal(0, 1, N))
            h0 = (h0 - h0.min()) / (h0.max() - h0.min() + 1e-12)
        else:
            h0 = rng.uniform(0, 1, N)
        
        # Normalize to sum = N/2
        s = h0.sum()
        if s > 1e-12:
            h0 = h0 / s * (N / 2)
        h0 = np.clip(h0, 0, 1)
        
        # Re-normalize after clipping
        for _ in range(10):
            diff = N / 2 - h0.sum()
            if abs(diff) < 1e-10:
                break
            h0 += diff / N
            h0 = np.clip(h0, 0, 1)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2}]
        bounds = [(0, 1)] * N
        
        try:
            result = minimize(
                _objective,
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            h_candidate = np.clip(result.x, 0, 1)
            # Fix sum
            for _ in range(10):
                diff = N / 2 - h_candidate.sum()
                if abs(diff) < 1e-10:
                    break
                h_candidate += diff / N
                h_candidate = np.clip(h_candidate, 0, 1)
            
            if np.isfinite(h_candidate).all() and abs(h_candidate.sum() - N / 2) < 1e-9 * N:
                val = _objective(h_candidate)
                if val < best_val:
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h

def construct():
    deadline = time.time() + 100
    return _construct_with_deadline(deadline)
