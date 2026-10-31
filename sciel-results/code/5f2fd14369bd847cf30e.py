import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    def objective(h):
        h_clipped = np.clip(h, 0, 1)
        overlaps = np.array([np.sum(h_clipped * (1 - np.roll(h_clipped, t))) for t in range(N)])
        beta = 100.0
        m = np.max(overlaps)
        smooth_max = m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta
        sum_penalty = 1000.0 * (np.sum(h_clipped) - N / 2.0) ** 2
        return smooth_max + sum_penalty

    best_h = None
    best_score = float('inf')

    # Multiple initial guesses
    inits = []
    inits.append(np.full(N, 0.5))
    inits.append(np.clip(0.5 + 0.1 * np.sin(2 * np.pi * np.arange(N) / N), 0, 1))
    inits.append(np.clip(0.5 + 0.1 * np.random.randn(N), 0, 1))
    inits.append(np.clip(0.5 + 0.2 * np.sin(4 * np.pi * np.arange(N) / N), 0, 1))
    
    bounds = [(0.0, 1.0)] * N
    
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            h = np.clip(res.x, 0, 1)
            
            # Enforce sum constraint strictly
            s = np.sum(h)
            if s > 0:
                h = h * (N / 2.0) / s
            h = np.clip(h, 0, 1)
            
            # Fine-tune sum
            diff = N / 2.0 - np.sum(h)
            idx = np.argsort(h)[::-1]
            for i in idx:
                if diff > 0:
                    add = min(diff, 1.0 - h[i])
                    h[i] += add
                    diff -= add
                elif diff < 0:
                    sub = min(-diff, h[i])
                    h[i] -= sub
                    diff += sub
                else:
                    break
            
            # Calculate true score
            overlaps = np.array([np.sum(h * (1 - np.roll(h, t))) for t in range(N)])
            score = np.max(overlaps)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
        except Exception:
            continue

    if best_h is None:
        return np.full(N, 0.5)
        
    return best_h
