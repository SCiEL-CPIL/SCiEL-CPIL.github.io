import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0

def autoconv_max(h):
    h = np.asarray(h, dtype=float)
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)

def ratio(h):
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0:
        return 0.0
    mc = autoconv_max(h)
    if mc <= 0:
        return 0.0
    return l2sq / (l1 * mc)

def neg_ratio(h):
    return -ratio(h)

def construct():
    t0 = time.time()
    def elapsed():
        return time.time() - t0
    
    best_h = np.ones(N)
    best_val = ratio(best_h)
    
    candidates = [
        np.ones(N),
        np.array([1.0] + [0.0]*(N-1)),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0] + [0.0]*(N-6)),
        np.linspace(1.0, 0.0, N),
        np.linspace(0.0, 1.0, N),
        np.array([1.0 if i%2==0 else 0.0 for i in range(N)]),
        np.array([1.0 if i%2==1 else 0.0 for i in range(N)]),
        np.array([1.0, 1.0] + [0.0]*(N-2)),
        np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0] + [0.0]*(N-6)),
        np.array([1.0, 1.0, 1.0] + [0.0]*(N-3)),
    ]
    
    for c in candidates:
        if elapsed() > DEADLINE:
            break
        v = ratio(c)
        if v > best_val:
            best_val = v
            best_h = c.copy()
    
    rng = np.random.default_rng(123)
    
    # Local optimization
    n_opt = 0
    max_opt = 40
    while n_opt < max_opt:
        if elapsed() > DEADLINE - 10:  # leave 10s margin
            break
        
        if n_opt == 0:
            h0 = best_h.copy()
        elif n_opt == 1:
            h0 = np.ones(N)
        elif n_opt == 2:
            h0 = np.array([1.0] + [0.0]*(N-1))
        elif n_opt == 3:
            h0 = np.linspace(1.0, 0.1, N)
        elif n_opt == 4:
            h0 = np.linspace(0.1, 1.0, N)
        elif n_opt == 5:
            h0 = np.array([1.0 if i < N//2 else 0.0 for i in range(N)])
        elif n_opt == 6:
            h0 = np.array([0.0 if i < N//2 else 1.0 for i in range(N)])
        elif n_opt == 7:
            h0 = np.array([1.0 if i%3==0 else 0.0 for i in range(N)])
        elif n_opt == 8:
            h0 = np.array([1.0 if i%3==1 else 0.0 for i in range(N)])
        else:
            # Random sparse
            k = int(rng.integers(1, N+1))
            h0 = np.zeros(N)
            idx = rng.choice(N, size=k, replace=False)
            h0[idx] = rng.uniform(0.5, 2.0, size=k)
        
        # Normalize
        s = np.sum(h0)
        if s > 0:
            h0 = h0 / s
        
        try:
            res = minimize(neg_ratio, h0, method='L-BFGS-B',
                          bounds=[(0, None)]*N,
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_h = np.maximum(res.x, 0)
        except Exception:
            pass
        
        n_opt += 1
    
    # Final polish: coordinate descent
    if elapsed() < DEADLINE - 5:
        h = best_h.copy()
        for _ in range(50):
            if elapsed() > DEADLINE - 5:
                break
            improved = False
            for i in range(N):
                old = h[i]
                # Try a few values
                for trial in [old*0.5, old*0.8, old*1.2, old*1.5, 0.0, 0.1, 0.5, 1.0, 2.0]:
                    h[i] = max(0, trial)
                    v = ratio(h)
                    if v > best_val:
                        best_val = v
                        improved = True
                # Reset and find best
                h[i] = old
                best_i = old
                best_v = ratio(h)
                for trial in [old*0.1, old*0.3, old*0.5, old*0.7, old*1.0, old*1.3, old*1.5, old*2.0, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]:
                    h[i] = max(0, trial)
                    v = ratio(h)
                    if v > best_v:
                        best_v = v
                        best_i = trial
                h[i] = best_i
                if best_i != old:
                    improved = True
            if not improved:
                break
        if ratio(h) > best_val:
            best_val = ratio(h)
            best_h = h
    
    return list(best_h)
