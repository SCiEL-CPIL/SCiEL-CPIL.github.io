import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 143.0

def _autoconv_sup(h):
    h = np.asarray(h, dtype=float)
    n = len(h)
    pad = 2 * n - 1
    a = np.zeros(pad)
    a[:n] = h
    c = np.fft.ifft(np.fft.fft(a) ** 2).real
    return float(np.max(c)) * (1.0 / n)

def _ratio(h):
    h = np.maximum(h, 0.0)
    s1 = float(np.sum(h))
    if s1 < 1e-15:
        return 0.0
    s2 = float(np.sum(h * h))
    sup = _autoconv_sup(h)
    if sup < 1e-15:
        return 0.0
    return s2 / (s1 * sup)

def _sidon_set(max_size, seed=0):
    rng = np.random.default_rng(seed)
    s = [0]
    diffs = set()
    order = list(range(1, N))
    rng.shuffle(order)
    for c in order:
        new_diffs = set()
        ok = True
        for i in s:
            d = abs(c - i)
            if d in diffs or d in new_diffs:
                ok = False
                break
            new_diffs.add(d)
        if ok:
            s.append(c)
            diffs.update(new_diffs)
        if len(s) >= max_size:
            break
    return sorted(s)

def _optimize_weights(support, deadline):
    """Optimize weights for a given support using Nelder-Mead."""
    if len(support) < 2:
        return np.zeros(N)
    
    def neg_r(w):
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s < 1e-15:
            return 0.0
        w = w / s
        full = np.zeros(N)
        full[support] = w
        return -_ratio(full)
    
    w0 = np.ones(len(support)) / len(support)
    try:
        res = minimize(neg_r, w0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        w = np.maximum(res.x, 0.0)
        s = w.sum()
        if s < 1e-15:
            return np.zeros(N)
        w = w / s
        full = np.zeros(N)
        full[support] = w
        return full
    except Exception:
        return np.zeros(N)

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Generate diverse supports
    supports = []
    # Sidon sets
    for sz in range(4, 20):
        for seed in range(5):
            s = _sidon_set(sz, seed)
            if len(s) >= 3:
                supports.append(s)
                
    # Random sets
    for _ in range(50):
        k = rng.integers(5, 25)
        idx = rng.choice(N, size=k, replace=False)
        supports.append(sorted(idx))
        
    # Periodic
    for p in [2, 3, 4, 5, 7]:
        supports.append(list(range(0, N, p)))
        
    # Evaluate and optimize
    for s in supports:
        if time.time() > DEADLINE - 5:
            break
        h = _optimize_weights(s, DEADLINE - 5)
        sc = _ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h
            
    # Final polish on best support
    if best_score > 0:
        support = np.where(best_h > 1e-8)[0]
        h_final = _optimize_weights(support, DEADLINE - 2)
        sc_final = _ratio(h_final)
        if sc_final > best_score:
            best_h = h_final
            
    return best_h.tolist()
