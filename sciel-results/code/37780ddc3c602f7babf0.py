import numpy as np
import time
from scipy.optimize import minimize

# ---- Task constants (interface) ----
N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 140.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return -np.inf
    l1 = np.sum(h)
    if l1 <= 0.0:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0.0:
        return 0.0
    return l2_sq / (l1 * m)

def _find_max_sidon_set(N, max_k=15, iterations=200):
    """
    Find the largest Sidon set in [0, N) using randomized greedy with restarts.
    A Sidon set S satisfies that all sums i+j for i,j in S are unique.
    """
    best_set = []
    rng = np.random.RandomState(42)
    
    for _ in range(iterations):
        S = []
        sums = set()
        candidates = list(range(N))
        rng.shuffle(candidates)
        
        for c in candidates:
            if len(S) >= max_k:
                break
            conflict = False
            for s in S:
                if c + s in sums:
                    conflict = True
                    break
            if not conflict:
                if c + c in sums:
                    conflict = True
            if not conflict:
                S.append(c)
                for s in S:
                    sums.add(c + s)
        
        if len(S) > len(best_set):
            best_set = S[:]
            if len(best_set) >= max_k:
                break
                
    return best_set

def _optimize_weights(h_init, max_iter=500):
    """
    Refine the weights of h_init to maximize the ratio.
    We fix the support (non-zero indices) and optimize the positive values.
    """
    support = np.where(h_init > 1e-12)[0]
    if len(support) == 0:
        return h_init
    
    # We will optimize the weights on the support.
    # To make the problem well-posed, we can fix the sum of weights or just optimize the shape.
    # The ratio is scale invariant: R(c*h) = R(h).
    # So we can fix sum(h) = 1.
    
    h0 = h_init.copy()
    h0[support] /= np.sum(h0[support])
    
    # Parameters: weights on support
    w0 = h0[support].copy()
    
    def obj(w):
        if np.any(w < 0):
            return 1.0
        h = np.zeros(N)
        h[support] = w
        # Normalize to sum=1 to keep on the simplex
        s = np.sum(w)
        if s <= 0:
            return 1.0
        h = w / s
        # Maximize L2^2 / (L1 * MaxConv). With L1=1, maximize L2^2 / MaxConv.
        l2_sq = np.dot(h, h)
        conv = np.convolve(h, h, mode="full")
        m = np.max(conv)
        if m <= 0:
            return 1.0
        return -l2_sq / m
    
    # Simplex constraint: sum(w) = 1, w >= 0
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * len(w0)
    
    try:
        res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': max_iter, 'ftol': 1e-10})
        if res.success:
            h_opt = np.zeros(N)
            h_opt[support] = res.x
            # Scale back? The ratio is scale invariant, so any scale works.
            # But let's keep sum=1 for consistency.
            return h_opt
    except Exception:
        pass
    
    return h0

def construct():
    np.random.seed(12345)
    
    best_h = None
    best_ratio = -np.inf
    
    # 1. Generate candidate supports using Sidon sets
    # We try different sizes to see which yields a better base.
    # Larger Sidon sets -> more non-zeros -> potentially lower L2^2/L1^2 if weights are equal,
    # but potentially much lower MaxConv.
    
    # The current program uses max_k=15.
    # Inspiration 1 suggests trying sizes 5-7.
    # Let's try a range of sizes.
    
    for k_target in [10, 12, 15, 18, 20]:
        if _time_left() < 10.0:
            break
            
        # Find a Sidon set of size approx k_target
        S = _find_max_sidon_set(N, max_k=k_target, iterations=100)
        
        if not S:
            continue
            
        h = np.zeros(N)
        h[S] = 1.0 / len(S)
        
        ratio = _ratio(h)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h.copy()
            
        # 2. Refine weights for this support
        if _time_left() < 20.0:
            continue
            
        h_refined = _optimize_weights(h, max_iter=200)
        ratio_ref = _ratio(h_refined)
        
        if ratio_ref > best_ratio:
            best_ratio = ratio_ref
            best_h = h_refined.copy()
            
    # 3. Fallback / Additional search if time permits
    # Try a few random sparse supports
    if _time_left() > 30.0:
        rng = np.random.RandomState(12345)
        for _ in range(50):
            if _time_left() < 10.0:
                break
            k = rng.randint(5, 15)
            S = rng.choice(N, k, replace=False)
            h = np.zeros(N)
            h[S] = 1.0 / k
            
            ratio = _ratio(h)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h.copy()
                
            # Quick refine
            if _time_left() > 20.0:
                h_refined = _optimize_weights(h, max_iter=100)
                ratio_ref = _ratio(h_refined)
                if ratio_ref > best_ratio:
                    best_ratio = ratio_ref
                    best_h = h_refined.copy()

    if best_h is None:
        # Ultimate fallback
        h = np.zeros(N)
        h[0] = 1.0
        return list(h)
        
    return list(best_h)
