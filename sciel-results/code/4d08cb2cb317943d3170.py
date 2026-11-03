import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.linalg import expm
import sys

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

def _random_sidon_set(N, k, rng):
    """Find a Sidon set of size k in [0, N) using randomized greedy."""
    for _ in range(1000):
        S = []
        sums = set()
        candidates = list(range(N))
        rng.shuffle(candidates)
        
        for c in candidates:
            if len(S) >= k:
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
        
        if len(S) >= k:
            return S[:k]
    return None

def _block_coordinate_descent(h, max_iter=5000, tol=1e-8):
    """
    Block coordinate descent for maximizing R(h) = L2^2 / (L1 * MaxConv).
    We normalize L1=1 so we maximize L2^2 / MaxConv.
    """
    h = h.copy()
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s <= 0:
        return h
    h = h / s  # normalize L1=1
    
    best_ratio = _ratio(h)
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
            
        old_ratio = _ratio(h)
        h_old = h.copy()
        
        # Compute autoconvolution
        conv = np.convolve(h, h, mode="full")
        max_conv = np.max(conv)
        
        # For each coordinate i, try to optimize h[i] while keeping others fixed.
        # This is expensive to do naively. Instead, use a coordinate-wise update.
        # 
        # Let h = h_fixed + h_i * e_i.
        # L1 = L1_fixed + h_i
        # L2^2 = L2_fixed^2 + h_i^2
        # MaxConv is more complex.
        #
        # Alternative: use gradient-based optimization on the simplex.
        
        # Let's use SLSQP on the full vector but with a tighter tolerance and more iterations.
        # Actually, let's try a different approach: optimize each coordinate in turn.
        
        # For coordinate i:
        # Fix all h[j] for j != i.
        # Let a = sum_{j!=i} h[j], b = sum_{j!=i} h[j]^2
        # L1 = a + x
        # L2^2 = b + x^2
        # MaxConv: The autoconvolution is (h_fixed * h_fixed) + x * (h_fixed * e_i + e_i * h_fixed) + x^2 * (e_i * e_i)
        # This is complicated. Let's just use scipy.optimize.minimize with SLSQP on the full vector.
        
        def obj(w):
            if np.any(w < 0):
                return 1.0
            s = np.sum(w)
            if s <= 0:
                return 1.0
            h_norm = w / s
            l2_sq = np.dot(h_norm, h_norm)
            conv = np.convolve(h_norm, h_norm, mode="full")
            m = np.max(conv)
            if m <= 0:
                return 1.0
            return -l2_sq / m
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(obj, h, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 200, 'ftol': 1e-12})
            if res.success and res.fun < -best_ratio + 1e-12:
                h = np.maximum(res.x, 0)
                s = np.sum(h)
                if s > 0:
                    h = h / s
                new_ratio = _ratio(h)
                if new_ratio > best_ratio + tol:
                    best_ratio = new_ratio
                elif abs(new_ratio - best_ratio) <= tol:
                    break
        except Exception:
            break
        
    return h

def _spectral_method(h, max_iter=100):
    """
    Use the fact that MaxConv = max_k (h*h)[k].
    For a fixed support, we can try to find weights that minimize MaxConv.
    This is a min-max problem.
    """
    support = np.where(h > 1e-10)[0]
    if len(support) == 0:
        return h
    
    h = h.copy()
    h[support] /= np.sum(h[support])
    
    # Use bisection on T: is there a feasible h with MaxConv <= T?
    # Minimize MaxConv subject to L1=1, L2^2 >= some value.
    # This is getting complicated. Let's stick with SLSQP but with better initializations.
    
    return h

def construct():
    np.random.seed(12345)
    rng = np.random.RandomState(12345)
    
    best_h = None
    best_ratio = -np.inf
    
    # 1. Try various Sidon set sizes
    # Smaller Sidon sets have sparser support, which can lead to higher L2^2/L1^2
    # but potentially higher MaxConv. We need to find the sweet spot.
    
    for k in [3, 4, 5, 6, 7, 8, 10, 12]:
        if _time_left() < 15.0:
            break
        
        # Find a Sidon set of size k
        S = _random_sidon_set(N, k, rng)
        if S is None:
            continue
        
        # Initialize with equal weights
        h = np.zeros(N)
        h[S] = 1.0 / k
        
        ratio = _ratio(h)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h.copy()
        
        # Refine with SLSQP
        if _time_left() < 10.0:
            continue
        
        h_refined = _block_coordinate_descent(h, max_iter=1000)
        ratio_ref = _ratio(h_refined)
        
        if ratio_ref > best_ratio:
            best_ratio = ratio_ref
            best_h = h_refined.copy()
        
        # Try a few more random Sidon sets of the same size
        for _ in range(3):
            if _time_left() < 10.0:
                break
            S2 = _random_sidon_set(N, k, rng)
            if S2 is None:
                continue
            h2 = np.zeros(N)
            h2[S2] = 1.0 / k
            ratio2 = _ratio(h2)
            if ratio2 > best_ratio:
                best_ratio = ratio2
                best_h = h2.copy()
            
            if _time_left() < 10.0:
                continue
            h2_ref = _block_coordinate_descent(h2, max_iter=500)
            ratio2_ref = _ratio(h2_ref)
            if ratio2_ref > best_ratio:
                best_ratio = ratio2_ref
                best_h = h2_ref.copy()
    
    # 2. Try random sparse supports
    if _time_left() > 20.0:
        for _ in range(20):
            if _time_left() < 10.0:
                break
            k = rng.randint(3, 10)
            S = rng.choice(N, k, replace=False)
            h = np.zeros(N)
            h[S] = 1.0 / k
            ratio = _ratio(h)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h.copy()
            
            if _time_left() < 10.0:
                continue
            h_ref = _block_coordinate_descent(h, max_iter=500)
            ratio_ref = _ratio(h_ref)
            if ratio_ref > best_ratio:
                best_ratio = ratio_ref
                best_h = h_ref.copy()
    
    # 3. Final polish on the best solution
    if best_h is not None and _time_left() > 10.0:
        best_h_polished = _block_coordinate_descent(best_h, max_iter=2000, tol=1e-10)
        ratio_polished = _ratio(best_h_polished)
        if ratio_polished > best_ratio:
            best_h = best_h_polished
            best_ratio = ratio_polished
    
    if best_h is None:
        # Ultimate fallback
        h = np.zeros(N)
        h[0] = 1.0
        return list(h)
        
    return list(best_h)
