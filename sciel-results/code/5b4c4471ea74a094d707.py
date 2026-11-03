import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.optimize import linprog

# ---- Task constants (interface) ----
N = 50
DELTA = 0.5 / N            # interval width; constant, does not affect argmax
# grid positions in [-1/4, 1/4]
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 140.0          # explicit wall-clock safety margin (< 150s kill)
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv).  Returns -inf/0 if degenerate."""
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


def _neg_ratio(h):
    r = _ratio(h)
    if not np.isfinite(r) or r <= 0.0:
        return 0.0
    return -r

def _neg_ratio_grad(h):
    """
    Gradient of -log(ratio) = -log(L2^2) + log(L1) + log(M)
    We minimize this.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    l2_sq = np.dot(h, h)
    if l1 <= 0 or l2_sq <= 0:
        return 0.0, np.zeros(N)
    
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0:
        return 0.0, np.zeros(N)
    
    # Value: -log(l2_sq) + log(l1) + log(m)
    val = -np.log(l2_sq) + np.log(l1) + np.log(m)
    
    # Gradient
    # d/dh (-log(l2_sq)) = -2h / l2_sq
    grad = -2.0 * h / l2_sq
    
    # d/dh (log(l1)) = 1/l1
    grad += 1.0 / l1
    
    # d/dh (log(m)) = (1/m) * grad(m)
    # grad(m) is subgradient. Pick index of max.
    idx_max = np.argmax(conv)
    # conv[k] = sum_i h_i h_{k-i}
    # d conv[k] / d h_j = h_{k-j} + h_{k-j} (if k-j in range)
    # effectively 2 * h_{k-j} (reflected)
    # Let's compute the vector g where g_j = d conv[idx_max] / d h_j
    g = np.zeros(N)
    k = idx_max
    # j ranges 0..N-1
    # k-j must be in 0..N-1
    for j in range(N):
        kj = k - j
        if 0 <= kj < N:
            # Term from i=j: h_j * h_{k-j} -> deriv h_{k-j}
            # Term from i=k-j: h_{k-j} * h_j -> deriv h_j? No, deriv of h_{k-j}*h_j wrt h_j is h_{k-j}.
            # Wait, if j == k-j, it's h_j^2, deriv 2h_j.
            # If j != k-j, both terms contribute h_{k-j}.
            # So always 2 * h_{k-j}?
            # Let's check.
            # C_k = sum_i h_i h_{k-i}.
            # dC_k/dh_j = h_{k-j} (from i=j) + h_{k-j} (from i=k-j, if k-j != j).
            # If k-j == j, then i=j is the only term, deriv is 2h_j.
            # So yes, it is 2 * h_{k-j} (with h_out=0).
            g[j] = 2.0 * h[kj]
            
    grad += (1.0 / m) * g
    
    return val, grad


def _bump(a, b, c):
    """Parametric smooth bump: sharpness a, width b, center c on grid index."""
    return np.exp(-a * (np.arange(N) - c) ** 2) * (1.0 + b * np.cos(0.5 * np.pi * np.arange(N) / N))


def _proj_simplex(v, s=1.0):
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, N + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _polish(h, maxiter=300):
    """Local refinement maximizing ratio under L1=1, h>=0 using analytic gradient."""
    h0 = h / np.sum(h)
    bounds = [(0.0, 1.0)] * N
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
             'jac': lambda x: np.ones(N)}]
    try:
        res = minimize(_neg_ratio_grad, h0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False})
        if res.x is not None:
            hx = np.clip(res.x, 0.0, None)
            if np.sum(hx) > 0:
                hx = hx / np.sum(hx)
                if _ratio(hx) > _ratio(h0):
                    return hx
    except Exception:
        pass
    return h0


def _find_sidon_support(k, seed=0):
    """
    Find a Sidon-like support set of size k in [0, N).
    A Sidon set has unique pairwise sums (i+j for i<=j).
    We use a greedy randomized approach.
    """
    rng = np.random.RandomState(seed)
    best_set = None
    best_score = -1
    
    for attempt in range(50):
        S = []
        sums = set()
        valid = True
        
        # Try to build a set of size k
        for _ in range(k):
            # Try random candidates
            found = False
            for _trial in range(100):
                cand = rng.randint(0, N)
                if cand in S:
                    continue
                
                # Check if adding cand creates a collision
                new_sums = set()
                conflict = False
                for s in S:
                    pair_sum = cand + s
                    if pair_sum in sums:
                        conflict = True
                        break
                    new_sums.add(pair_sum)
                if not conflict:
                    # Also check cand+cand
                    if cand + cand in sums:
                        conflict = True
                    else:
                        new_sums.add(cand + cand)
                    
                    if not conflict:
                        S.append(cand)
                        sums.update(new_sums)
                        found = True
                        break
            
            if not found:
                valid = False
                break
        
        if valid and len(S) == k:
            # Score: how "spread out" is the set? Prefer larger gaps
            S_sorted = sorted(S)
            gaps = np.diff(S_sorted)
            score = np.min(gaps) if len(gaps) > 0 else 0
            if score > best_score:
                best_score = score
                best_set = S_sorted
    
    if best_set is None:
        # Fallback: random set
        best_set = sorted(rng.choice(N, k, replace=False).tolist())
    
    return best_set


def _lp_minimize_peak(h_fixed):
    pass


def construct():
    np.random.seed(12345)
    best_h = np.ones(N) / N
    best_r = _ratio(best_h)

    def upd(h):
        nonlocal best_h, best_r
        h = np.asarray(h, dtype=np.float64)
        if np.sum(h) <= 0 or not np.all(np.isfinite(h)):
            return
        hh = np.clip(h, 0.0, None)
        r = _ratio(hh)
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_h = hh.copy()

    # ---------------- PHASE 1: Sidon Set Search & Weight Optimization ----------------
    candidate_supports = []
    
    # 1. Small Sidon sets (sparse)
    for k in range(4, 12):
        for seed in range(5):
            S = _find_sidon_support(k, seed=seed + k*100)
            candidate_supports.append(S)
    
    # 2. Specific known Sidon sets (e.g. powers of 2, primes)
    candidate_supports.append([0, 1, 2, 4, 8, 16, 32])
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    candidate_supports.append([p for p in primes if p < N])
    
    # 3. Random smooth sets (from current program)
    for _ in range(10):
        a = np.random.uniform(0.5, 5.0)
        b = np.random.uniform(-0.9, 0.9)
        c = np.random.uniform(4, N - 4)
        h_bump = _bump(a, b, c)
        # Threshold to get support
        threshold = np.percentile(h_bump, 50)
        support = np.where(h_bump > threshold)[0].tolist()
        if len(support) > 3:
            candidate_supports.append(support)

    # Optimize weights for each support
    for S in candidate_supports:
        if _time_left() < 10:
            break
        
        # Create initial vector with ones on support
        h_init = np.zeros(N)
        for idx in S:
            h_init[idx] = 1.0
        
        # Try different initial weight distributions
        h_cands = []
        h_cands.append(h_init.copy())
        
        # Geometric
        for r in [0.5, 0.8, 1.2, 2.0]:
            h_g = np.zeros(N)
            for i, idx in enumerate(S):
                h_g[idx] = r ** i
            h_cands.append(h_g)
        
        # Random
        for _ in range(3):
            h_r = np.zeros(N)
            for idx in S:
                h_r[idx] = np.random.uniform(0.1, 1.0)
            h_cands.append(h_r)
        
        for h_c in h_cands:
            if _time_left() < 5:
                break
            # Normalize
            h_c = h_c / np.sum(h_c)
            # Polish
            h_opt = _polish(h_c, maxiter=200)
            upd(h_opt)

    # ---------------- PHASE 2: Global Random Search with Polish ----------------
    inits = [best_h.copy(), np.ones(N) / N]
    
    # More random smooth starts
    for _ in range(10):
        if _time_left() < 15:
            break
        a = np.random.uniform(0.5, 8.0)
        b = np.random.uniform(-0.9, 0.9)
        c = np.random.uniform(2, N - 2)
        inits.append(_bump(a, b, c) / N)
    
    # Sparse random starts
    for _ in range(10):
        if _time_left() < 15:
            break
        k = np.random.randint(4, 15)
        h = np.zeros(N)
        h[np.random.choice(N, k, replace=False)] = np.random.uniform(0.1, 1.0, k)
        inits.append(h)
    
    # "Peaky" starts: one large, rest small
    for _ in range(5):
        if _time_left() < 15:
            break
        h = np.random.rand(N) * 0.01
        h[np.random.randint(N)] = 1.0
        inits.append(h)

    for h in inits:
        if _time_left() < 12:
            break
        h = np.clip(h, 0.0, None)
        if np.sum(h) <= 0:
            continue
        upd(_polish(h / np.sum(h), maxiter=250))

    # ---------------- PHASE 3: Final Tight Polish ----------------
    if _time_left() > 8:
        upd(_polish(best_h, maxiter=1500))

    return list(best_h)
