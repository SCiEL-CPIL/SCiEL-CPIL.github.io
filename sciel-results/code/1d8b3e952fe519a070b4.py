import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10

_DEADLINE = 150.0
_SAFETY = 15.0
_start = time.time()

def _elapsed():
    return time.time() - _start

def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()

def _autoconv_from_p(p):
    """
    Compute autoconvolution of h = p^2.
    Returns array of length N.
    """
    h = p * p
    full = np.convolve(h, h, mode='full')
    out = full[OFFSET:OFFSET + N]
    return DH * out

def _ratio_from_p(p):
    h = p * p
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv_from_p(p)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)

def _soft_logR_and_grad_p(p, beta=50.0):
    """
    Compute -log R(p) and its gradient.
    R(p) = ||h||_2^2 / (||h||_1 * ||h*h||_inf) where h = p^2.
    We minimize -log R(p) = -log(l2sq) + log(l1) + logsumexp(ac).
    """
    p = np.asarray(p, dtype=float)
    h = p * p
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    
    if l1 <= 1e-12 or l2sq <= 1e-12:
        return 1e12, np.zeros_like(p)

    ac = _autoconv_from_p(p)
    ac_max = float(np.max(ac))
    if ac_max <= 1e-12:
        return 1e12, np.zeros_like(p)

    # Soft max for log(ac_max)
    ac_shifted = ac - ac_max
    exp_ac = np.exp(beta * ac_shifted)
    sum_exp = float(np.sum(exp_ac))
    if sum_exp <= 0.0:
        return 1e12, np.zeros_like(p)
    
    logsumexp = ac_max + (1.0 / beta) * math.log(sum_exp)
    w = exp_ac / sum_exp

    # Gradient of logsumexp wrt p
    # d(logsumexp)/dp_k = sum_i w_i * d(ac_i)/dp_k
    # d(ac_i)/dp_k = 4 * p_k * p_{i+10-k}^2 (if in bounds)
    
    d_logsumexp_dp = np.zeros(N)
    p_sq = h
    
    # We want S_k = sum_i w_i * p_sq[i + 10 - k]
    for k in range(N):
        s = 0.0
        for i in range(N):
            j = i + 10 - k
            if 0 <= j < N:
                s += w[i] * p_sq[j]
        d_logsumexp_dp[k] = 4.0 * p[k] * s

    # Gradient of -log R
    # -log R = -log(l2sq) + log(l1) + logsumexp
    # d(-log R)/dp_k = - (4 p_k^3 / l2sq) + (2 p_k / l1) + d_logsumexp_dp[k] / logsumexp
    
    g = np.zeros(N)
    g -= 4.0 * p * (p * p) / l2sq
    g += 2.0 * p / l1
    g += d_logsumexp_dp / logsumexp
    
    val = -math.log(l2sq) + math.log(l1) + logsumexp
    return val, g

def _local_optimize_p(p0, time_budget):
    p0 = np.asarray(p0, dtype=float)
    best_p = p0.copy()
    best_v = _ratio_from_p(p0)
    t0 = time.time()

    def neg_logR(x):
        v, _ = _soft_logR_and_grad_p(x, beta=50.0)
        return v

    def neg_logR_grad(x):
        _, g = _soft_logR_and_grad_p(x, beta=50.0)
        return g

    try:
        res = minimize(
            neg_logR, p0,
            method='L-BFGS-B',
            jac=neg_logR_grad,
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-14,
                     'gtol': 1e-10, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > time_budget else x),
        )
        if res.x is not None:
            rp = res.x
            rv = _ratio_from_p(rp)
            if rv > best_v:
                best_v = rv
                best_p = rp.copy()
    except Exception:
        pass

    # Polish with Nelder-Mead on h directly? Or p?
    # Let's stick to p for consistency, but NM doesn't use grad.
    # Actually, NM on p might be slow.
    # Let's try a few NM steps on h using the ratio function.
    h_best = best_p * best_p
    
    def ratio_h(h):
        h = np.maximum(h, 0)
        return -_ratio_from_p(np.sqrt(h))

    try:
        for _ in range(5):
            if _time_left() <= 5.0:
                break
            res2 = minimize(
                ratio_h, h_best,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
                         'adaptive': True, 'disp': False},
            )
            rh = np.maximum(res2.x, 0.0)
            rp = np.sqrt(rh)
            rv = _ratio_from_p(rp)
            if rv > best_v:
                best_v = rv
                best_p = rp.copy()
                h_best = rh.copy()
    except Exception:
        pass

    return best_p, best_v

def _find_sidon_set(n, target_size):
    """Find a Sidon set of size target_size in {0, ..., n-1}."""
    sidon = []
    for i in range(n):
        can_add = True
        existing_sums = set()
        for a in range(len(sidon)):
            for b in range(a, len(sidon)):
                existing_sums.add(sidon[a] + sidon[b])
        
        new_sums = set()
        for s in sidon:
            new_sums.add(i + s)
        new_sums.add(2*i)
        
        if existing_sums & new_sums:
            can_add = False
        
        if can_add:
            sidon.append(i)
            if len(sidon) >= target_size:
                break
    return sidon

def _initializations_p():
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Single spike
    for pos in [0, N//2, N-1]:
        p = np.zeros(N)
        p[pos] = 1.0
        inits.append(p)
        
    # 2. Two spikes
    for a in range(0, N, 5):
        for b in range(a+5, N, 5):
            p = np.zeros(N)
            p[a] = 1.0
            p[b] = 1.0
            inits.append(p)
            
    # 3. Gaussians
    for center in range(0, N, 5):
        x = np.arange(N)
        p = np.exp(-0.5 * ((x - center) / 3.0) ** 2)
        inits.append(p)
        
    # 4. Two Gaussians
    for c1 in range(0, N, 10):
        for c2 in range(c1+10, N, 10):
            x = np.arange(N)
            p = np.exp(-0.5 * ((x - c1) / 2.0) ** 2) + np.exp(-0.5 * ((x - c2) / 2.0) ** 2)
            inits.append(p)
            
    # 5. Random
    for _ in range(50):
        p = rng.standard_normal(N)
        inits.append(p)
        
    # 6. Sparse random
    for _ in range(50):
        p = np.zeros(N)
        idx = rng.choice(N, size=rng.integers(2, 10), replace=False)
        p[idx] = rng.standard_normal(len(idx))
        inits.append(p)

    # 7. Sidon sets (inspiration 1)
    for k in [8, 9, 10, 11, 12]:
        sidon = _find_sidon_set(N, k)
        if len(sidon) >= k:
            p = np.zeros(N)
            for idx in sidon:
                p[idx] = 1.0
            inits.append(p)

    return inits

def construct():
    global _start
    _start = time.time()
    
    best_p = None
    best_v = 0.0
    
    inits = _initializations_p()
    scored = []
    for p in inits:
        v = _ratio_from_p(p)
        scored.append((v, p))
        if v > best_v:
            best_v = v
            best_p = p.copy()
            
    scored.sort(key=lambda t: -t[0])
    
    budget_per = 3.0
    for idx, (v0, p) in enumerate(scored):
        if _time_left() <= budget_per + 2.0:
            break
        if v0 < best_v * 0.5 and best_v > 1.0:
            continue
            
        p_opt, v_opt = _local_optimize_p(p, time_budget=budget_per)
        if v_opt > best_v:
            best_v = v_opt
            best_p = p_opt.copy()
            
        if _time_left() <= 20.0:
            break
            
    if best_p is not None and _time_left() > 15.0:
        try:
            p_pol, v_pol = _local_optimize_p(best_p, time_budget=_time_left() - 18.0)
            if v_pol > best_v:
                best_v = v_pol
                best_p = p_pol.copy()
        except Exception:
            pass
            
    if best_p is None:
        best_p = np.ones(N)
        
    h = best_p * best_p
    h = np.maximum(h, 0.0)
    return [float(x) for x in h]

if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio_from_p(np.sqrt(h)))
    print("time =", time.time() - t0)
