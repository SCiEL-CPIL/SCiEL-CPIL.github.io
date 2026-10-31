"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 10.0


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]  # shifted[x] = h[x+1] for x in [0,N-2]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _smooth_overlap(h, beta=100.0):
    """Compute smooth max overlap."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    
    # Softmax
    max_ov = np.max(overlap)
    if max_ov < 0:
        return max_ov
    exp_ov = np.exp(beta * (overlap - max_ov))
    sum_exp = np.sum(exp_ov)
    if sum_exp == 0:
        return max_ov
    return max_ov + np.log(sum_exp) / beta


def _smooth_overlap_grad(h, beta=100.0):
    """Compute gradient of smooth max overlap."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    
    max_ov = np.max(overlap)
    exp_ov = np.exp(beta * (overlap - max_ov))
    sum_exp = np.sum(exp_ov)
    if sum_exp == 0:
        return np.zeros(n)
    weights = exp_ov / sum_exp
    
    grad = np.zeros(n)
    for k in range(n):
        # O_k = h_k (1 - h_{k+1})
        # dO_k/dh_k = 1 - h_{k+1}
        # dO_k/dh_{k+1} = -h_k
        grad[k] += weights[k] * (1.0 - shifted[k])
        if k + 1 < n:
            grad[k+1] += weights[k] * (-h[k])
            
    return grad

def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0,1] if possible."""
    target_sum = N / 2.0
    current_sum = np.sum(h)
    diff = target_sum - current_sum
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    current_sum = np.sum(h)
    diff = target_sum - current_sum
    if abs(diff) > 1e-10:
        mask = (h > 0.01) & (h < 0.99)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_score = 1.0
    best_h = np.full(N, 0.5)
    
    starts = []
    starts.append(np.full(N, 0.5))
    
    A0 = (3.0 - math.sqrt(5.0)) / 2.0
    B0 = 1.0 - A0
    m0 = N / 2.0
    h_2block = np.zeros(N)
    h_2block[:int(m0)] = A0
    h_2block[int(m0):] = B0
    h_2block = _fix_sum(h_2block)
    starts.append(h_2block)
    
    for A in [0.3, 0.5, 0.7]:
        for phi in [0, math.pi/4, math.pi/2]:
            h_sin = 0.5 + A * np.sin(2 * np.pi * np.arange(N) / N + phi)
            h_sin = _fix_sum(h_sin)
            starts.append(h_sin)
            
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _fix_sum(h_rand)
        starts.append(h_rand)

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}
    ]
    bounds = [(0.0, 1.0)] * N
    
    for h0 in starts:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(_smooth_overlap, h0, jac=_smooth_overlap_grad, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.success or res.fun < best_score:
                h_opt = _fix_sum(res.x)
                score_opt = _score(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt
        except Exception:
            pass

    if time.time() < deadline - 5.0:
        try:
            def obj_high_beta(h):
                return _smooth_overlap(h, beta=500.0)
            def grad_high_beta(h):
                return _smooth_overlap_grad(h, beta=500.0)
            
            res = minimize(obj_high_beta, best_h, jac=grad_high_beta, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 100, 'ftol': 1e-14})
            h_polish = _fix_sum(res.x)
            score_polish = _score(h_polish)
            if score_polish < best_score:
                best_score = score_polish
                best_h = h_polish
        except Exception:
            pass

    return best_h
