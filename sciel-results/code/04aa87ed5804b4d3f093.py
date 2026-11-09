import numpy as np
import time
from scipy.optimize import differential_evolution

def _calc_prob(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for given support and weights."""
    n = len(support)
    total = 0.0
    for k in range(n):
        threshold = 2.0 * support[k]
        wk = weights[k]
        # Compute P[X1+X2+X3 < threshold]
        prob3 = 0.0
        for i in range(n):
            si = support[i]
            wi = weights[i]
            for j in range(n):
                sij = si + support[j]
                wij = wi * weights[j]
                for l in range(n):
                    if sij + support[l] < threshold:
                        prob3 += wij * weights[l]
        total += wk * prob3
    return total


def _eval_4pt(params):
    """
    Evaluate 4-point measure.
    params: [log_a, log_b, log_c, p0, pa, pb]
    support = [0, exp(log_a), exp(log_b), exp(log_c)] with log_a < log_b < log_c
    weights = [p0, pa, pb, pc] where pc = 1-p0-pa-pb
    """
    log_a, log_b, log_c, p0, pa, pb = params
    a = np.exp(log_a)
    b = np.exp(log_b)
    c = np.exp(log_c)
    
    if not (a < b < c):
        return -0.5  # penalty
    
    pc = 1.0 - p0 - pa - pb
    if p0 < 0 or pa < 0 or pb < 0 or pc < 0:
        return -0.5
    
    support = np.array([0.0, a, b, c], dtype=np.float64)
    weights = np.array([p0, pa, pb, pc], dtype=np.float64)
    
    p = _calc_prob(support, weights)
    return -p  # minimize negative


def _eval_3pt(params):
    """
    Evaluate 3-point measure.
    params: [log_a, log_b, p0, pa]
    support = [0, exp(log_a), exp(log_b)] with log_a < log_b
    weights = [p0, pa, pb] where pb = 1-p0-pa
    """
    log_a, log_b, p0, pa = params
    a = np.exp(log_a)
    b = np.exp(log_b)
    
    if not (a < b):
        return -0.5
    
    pb = 1.0 - p0 - pa
    if p0 < 0 or pa < 0 or pb < 0:
        return -0.5
    
    support = np.array([0.0, a, b], dtype=np.float64)
    weights = np.array([p0, pa, pb], dtype=np.float64)
    
    p = _calc_prob(support, weights)
    return -p


def _eval_5pt(params):
    """
    Evaluate 5-point measure.
    params: [log_a, log_b, log_c, log_d, p0, pa, pb, pc]
    support = [0, exp(log_a), ..., exp(log_d)]
    weights = [p0, pa, pb, pc, pd] where pd = 1-sum
    """
    log_a, log_b, log_c, log_d, p0, pa, pb, pc = params
    a = np.exp(log_a)
    b = np.exp(log_b)
    c = np.exp(log_c)
    d = np.exp(log_d)
    
    if not (a < b < c < d):
        return -0.5
    
    pd = 1.0 - p0 - pa - pb - pc
    if p0 < 0 or pa < 0 or pb < 0 or pc < 0 or pd < 0:
        return -0.5
    
    support = np.array([0.0, a, b, c, d], dtype=np.float64)
    weights = np.array([p0, pa, pb, pc, pd], dtype=np.float64)
    
    p = _calc_prob(support, weights)
    return -p


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using differential evolution with variable support size.
    
    Strategy: Use scipy's differential_evolution (population-based global optimizer)
    to search over log-parameterized support points and simplex weights.
    Try 3, 4, and 5 point measures, keeping the best result.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin (630s limit, 40s buffer)
    
    best_prob = -1.0
    best_support = None
    best_weights = None
    
    def try_update(prob, support, weights):
        nonlocal best_prob, best_support, best_weights
        if prob > best_prob:
            best_prob = prob
            best_support = support.copy()
            best_weights = weights.copy()
    
    # --- 3-point search ---
    if time.time() < deadline - 120:
        try:
            bounds_3 = [
                (-2.0, 2.0),   # log_a: a in [0.135, 7.39]
                (0.0, 5.0),    # log_b: b in [1, 148]
                (0.01, 0.99),  # p0
                (0.01, 0.99),  # pa
            ]
            
            result_3 = differential_evolution(
                _eval_3pt, bounds_3,
                seed=42, maxiter=300, tol=1e-8,
                popsize=20, mutation=(0.3, 1.2), recombination=0.9,
                polish=True, init='sobol'
            )
            
            if result_3.fun < -0.4:  # prob > 0.4
                log_a, log_b, p0, pa = result_3.x
                a = np.exp(log_a)
                b = np.exp(log_b)
                pb = 1.0 - p0 - pa
                support = np.array([0.0, a, b], dtype=np.float64)
                weights = np.array([p0, pa, pb], dtype=np.float64)
                if np.all(weights >= 0) and a > 0 and b > a:
                    p = _calc_prob(support, weights)
                    try_update(p, support, weights)
        except Exception:
            pass
    
    # --- 4-point search ---
    if time.time() < deadline - 120:
        try:
            bounds_4 = [
                (-2.0, 2.0),   # log_a
                (-1.0, 3.0),   # log_b
                (0.0, 5.0),    # log_c
                (0.01, 0.99),  # p0
                (0.01, 0.99),  # pa
                (0.01, 0.99),  # pb
            ]
            
            result_4 = differential_evolution(
                _eval_4pt, bounds_4,
                seed=42, maxiter=300, tol=1e-8,
                popsize=25, mutation=(0.3, 1.2), recombination=0.9,
                polish=True, init='sobol'
            )
            
            if result_4.fun < -0.4:
                log_a, log_b, log_c, p0, pa, pb = result_4.x
                a = np.exp(log_a)
                b = np.exp(log_b)
                c = np.exp(log_c)
                pc = 1.0 - p0 - pa - pb
                if a < b < c and p0 >= 0 and pa >= 0 and pb >= 0 and pc >= 0:
                    support = np.array([0.0, a, b, c], dtype=np.float64)
                    weights = np.array([p0, pa, pb, pc], dtype=np.float64)
                    p = _calc_prob(support, weights)
                    try_update(p, support, weights)
        except Exception:
            pass
    
    # --- 5-point search (if time permits) ---
    if time.time() < deadline - 60:
        try:
            bounds_5 = [
                (-2.0, 1.0),   # log_a
                (-1.0, 2.0),   # log_b
                (0.0, 3.5),    # log_c
                (1.0, 5.0),    # log_d
                (0.01, 0.99),  # p0
                (0.01, 0.99),  # pa
                (0.01, 0.99),  # pb
                (0.01, 0.99),  # pc
            ]
            
            result_5 = differential_evolution(
                _eval_5pt, bounds_5,
                seed=123, maxiter=200, tol=1e-8,
                popsize=20, mutation=(0.3, 1.2), recombination=0.9,
                polish=True, init='sobol'
            )
            
            if result_5.fun < -0.4:
                log_a, log_b, log_c, log_d, p0, pa, pb, pc = result_5.x
                a = np.exp(log_a)
                b = np.exp(log_b)
                c = np.exp(log_c)
                d = np.exp(log_d)
                pd = 1.0 - p0 - pa - pb - pc
                if a < b < c < d and all(w >= 0 for w in [p0, pa, pb, pc, pd]):
                    support = np.array([0.0, a, b, c, d], dtype=np.float64)
                    weights = np.array([p0, pa, pb, pc, pd], dtype=np.float64)
                    p = _calc_prob(support, weights)
                    try_update(p, support, weights)
        except Exception:
            pass
    
    # --- Second round of 4-point with different seed if time permits ---
    if time.time() < deadline - 60 and best_prob < 0.55:
        try:
            bounds_4b = [
                (-3.0, 3.0),   # wider log_a
                (-2.0, 4.0),   # wider log_b
                (0.0, 6.0),    # wider log_c
                (0.01, 0.99),
                (0.01, 0.99),
                (0.01, 0.99),
            ]
            
            result_4b = differential_evolution(
                _eval_4pt, bounds_4b,
                seed=999, maxiter=200, tol=1e-8,
                popsize=30, mutation=(0.4, 1.5), recombination=0.8,
                polish=True, init='latinhypercube'
            )
            
            if result_4b.fun < -0.4:
                log_a, log_b, log_c, p0, pa, pb = result_4b.x
                a = np.exp(log_a)
                b = np.exp(log_b)
                c = np.exp(log_c)
                pc = 1.0 - p0 - pa - pb
                if a < b < c and p0 >= 0 and pa >= 0 and pb >= 0 and pc >= 0:
                    support = np.array([0.0, a, b, c], dtype=np.float64)
                    weights = np.array([p0, pa, pb, pc], dtype=np.float64)
                    p = _calc_prob(support, weights)
                    try_update(p, support, weights)
        except Exception:
            pass
    
    # --- Final: if nothing found, use a known-good default ---
    if best_support is None:
        # Static fallback: 3-point measure
        best_support = np.array([0.0, 1.0, 10.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        best_prob = _calc_prob(best_support, best_weights)
    
    # Final validity check and cleanup
    if np.any(best_weights < -1e-9) or np.sum(best_weights) <= 1e-9 or np.any(best_support < -1e-9):
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    
    # Ensure weights sum to 1 (normalize for safety)
    wsum = np.sum(best_weights)
    if wsum > 1e-9:
        best_weights = best_weights / wsum
    
    # Ensure support is non-negative
    best_support = np.maximum(best_support, 0.0)
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)


if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print("Support:", s)
    print("Weights:", w)
    print("Probability:", _calc_prob(s, w))
