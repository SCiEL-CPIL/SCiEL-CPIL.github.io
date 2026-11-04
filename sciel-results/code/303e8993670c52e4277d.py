import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    autoconv = np.convolve(h, h, mode='full')
    sup_auto = np.max(autoconv)
    if sup_auto == 0:
        return 0.0
    return l2_sq / (l1 * sup_auto)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = compute_ratio([1.0] * N)
    best_h = [1.0] * N
    
    # Try sparse patterns with Sidon-like sets
    # A Sidon set ensures distinct pairwise sums, minimizing autocorrelation peaks
    def make_sidon_set(K, n):
        # Greedy Sidon set
        S = []
        for i in range(n):
            if len(S) >= K:
                break
            if all((i + s) not in [a + b for a, b in S if a < b] and 
                   (i - s) not in [a - b for a, b in S if a > b] for s in S):
                S.append(i)
        return S
    
    # Try various sparse configurations
    for K in [1, 2, 3, 5, 8, 10]:
        if time.time() > deadline:
            break
        sidon = make_sidon_set(K, N)
        if len(sidon) < K:
            continue
        h = np.zeros(N)
        h[sidon] = 1.0
        score = compute_ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.tolist()
    
    # Local optimization from best starting point
    if time.time() < deadline - 5.0:
        def neg_ratio(x):
            h = np.maximum(x, 0)
            return -compute_ratio(h)
        
        def neg_ratio_grad(x):
            h = np.maximum(x, 0)
            # Simple numerical gradient
            eps = 1e-7
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x2 = x.copy()
                x2[i] += eps
                grad[i] = (neg_ratio(x2) - neg_ratio(x)) / eps
            return grad
        
        result = minimize(neg_ratio, np.array(best_h), method='L-BFGS-B',
                         options={'maxiter': 50, 'ftol': 1e-10})
        if time.time() < deadline:
            h_opt = np.maximum(result.x, 0)
            score = compute_ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt.tolist()
    
    # Final check
    if time.time() > deadline - 2.0:
        return best_h
    
    return best_h
