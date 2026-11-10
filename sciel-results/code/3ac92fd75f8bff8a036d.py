import numpy as np
import time
from scipy.optimize import linprog

def _compute_mobius(K):
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def _verify_feasibility(f1, f_vals, ks, deadline=None):
    """Verify M(x) <= 1.0001 for all x in [1, 10*K] using chunked computation."""
    K = int(ks.max())
    max_x = 10 * K
    m = len(ks)
    
    chunk_size = max(1, 10000000 // m)
    max_val = -np.inf
    
    for start in range(1, max_x + 1, chunk_size):
        end = min(start + chunk_size, max_x + 1)
        xs = np.arange(start, end)
        floors = np.floor(xs[:, None] / ks[None, :]).astype(np.int64)
        M_vals = f1 * xs + floors @ f_vals
        chunk_max = np.max(M_vals)
        if chunk_max > max_val:
            max_val = chunk_max
        if max_val > 1.0001:
            return False
        if deadline is not None and time.time() > deadline:
            return False
    
    return max_val <= 1.0001

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 590.0
    
    def try_K(K_max):
        if time.time() > deadline:
            return None
        ks = np.arange(2, K_max + 1)
        m = len(ks)
        max_x = 10 * K_max
        
        # Objective: minimize sum f_k log(k)/k (equivalent to maximizing -sum f_k log(k)/k)
        c_obj = np.log(ks) / ks
        
        # Constraint: M(x) = sum_k f_k * (floor(x/k) - x/k) = -sum_k f_k * {x/k} <= 1.0001
        # So A_ub row x is -{x/k} for k in ks, b_ub = 1.0001
        A_ub_chunks = []
        chunk_size = max(1, 10000000 // m)
        
        for start in range(1, max_x + 1, chunk_size):
            end = min(start + chunk_size, max_x + 1)
            xs = np.arange(start, end)
            xk = xs[:, None] / ks[None, :]
            frac = xk - np.floor(xk)
            A_ub_chunks.append(-frac)
            if time.time() > deadline:
                return None
        
        A_ub = np.vstack(A_ub_chunks)
        b_ub = np.full(max_x, 1.0001)
        bounds = [(-10, 10)] * m
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                f_vals = np.clip(res.x, -10, 10)
                S = np.sum(f_vals / ks)
                f1 = np.clip(-S, -10, 10)
                
                if time.time() > deadline:
                    return None
                if _verify_feasibility(f1, f_vals, ks, deadline):
                    domain = np.concatenate(([1], ks))
                    values = np.concatenate(([f1], f_vals))
                    return domain, values
        except Exception:
            pass
        return None
    
    for K_max in [50, 100, 150, 200, 300, 400, 500]:
        result = try_K(K_max)
        if result is not None:
            return result
    
    K = 200000
    mu = _compute_mobius(K)
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[domain].astype(np.float64)
    return domain, values
