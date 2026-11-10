import numpy as np
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    V_minus1 = X - Y
    V_0 = X
    V_1 = X + Y
    V_2 = X + 2 * Y
    
    idx_minus1 = (V_minus1 - np.min(V_minus1)).flatten()
    idx_0 = (V_0 - np.min(V_0)).flatten()
    idx_1 = (V_1 - np.min(V_1)).flatten()
    idx_2 = (V_2 - np.min(V_2)).flatten()
    
    size_minus1 = np.max(idx_minus1) + 1
    size_0 = np.max(idx_0) + 1
    size_1 = np.max(idx_1) + 1
    size_2 = np.max(idx_2) + 1

    def entropy_from_binned(p_binned):
        p_nonzero = p_binned[p_binned > 1e-15]
        if len(p_nonzero) == 0:
            return 0.0
        return -np.sum(p_nonzero * np.log2(p_nonzero))

    def score(P):
        p_flat = P.flatten()
        p_m1 = np.bincount(idx_minus1, weights=p_flat, minlength=size_minus1)
        p_0 = np.bincount(idx_0, weights=p_flat, minlength=size_0)
        p_1 = np.bincount(idx_1, weights=p_flat, minlength=size_1)
        p_2 = np.bincount(idx_2, weights=p_flat, minlength=size_2)
        
        D = entropy_from_binned(p_m1)
        M0 = entropy_from_binned(p_0)
        M1 = entropy_from_binned(p_1)
        M2 = entropy_from_binned(p_2)
        M = max(M0, M1, M2)
        
        if M < 1e-9:
            return 0.0
        return D / M

    best_P = np.ones((n, n)) / (n * n)
    best_score = score(best_P)
    
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 10.0
    
    P = best_P.copy()
    cur_score = best_score
    
    T = 0.05
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit - safety_margin:
        for _ in range(2000):
            if time.time() - start_time > time_limit - safety_margin:
                break
            
            i = np.random.randint(n*n)
            j = np.random.randint(n*n)
            if i == j:
                continue
            
            step_size = np.random.exponential(0.05) * P[i]
            if step_size > P[i]:
                step_size = P[i] * 0.5
            if step_size < 1e-12:
                continue
                
            new_P = P.copy()
            new_P.flat[i] -= step_size
            new_P.flat[j] += step_size
            
            new_score = score(new_P)
            
            if new_score > cur_score:
                P = new_P
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_P = P.copy()
            else:
                delta = cur_score - new_score
                if delta < 1e-9:
                    accept_prob = 1.0
                else:
                    accept_prob = np.exp(-delta / T)
                    
                if np.random.random() < accept_prob:
                    P = new_P
                    cur_score = new_score
        
        T *= cooling_rate
        if T < 1e-5:
            P = np.random.dirichlet(np.ones(n*n) * 0.5)
            P = P.reshape(n, n)
            cur_score = score(P)
            if cur_score > best_score:
                best_score = cur_score
                best_P = P.copy()
            T = 0.05

    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s < 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P /= s
        
    return best_P
