import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN


def _autoconv(f):
    """Compute the linear (full) autoconvolution of f."""
    return np.convolve(f, f, mode='full')


def _objective_vectorized(f):
    """
    Compute max(|autoconv(f)|) / (sum(f))^2.
    If sum(f) == 0, return a large penalty.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    return np.max(np.abs(C)) / (S * S)


def _objective_smooth(f, t=0.01):
    """
    Smooth surrogate using log-sum-exp for the max.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    absC = np.abs(C)
    max_val = np.max(absC)
    lse = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    return (lse / t) / (S * S)


def _gradient_smooth(f, t=0.01):
    """
    Gradient of the smooth objective.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.zeros_like(f)
    
    C = _autoconv(f)
    absC = np.abs(C)
    signC = np.sign(C)
    max_val = np.max(absC)
    
    exp_term = np.exp(t * (absC - max_val))
    sum_exp = np.sum(exp_term)
    weights = exp_term / sum_exp
    
    A = weights * signC
    
    grad = np.zeros_like(f)
    for i in range(N):
        k_start = i
        k_end = i + N
        if k_start < len(A) and k_end <= len(A):
            grad[i] = 2.0 * np.dot(A[k_start:k_end], f)
        elif k_start < len(A):
            grad[i] = 2.0 * np.dot(A[k_start:], f[:len(A)-k_start])
    
    LSE_val = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    G_val = LSE_val / t
    
    return (1.0 / (S * S)) * grad - (G_val / (S * S * S)) * 2.0 * S


def _local_search_binary(f, time_budget=100.0):
    """
    Local search on binary {-1, 1} sequences.
    Uses vectorized evaluation of candidate flips for speed.
    """
    f = f.copy()
    S = np.sum(f)
    C = _autoconv(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_budget:
        if time.time() > DEADLINE:
            break
            
        k_max = np.argmax(np.abs(C))
        
        candidates = set()
        for j in range(N):
            if 0 <= k_max - j < N:
                candidates.add(j)
        for _ in range(10):
            candidates.add(np.random.randint(N))
            
        candidates = list(candidates)
        M = len(candidates)
        
        if M == 0:
            break
            
        # Create delta matrix
        D = np.zeros((M, 2*N - 1))
        S_new_arr = np.zeros(M)
        
        for m, i in enumerate(candidates):
            v = f[i]
            S_new_arr[m] = S - 2.0 * v
            
            k_start = i
            k_end = min(i + N, 2 * N - 1)
            k_end_clamped = min(k_end, 2 * N - 1)
            
            f_slice = f[:k_end_clamped - k_start]
            delta = -2.0 * v * f_slice
            
            D[m, k_start:k_end_clamped] = delta
            
            # Fix k=2i case where change is 0
            k_center = 2 * i
            if k_start <= k_center < k_end_clamped:
                D[m, k_center] = 0.0
                
        # Compute new scores
        C_new = C[None, :] + D
        max_abs_C = np.max(np.abs(C_new), axis=1)
        
        # Avoid division by zero
        valid = np.abs(S_new_arr) > 1e-6
        scores = np.full(M, 1e10)
        scores[valid] = max_abs_C[valid] / (S_new_arr[valid] ** 2)
        
        min_idx = np.argmin(scores)
        min_score = scores[min_idx]
        
        if min_score < score:
            i = candidates[min_idx]
            v = f[i]
            f[i] = -v
            S = S_new_arr[min_idx]
            C = C_new[min_idx].copy()
            score = min_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                break
                
    return f, score


def _local_search_real(f, max_iter=2000, time_budget=8.0, step=0.5):
    """
    Local search on real-valued sequences.
    Adjusts individual values by ±step, accepting if it improves the objective.
    """
    f = f.copy()
    S = np.sum(f)
    C = _autoconv(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        best_improvement = 0
        best_change = None
        
        # Try a random subset of positions
        trial_positions = np.random.choice(N, size=min(50, N), replace=False)
        
        for i in trial_positions:
            for delta in [-step, step]:
                f[i] += delta
                new_S = S + delta
                if abs(new_S) < 1e-6:
                    f[i] -= delta
                    continue
                
                C = _autoconv(f)
                new_score = np.max(np.abs(C)) / (new_S * new_S)
                
                if new_score < score - best_improvement:
                    best_improvement = score - new_score
                    best_change = (i, delta)
                
                f[i] -= delta
                S = np.sum(f)
        
        if best_change is not None:
            i, delta = best_change
            f[i] += delta
            S += delta
            C = _autoconv(f)
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 30:
                step *= 0.7
                if step < 0.01:
                    break
    
    return f, score


def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _objective_vectorized(best_f)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Multiple restarts of vectorized binary local search.
    # Start with all 1s, and also with perturbations of all 1s.
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > DEADLINE:
            break
            
        # Generate initial state
        if r == 0:
            f0 = np.ones(N)
        else:
            # Start with all 1s, flip k random bits
            # k should be small to keep sum high
            k = rng.integers(0, N // 4)
            f0 = np.ones(N)
            if k > 0:
                idxs = rng.choice(N, size=k, replace=False)
                f0[idxs] = -1.0
        
        S0 = np.sum(f0)
        if abs(S0) < 5:
            continue
            
        remaining_time = DEADLINE - time.time()
        time_budget = max(0.1, remaining_time / (num_restarts - r))
        
        f_local, score_local = _local_search_binary(f0, time_budget=time_budget)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()
            
    return best_f
