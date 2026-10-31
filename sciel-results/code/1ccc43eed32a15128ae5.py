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


def _local_search_binary(f, max_iter=2000, time_budget=8.0):
    """
    Local search on binary {-1, 1} sequences.
    Flips one sign at a time, accepting if it improves the objective.
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
        
        k_max = np.argmax(np.abs(C))
        
        candidates = []
        
        for _ in range(10):
            candidates.append(np.random.randint(N))
        
        for j in range(N):
            if 0 <= k_max - j < N:
                impact = -2.0 * f[j] * f[k_max - j]
                if abs(impact) > 0:
                    candidates.append(j)
        
        candidates = list(set(candidates))
        
        best_improvement = 0
        best_flip_idx = -1
        
        for i in candidates:
            v = f[i]
            k_start = i
            k_end = min(i + N, 2 * N - 1)
            
            if k_start >= len(C) or k_end <= 0:
                continue
            
            k_end_clamped = min(k_end, len(C))
            f_slice = f[:k_end_clamped - k_start]
            
            delta_C = np.zeros_like(C)
            delta_C[k_start:k_end_clamped] = -2.0 * v * f_slice
            
            new_C = C + delta_C
            new_S = S - 2.0 * v
            
            if abs(new_S) < 1e-6:
                continue
            
            new_score = np.max(np.abs(new_C)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_flip_idx = i
        
        if best_flip_idx >= 0:
            v = f[best_flip_idx]
            f[best_flip_idx] = -v
            S -= 2.0 * v
            C = _autoconv(f)
            score = np.max(np.abs(C)) / (S * S)
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
    
    # Continuous optimization with smooth objective
    for restart in range(5):
        if time.time() > DEADLINE:
            break
        
        f0 = rng.uniform(-1, 1, N)
        if abs(np.sum(f0)) > 1e-6:
            f0 = f0 / np.sum(f0)
        else:
            f0 = np.ones(N)
        
        ts = [0.1, 0.05, 0.01, 0.005]
        current_f = f0
        
        for t in ts:
            if time.time() > DEADLINE:
                break
            
            try:
                res = minimize(
                    _objective_smooth,
                    current_f,
                    args=(t,),
                    method='L-BFGS-B',
                    jac=_gradient_smooth,
                    options={'maxiter': 30, 'maxls': 30, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if res.fun < best_score:
                    true_score = _objective_vectorized(res.x)
                    if true_score < best_score:
                        best_score = true_score
                        best_f = res.x.copy()
                    current_f = res.x
            except Exception:
                continue
    
    # Binary local search
    for restart in range(8):
        if time.time() > DEADLINE:
            break
        
        f0 = rng.choice([-1.0, 1.0], size=N)
        S0 = np.sum(f0)
        
        if abs(S0) < 5:
            for _ in range(N // 2):
                idx = rng.integers(0, N)
                if f0[idx] * S0 < 0:
                    f0[idx] = -f0[idx]
                    S0 = np.sum(f0)
            if abs(S0) < 5:
                continue
        
        remaining_time = DEADLINE - time.time()
        time_budget = min(8.0, remaining_time * 0.8)
        
        if time_budget < 0.5:
            break
        
        f_local, score_local = _local_search_binary(f0, max_iter=3000, time_budget=time_budget)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()
    
    # Real-valued local search from best found
    if time.time() < DEADLINE:
        remaining_time = DEADLINE - time.time()
        if remaining_time > 10:
            time_budget = min(10.0, remaining_time * 0.8)
            f_start = best_f.copy()
            f_local, score_local = _local_search_real(f_start, max_iter=2000, time_budget=time_budget, step=0.5)
            if score_local < best_score:
                best_score = score_local
                best_f = f_local.copy()
    
    return best_f
