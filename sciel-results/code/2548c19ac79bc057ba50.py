import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

# Precompute FFT size for linear convolution of two length N sequences
# Result length is 2N - 1
FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    F = np.fft.rfft(f, n=FFT_SIZE)
    C = np.fft.irfft(F * F, n=FFT_SIZE)
    return C[:2 * N - 1]

def _objective_vectorized(f):
    """
    Compute max(|autoconv(f)|) / (sum(f))^2.
    If sum(f) == 0, return a large penalty.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _objective_smooth(f, t=0.01):
    """
    Smooth surrogate using log-sum-exp for the max.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    absC = np.abs(C)
    max_val = np.max(absC)
    # Numerical stability: subtract max before exp
    lse = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    return (lse / t) / (S * S)

def _gradient_smooth(f, t=0.01):
    """
    Gradient of the smooth objective using FFT-based convolution.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.zeros_like(f)
    
    C = _autoconv_fft(f)
    absC = np.abs(C)
    signC = np.sign(C)
    max_val = np.max(absC)
    
    exp_term = np.exp(t * (absC - max_val))
    sum_exp = np.sum(exp_term)
    weights = exp_term / sum_exp
    
    A = weights * signC
    
    # Compute gradient via direct dot products
    # grad[i] = 2 * sum_j A[i+j] * f[j]
    grad = np.zeros(N)
    for i in range(N):
        k_start = i
        k_end = min(i + N, 2 * N - 1)
        if k_start < len(A) and k_end > k_start:
            f_slice = f[:k_end - k_start]
            A_slice = A[k_start:k_end]
            grad[i] = 2.0 * np.dot(A_slice, f_slice)
    
    LSE_val = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    G_val = LSE_val / t
    
    # d/d f [ G / S^2 ] = (1/S^2) dG - G * d(1/S^2)
    # d(1/S^2) = -2 S^{-3}
    # dG = grad
    return (1.0 / (S * S)) * grad - (G_val / (S * S * S)) * 2.0 * S

def _local_search_binary_fast(f_init, time_budget=10.0):
    """
    Local search on binary {-1, 1} sequences using incremental convolution updates.
    This is significantly faster than recomputing FFT for each flip.
    """
    f = np.sign(f_init).astype(np.float64)
    # Ensure no zeros
    f[f == 0] = 1.0
    
    S = np.sum(f)
    if abs(S) < 1e-6:
        return f, 1e10
        
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_budget:
        if time.time() > DEADLINE:
            break
            
        best_improvement = 0
        best_flip_idx = -1
        best_new_C = None
        best_new_S = S
        
        # Try flipping each position
        # N=400 is small enough to try all
        for i in range(N):
            v = f[i]
            # delta = -2 * v
            # C_new[k] = C[k] - 4 * v * f[k-i] + 4 * delta_{k, 2i}
            
            # We only need to check if this flip improves the score.
            # We can construct C_new efficiently.
            
            # Optimization: Instead of copying full C every time, 
            # we can calculate the new max on the fly? 
            # No, max is global. But we can update a copy.
            # Copying 800 floats is fast.
            
            C_new = C.copy()
            
            # Apply linear change: -4 * v * f[k-i]
            # k ranges from i to i+N-1
            k_start = i
            k_end = i + N
            
            if k_start < len(C_new):
                end_idx = min(k_end, len(C_new))
                # f[k-i] corresponds to f[0 : end_idx - k_start]
                f_segment = f[:end_idx - k_start]
                C_new[k_start:end_idx] -= 4.0 * v * f_segment
                
            # Apply quadratic change: +4 at k=2i
            if 2 * i < len(C_new):
                C_new[2 * i] += 4.0
                
            new_S = S - 2.0 * v
            if abs(new_S) < 1e-6:
                continue
                
            new_score = np.max(np.abs(C_new)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_flip_idx = i
                best_new_C = C_new
                best_new_S = new_S
        
        if best_flip_idx >= 0:
            v = f[best_flip_idx]
            f[best_flip_idx] = -v
            S = best_new_S
            C = best_new_C
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 5:
                break
                
    return f, score

def _sparse_annealing(f_init, time_budget=15.0):
    """
    Simulated annealing on sparse signed sequences.
    """
    support = list(np.nonzero(f_init)[0])
    signs = [f_init[i] for i in support]
    
    def make_f(sup, sgn):
        f = np.zeros(N)
        for pos, s in zip(sup, sgn):
            f[pos] = s
        return f
    
    current_f = make_f(support, signs)
    current_score = _objective_vectorized(current_f)
    
    best_f = current_f.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 0.5
    cooling = 0.995
    
    while time.time() - start_time < time_budget and T > 0.01:
        if time.time() > DEADLINE:
            break
        
        move_type = np.random.random()
        
        if move_type < 0.3 and len(support) < 30:
            pos = np.random.randint(N)
            if pos not in support:
                sign = 1.0 if np.random.random() < 0.5 else -1.0
                support.append(pos)
                signs.append(sign)
        elif move_type < 0.6 and len(support) > 5:
            idx = np.random.randint(0, len(support))
            support.pop(idx)
            signs.pop(idx)
        elif move_type < 0.9 and len(support) > 0:
            idx = np.random.randint(0, len(support))
            signs[idx] = -signs[idx]
        else:
            if len(support) == 0:
                continue
            idx = np.random.randint(0, len(support))
            old_pos = support[idx]
            new_pos = np.random.randint(N)
            if new_pos not in support:
                support[idx] = new_pos
            else:
                continue
        
        new_f = make_f(support, signs)
        new_score = _objective_vectorized(new_f)
        
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_f = new_f
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = new_f.copy()
        else:
            # Revert
            if move_type < 0.3 and len(support) > len(f_init[np.nonzero(f_init)[0]]):
                support.pop()
                signs.pop()
            elif move_type < 0.6:
                pass
            elif move_type < 0.9:
                signs[idx] = -signs[idx]
            else:
                support[idx] = old_pos
        
        T *= cooling
    
    return best_f, best_score

def _polish_with_tighter_t(f, time_budget=8.0):
    """
    Final polish pass with tighter smoothness parameter for higher precision.
    """
    start_time = time.time()
    current_f = f.copy()
    current_score = _objective_vectorized(current_f)
    
    ts = [0.005, 0.002, 0.001]
    
    for t in ts:
        if time.time() > DEADLINE or time.time() - start_time > time_budget:
            break
        
        try:
            res = minimize(
                _objective_smooth,
                current_f,
                args=(t,),
                method='L-BFGS-B',
                jac=_gradient_smooth,
                options={'maxiter': 50, 'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            true_score = _objective_vectorized(res.x)
            if true_score < current_score:
                current_f = res.x
                current_score = true_score
        except Exception:
            continue
    
    return current_f, current_score

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _objective_vectorized(best_f)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Continuous optimization with smooth objective
    # Use higher betas for sharper max approximation
    for restart in range(6):
        if time.time() > DEADLINE:
            break
        
        if restart == 0:
            f0 = np.ones(N)
        elif restart == 1:
            f0 = np.sin(np.linspace(0, 2 * np.pi, N))
        elif restart == 2:
            f0 = np.cos(np.linspace(0, 4 * np.pi, N))
        else:
            f0 = rng.uniform(-1, 1, N)
        
        S0 = np.sum(f0)
        if abs(S0) > 1e-6:
            f0 = f0 / S0
        else:
            f0 = np.ones(N)
        
        ts = [10.0, 50.0, 100.0]
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
    
    # Phase 2: Binary local search with FFT (Fast Incremental)
    for restart in range(6):
        if time.time() > DEADLINE:
            break
        
        if restart == 0:
            f0 = np.ones(N)
        elif restart == 1:
            f0 = np.sign(np.sin(np.linspace(0, 2 * np.pi, N)))
            f0[f0 == 0] = 1.0
        elif restart == 2:
            f0 = np.sign(np.cos(np.linspace(0, 4 * np.pi, N)))
            f0[f0 == 0] = 1.0
        else:
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
        time_budget = min(10.0, remaining_time * 0.6)
        
        if time_budget < 0.5:
            break
        
        f_local, score_local = _local_search_binary_fast(f0, time_budget=time_budget)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()
    
    # Phase 3: Sparse annealing
    for restart in range(5):
        if time.time() > DEADLINE:
            break
        
        K = rng.integers(10, 25)
        support = list(rng.choice(N, size=K, replace=False))
        signs = [1.0 if rng.random() < 0.6 else -1.0 for _ in range(K)]
        
        S = sum(signs)
        if abs(S) < 3:
            signs = [1.0] * K
        
        f_sparse = np.zeros(N)
        for pos, s in zip(support, signs):
            f_sparse[pos] = s
        
        remaining_time = DEADLINE - time.time()
        time_budget = min(12.0, remaining_time * 0.4)
        
        if time_budget < 1.0:
            break
        
        f_sa, score_sa = _sparse_annealing(f_sparse, time_budget=time_budget)
        
        if score_sa < best_score:
            best_score = score_sa
            best_f = f_sa.copy()
    
    # Phase 4: Real-valued polish on best found
    if time.time() < DEADLINE:
        remaining_time = DEADLINE - time.time()
        if remaining_time > 5:
            time_budget = min(8.0, remaining_time * 0.5)
            f_start = best_f.copy()
            # Simple real local search
            f_local, score_local = _local_search_real(f_start, time_budget=time_budget, step=0.5)
            if score_local < best_score:
                best_score = score_local
                best_f = f_local.copy()
    
    # Phase 5: High-precision polish with tighter t
    if time.time() < DEADLINE:
        remaining_time = DEADLINE - time.time()
        if remaining_time > 3:
            time_budget = min(8.0, remaining_time * 0.7)
            f_polished, score_polished = _polish_with_tighter_t(best_f, time_budget=time_budget)
            if score_polished < best_score:
                best_score = score_polished
                best_f = f_polished.copy()
    
    return best_f

def _local_search_real(f, time_budget=10.0, step=0.5):
    """
    Local search on real-valued sequences.
    """
    f = f.copy()
    S = np.sum(f)
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_budget:
        if time.time() > DEADLINE:
            break
        
        best_improvement = 0
        best_change = None
        
        trial_positions = np.random.choice(N, size=min(50, N), replace=False)
        
        for i in trial_positions:
            for delta in [-step, step]:
                f[i] += delta
                new_S = S + delta
                if abs(new_S) < 1e-6:
                    f[i] -= delta
                    continue
                
                C = _autoconv_fft(f)
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
            C = _autoconv_fft(f)
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 30:
                step *= 0.7
                if step < 0.01:
                    break
    
    return f, score
