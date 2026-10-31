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
    Supports single and double flips for better escape from local minima.
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
    max_no_improve = 8
    
    while time.time() - start_time < time_budget:
        if time.time() > DEADLINE:
            break
            
        best_improvement = 0
        best_move = None  # (indices, new_C, new_S)
        
        # Try single flips
        for i in range(N):
            v = f[i]
            C_new = C.copy()
            k_start = i
            k_end = i + N
            
            if k_start < len(C_new):
                end_idx = min(k_end, len(C_new))
                f_segment = f[:end_idx - k_start]
                C_new[k_start:end_idx] -= 4.0 * v * f_segment
            if 2 * i < len(C_new):
                C_new[2 * i] += 4.0
                
            new_S = S - 2.0 * v
            if abs(new_S) < 1e-6:
                continue
                
            new_score = np.max(np.abs(C_new)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_move = ([i], C_new, new_S)
        
        # If no single flip helps, try double flips (sampled for speed)
        if best_move is None and no_improve_count >= 3:
            # Sample pairs to try
            n_pairs = min(50, N * (N-1) // 2)
            pair_indices = []
            for _ in range(n_pairs):
                i1 = np.random.randint(0, N)
                i2 = np.random.randint(0, N)
                if i1 != i2:
                    pair_indices.append((i1, i2))
            
            for i1, i2 in pair_indices:
                if time.time() > DEADLINE:
                    break
                v1 = f[i1]
                v2 = f[i2]
                
                C_new = C.copy()
                
                # Apply flip at i1
                k_start1 = i1
                k_end1 = i1 + N
                if k_start1 < len(C_new):
                    end_idx1 = min(k_end1, len(C_new))
                    C_new[k_start1:end_idx1] -= 4.0 * v1 * f[:end_idx1 - k_start1]
                if 2 * i1 < len(C_new):
                    C_new[2 * i1] += 4.0
                    
                # Apply flip at i2
                k_start2 = i2
                k_end2 = i2 + N
                if k_start2 < len(C_new):
                    end_idx2 = min(k_end2, len(C_new))
                    C_new[k_start2:end_idx2] -= 4.0 * v2 * f[:end_idx2 - k_start2]
                if 2 * i2 < len(C_new):
                    C_new[2 * i2] += 4.0
                    
                new_S = S - 2.0 * v1 - 2.0 * v2
                if abs(new_S) < 1e-6:
                    continue
                
                new_score = np.max(np.abs(C_new)) / (new_S * new_S)
                
                if new_score < score - best_improvement:
                    best_improvement = score - new_score
                    best_move = ([i1, i2], C_new, new_S)
        
        if best_move is not None:
            indices, C_new, new_S = best_move
            for idx in indices:
                f[idx] = -f[idx]
            S = new_S
            C = C_new
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
                
    return f, score

def _sparse_annealing(f_init, time_budget=15.0):
    """
    Simulated annealing on sparse signed sequences with adaptive temperature.
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
    T = 1.0
    cooling = 0.99
    min_T = 0.005
    
    while time.time() - start_time < time_budget and T > min_T:
        if time.time() > DEADLINE:
            break
        
        move_type = np.random.random()
        old_state = (support[:], signs[:])
        
        if move_type < 0.25 and len(support) < 40:
            pos = np.random.randint(N)
            if pos not in support:
                sign = 1.0 if np.random.random() < 0.5 else -1.0
                support.append(pos)
                signs.append(sign)
        elif move_type < 0.50 and len(support) > 3:
            idx = np.random.randint(0, len(support))
            support.pop(idx)
            signs.pop(idx)
        elif move_type < 0.80 and len(support) > 0:
            idx = np.random.randint(0, len(support))
            signs[idx] = -signs[idx]
        else:
            if len(support) == 0:
                T *= cooling
                continue
            idx = np.random.randint(0, len(support))
            old_pos = support[idx]
            new_pos = np.random.randint(N)
            if new_pos not in support:
                support[idx] = new_pos
            else:
                T *= cooling
                continue
        
        new_f = make_f(support, signs)
        new_score = _objective_vectorized(new_f)
        
        delta = new_score - current_score
        if delta < 0 or (T > min_T and np.random.random() < np.exp(-delta / T)):
            current_f = new_f
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = new_f.copy()
        else:
            # Revert
            support, signs = old_state
        
        T *= cooling
    
    return best_f, best_score

def _polish_with_tighter_t(f, time_budget=8.0):
    """
    Final polish pass with tighter smoothness parameter for higher precision.
    """
    start_time = time.time()
    current_f = f.copy()
    current_score = _objective_vectorized(current_f)
    
    ts = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
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
                options={'maxiter': 80, 'maxls': 80, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            true_score = _objective_vectorized(res.x)
            if true_score < current_score:
                current_f = res.x
                current_score = true_score
        except Exception:
            continue
    
    return current_f, current_score

def _local_search_real_fast(f_init, time_budget=10.0, step=0.5):
    """
    Local search on real-valued sequences with incremental FFT updates.
    Much faster than recomputing full FFT for each trial.
    """
    f = f_init.copy().astype(np.float64)
    S = np.sum(f)
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    current_step = step
    
    while time.time() - start_time < time_budget:
        if time.time() > DEADLINE:
            break
        
        best_improvement = 0
        best_change = None
        
        # Try a subset of positions for speed
        n_trials = min(80, N)
        trial_positions = np.random.choice(N, size=n_trials, replace=False)
        
        for i in trial_positions:
            if time.time() > DEADLINE:
                break
                
            for delta in [-current_step, current_step]:
                new_val = f[i] + delta
                if abs(new_val) < 1e-12:
                    continue
                    
                # Incremental update: C_new[k] = C[k] + 2*delta*f[k-i] + delta^2 * delta_{k,2i}
                # Actually: (f + delta*e_i) * (f + delta*e_i) = f*f + 2*delta*(f*e_i) + delta^2*(e_i*e_i)
                # e_i * e_i has 1 at position 2i
                # f * e_i has f[j] at position i+j
                C_new = C.copy()
                k_start = i
                k_end = i + N
                if k_start < len(C_new):
                    end_idx = min(k_end, len(C_new))
                    C_new[k_start:end_idx] += 2.0 * delta * f[:end_idx - k_start]
                if 2 * i < len(C_new):
                    C_new[2 * i] += delta * delta
                    
                new_S = S + delta
                if abs(new_S) < 1e-6:
                    continue
                
                new_score = np.max(np.abs(C_new)) / (new_S * new_S)
                
                if new_score < score - best_improvement:
                    best_improvement = score - new_score
                    best_change = (i, delta, C_new, new_S)
        
        if best_change is not None:
            i, delta, C_new, new_S = best_change
            f[i] += delta
            S = new_S
            C = C_new
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 20:
                current_step *= 0.7
                if current_step < 0.01:
                    break
                no_improve_count = 0
    
    return f, score

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _objective_vectorized(best_f)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Continuous optimization with smooth objective
    # Use higher betas for sharper max approximation
    for restart in range(8):
        if time.time() > DEADLINE:
            break
        
        if restart == 0:
            f0 = np.ones(N)
        elif restart == 1:
            f0 = np.sin(np.linspace(0, 2 * np.pi, N))
        elif restart == 2:
            f0 = np.cos(np.linspace(0, 4 * np.pi, N))
        elif restart == 3:
            f0 = np.sin(np.linspace(0, 3 * np.pi, N))
        elif restart == 4:
            f0 = np.cos(np.linspace(0, 6 * np.pi, N))
        else:
            f0 = rng.uniform(-1, 1, N)
        
        S0 = np.sum(f0)
        if abs(S0) > 1e-6:
            f0 = f0 / S0
        else:
            f0 = np.ones(N)
        
        ts = [10.0, 50.0, 100.0, 200.0]
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
                    options={'maxiter': 40, 'maxls': 40, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                true_score = _objective_vectorized(res.x)
                if true_score < best_score:
                    best_score = true_score
                    best_f = res.x.copy()
                current_f = res.x
            except Exception:
                continue
    
    # Phase 2: Binary local search with FFT (Fast Incremental)
    for restart in range(8):
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
        elif restart == 3:
            f0 = np.sign(np.sin(np.linspace(0, 3 * np.pi, N)))
            f0[f0 == 0] = 1.0
        elif restart == 4:
            f0 = np.sign(np.cos(np.linspace(0, 6 * np.pi, N)))
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
        time_budget = min(12.0, remaining_time * 0.5)
        
        if time_budget < 0.5:
            break
        
        f_local, score_local = _local_search_binary_fast(f0, time_budget=time_budget)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()
    
    # Phase 3: Sparse annealing
    for restart in range(6):
        if time.time() > DEADLINE:
            break
        
        K = rng.integers(8, 30)
        support = list(rng.choice(N, size=K, replace=False))
        signs = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(K)]
        
        S = sum(signs)
        if abs(S) < 3:
            signs = [1.0] * K
        
        f_sparse = np.zeros(N)
        for pos, s in zip(support, signs):
            f_sparse[pos] = s
        
        remaining_time = DEADLINE - time.time()
        time_budget = min(15.0, remaining_time * 0.3)
        
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
            time_budget = min(10.0, remaining_time * 0.4)
            f_start = best_f.copy()
            f_local, score_local = _local_search_real_fast(f_start, time_budget=time_budget, step=0.5)
            if score_local < best_score:
                best_score = score_local
                best_f = f_local.copy()
    
    # Phase 5: High-precision polish with tighter t
    if time.time() < DEADLINE:
        remaining_time = DEADLINE - time.time()
        if remaining_time > 3:
            time_budget = min(10.0, remaining_time * 0.6)
            f_polished, score_polished = _polish_with_tighter_t(best_f, time_budget=time_budget)
            if score_polished < best_score:
                best_score = score_polished
                best_f = f_polished.copy()
    
    return best_f
