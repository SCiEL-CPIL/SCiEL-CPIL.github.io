import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

# Precompute FFT size
FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    F = np.fft.rfft(f, n=FFT_SIZE)
    C = np.fft.irfft(F * F, n=FFT_SIZE)
    return C[:2 * N - 1]

def _score(f):
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _local_search_binary_fast(f, max_iter=1000, time_budget=5.0):
    """
    Local search on binary {-1, 1} sequences using incremental convolution updates.
    """
    f = np.sign(f).astype(np.float64)
    S = np.sum(f)
    if abs(S) < 1e-6:
        return f, 1e10
        
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
            
        best_improvement = 0
        best_flip_idx = -1
        best_new_C = None
        best_new_S = S
        
        # Try flipping each position? N=400 is small.
        # Or a random subset if it's too slow.
        # With O(N) update, 400 flips * 400 ops = 160k ops per iteration.
        # 1000 iterations = 160M ops. Should be fast in numpy if vectorized?
        # But the update depends on the specific index i.
        # We can vectorize the evaluation of all flips?
        # C_new[i] = C + 2*(-2*f[i]) * shift(f, i) + ...
        # This is complex to vectorize fully because the shift is different for each i.
        # Let's just loop over i, it's fast enough.
        
        # Optimization: Only check a subset if time is tight?
        # Let's try all positions first.
        
        for i in range(N):
            v = f[i]
            # Flip f[i]
            # delta = -2 * v
            delta = -2.0 * v
            
            # Update C
            # C_new[k] = C[k] + 2 * delta * f[k-i] + delta^2 * (1 if k==2i else 0)
            
            # We need to compute max abs of C_new.
            # Instead of constructing C_new fully, we can check if it improves.
            # But we need the new C for the next step if we accept.
            
            # Let's construct C_new efficiently.
            # The term 2 * delta * f[k-i] is a shifted version of f.
            # Let's just do it in a loop for clarity and speed of implementation, 
            # relying on the fact that N=400 is small.
            
            # Actually, we can precompute the shift effects? No.
            
            # Let's just compute the new score for this flip.
            # We need C_new.
            # C_new = C.copy()
            # This is O(N).
            # Then update the slice.
            
            # Let's try to be clever.
            # We only care if max |C_new| / S_new^2 < score.
            
            # Approximation:
            # If we don't want to copy C every time, we can compute the change.
            # But max is a global property.
            
            # Let's just do the copy and update. 400 * 400 = 160k float ops.
            # In Python, a loop of 400 iterations, each doing numpy ops on arrays of size 800.
            # This might be slow if not careful.
            
            # Alternative: Use FFT for the check? No, that defeats the purpose.
            
            # Let's try a simpler approach: 
            # Evaluate the flip by constructing the new C.
            
            C_new = C.copy()
            
            # Apply linear change: 2 * delta * f[k-i]
            # k ranges from 0 to 2N-2.
            # f[k-i] is valid if 0 <= k-i < N => i <= k < i+N.
            
            k_start = i
            k_end = i + N
            
            if k_start < len(C_new):
                # f[0 : N-k_start] corresponds to f[k-i] for k in [i, i+N)
                # Wait.
                # k = i + j. f[k-i] = f[j].
                # j ranges from 0 to N-1.
                # k ranges from i to i+N-1.
                
                # Check bounds
                end_idx = min(k_end, len(C_new))
                start_idx = k_start
                
                # The segment of f is f[0 : end_idx - start_idx]
                # The segment of C is C[start_idx : end_idx]
                
                f_segment = f[:end_idx - start_idx]
                
                C_new[start_idx:end_idx] += 2.0 * delta * f_segment
                
            # Apply quadratic change: delta^2 at k=2i
            if 2 * i < len(C_new):
                C_new[2 * i] += delta * delta
                
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
            if no_improve_count > 10:
                break
                
    return f, score

def _objective_soft(f, beta=10.0):
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    absC = np.abs(C)
    max_val = np.max(absC)
    lse = max_val + np.log(np.sum(np.exp(beta * (absC - max_val))))
    return (lse / beta) / (S * S)

def _gradient_soft(f, beta=10.0):
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.zeros_like(f)
    
    C = _autoconv_fft(f)
    absC = np.abs(C)
    signC = np.sign(C)
    
    max_val = np.max(absC)
    exp_term = np.exp(beta * (absC - max_val))
    sum_exp = np.sum(exp_term)
    weights = exp_term / sum_exp
    
    A = weights * signC
    
    grad_LSE = np.zeros(N)
    for i in range(N):
        k_start = i
        k_end = i + N
        if k_start < len(A) and k_end <= len(A):
            grad_LSE[i] = 2.0 * np.dot(A[k_start:k_end], f)
        elif k_start < len(A):
            grad_LSE[i] = 2.0 * np.dot(A[k_start:], f[:len(A)-k_start])
            
    LSE_val = np.max(absC) + np.log(np.sum(np.exp(beta * (absC - np.max(absC)))))
    G_val = LSE_val / beta
    
    return (1.0 / (S * S)) * grad_LSE - (2.0 * G_val / (S * S * S)) * S

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    rng = np.random.default_rng(42)
    
    # 1. Continuous Optimization to find a good basin
    initializations = [
        np.ones(N),
        np.sin(2 * np.pi * np.arange(N) / N),
        np.cos(2 * np.pi * np.arange(N) / N),
        rng.uniform(-1, 1, N),
        rng.choice([-1.0, 1.0], size=N)
    ]
    
    for i in range(len(initializations)):
        S_init = np.sum(initializations[i])
        if abs(S_init) < 1e-6:
            initializations[i] = initializations[i] + 0.1 * rng.normal(size=N)
            S_init = np.sum(initializations[i])
        initializations[i] = initializations[i] / S_init

    for f0 in initializations:
        if time.time() > DEADLINE:
            break
            
        current_f = f0.copy()
        betas = [5.0, 20.0, 50.0]
        
        for beta in betas:
            if time.time() > DEADLINE:
                break
            try:
                res = minimize(
                    _objective_soft,
                    current_f,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=_gradient_soft,
                    options={'maxiter': 20, 'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                S_opt = np.sum(res.x)
                if abs(S_opt) < 1e-6:
                    continue
                    
                score_opt = _score(res.x)
                if score_opt < best_score:
                    best_score = score_opt
                    best_f = res.x.copy()
                
                current_f = res.x
            except Exception:
                continue

    # 2. Binary Local Search from best continuous solution
    f_bin = np.sign(best_f)
    if np.sum(f_bin) == 0:
        f_bin = np.ones(N)
        
    f_bin_local, score_bin_local = _local_search_binary_fast(f_bin, max_iter=2000, time_budget=20.0)
    
    if score_bin_local < best_score:
        best_score = score_bin_local
        best_f = f_bin_local.copy()

    # 3. Random Binary Restarts with Local Search
    # Use different densities of 1s to ensure S is not too small
    densities = [0.9, 0.8, 0.7, 0.5]
    
    for d in densities:
        for _ in range(5):
            if time.time() > DEADLINE:
                break
            
            # Generate random binary sequence with bias towards 1
            # p=0.5 for symmetric, but we want high sum.
            # Let's just use random and check sum.
            f_rand = rng.choice([-1.0, 1.0], size=N)
            S_rand = np.sum(f_rand)
            if abs(S_rand) < N * 0.2: # Skip if sum is too small
                continue
                
            f_rand_local, score_rand_local = _local_search_binary_fast(f_rand, max_iter=1000, time_budget=10.0)
            
            if score_rand_local < best_score:
                best_score = score_rand_local
                best_f = f_rand_local.copy()

    return best_f
