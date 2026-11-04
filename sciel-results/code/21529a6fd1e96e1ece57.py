import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 110.0  # Safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    """Compute discrete autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')

def _ratio(h, beta=50.0):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf) using soft-max for peak."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0
    # Soft-max approximation of max using log-sum-exp
    shifted = beta * (conv - max_val)
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    log_sum_exp = max_shifted + np.log(np.sum(exp_shifted))
    soft_max = (log_sum_exp + beta * max_val) / beta
    if soft_max <= 0:
        return 0.0
    return l2_sq / soft_max

def _ratio_neg(h, beta=50.0):
    return -_ratio(h, beta)

def _dft_parameterize(z):
    """
    Parameterize heights as |DFT(z)|^2 where z is a complex vector.
    """
    H = np.fft.fft(z)
    h = np.abs(H) ** 2
    return h

def _dft_inverse(h):
    """Given h = |DFT(z)|^2, find a z (phase choice)."""
    H_mags = np.sqrt(h + 1e-30)
    H = H_mags.astype(complex)
    z = np.fft.ifft(H)
    return z

def _optimize_from_z(z0, start_time, max_iter=100, tol=1e-8):
    """Optimize over complex z to maximize ratio via |DFT(z)|^2."""
    z = z0.astype(complex).copy()
    
    def objective(z_flat):
        z_c = z_flat[:N] + 1j * z_flat[N:]
        h = _dft_parameterize(z_c)
        return _ratio_neg(h, beta=50.0)
    
    z_flat = np.concatenate([z.real, z.imag])
    
    result = minimize(objective, z_flat, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 20})
    
    z_opt = result.x[:N] + 1j * result.x[N:]
    h_opt = _dft_parameterize(z_opt)
    score = _ratio(h_opt)
    return h_opt, score

def _local_optimize(h0, start_time, max_iter=200, tol=1e-9):
    """Direct optimization of h with non-negativity constraint."""
    h = np.maximum(h0, 0.0).copy()
    
    def objective(h_flat):
        return _ratio_neg(h_flat, beta=50.0)
    
    result = minimize(objective, h, method='L-BFGS-B',
                      bounds=[(0, None)] * N,
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 20})
    
    h_opt = np.maximum(result.x, 0.0)
    score = _ratio(h_opt)
    return h_opt, score

def _find_sidon_set(size, max_iter=2000):
    """
    Find a set of 'size' indices in [0, N-1] such that pairwise sums are distinct.
    Greedy randomized search.
    """
    rng = np.random.RandomState(int(time.time() * 1000) % 100000)
    for _ in range(max_iter):
        current_set = []
        used_sums = set()
        
        # Try to build the set
        success = True
        for _ in range(size):
            found = False
            for _ in range(100):
                p = rng.randint(0, N)
                if p in current_set:
                    continue
                
                # Check if adding p causes sum collision
                valid = True
                new_sums = set()
                for q in current_set:
                    s = p + q
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.add(s)
                
                if not valid:
                    continue
                
                # Self-sum p+p
                s_self = 2 * p
                if s_self in used_sums:
                    valid = False
                    continue
                new_sums.add(s_self)
                
                if valid:
                    current_set.append(p)
                    used_sums.update(new_sums)
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success and len(current_set) == size:
            return current_set
            
    return None

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # 1. Try to find large Sidon sets and optimize from there
    # Theoretical max score for Sidon set of size k is ~ k/2.
    # For N=50, max k is around 7-8.
    for size in [8, 7, 6, 5]:
        if _time_left(start_time) < 10.0:
            break
            
        sidon_set = _find_sidon_set(size, max_iter=500)
        if sidon_set is None:
            continue
            
        # Create initial sparse vector
        h_sparse = np.zeros(N)
        for idx in sidon_set:
            h_sparse[idx] = 1.0
        
        # Normalize to L1=1 for consistent scaling in optimizer
        h_sparse = h_sparse / np.sum(h_sparse)
        
        s_init = _ratio(h_sparse)
        if s_init > best_score:
            best_score = s_init
            best_h = h_sparse.copy()
            
        # Refine using local optimization
        if _time_left(start_time) > 5.0:
            h_opt, s_opt = _local_optimize(h_sparse, start_time, max_iter=300, tol=1e-10)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
        # Refine using DFT parameterization (might find better phase alignments)
        if _time_left(start_time) > 5.0:
            z_init = _dft_inverse(h_sparse)
            h_dft_opt, s_dft = _optimize_from_z(z_init, start_time, max_iter=100, tol=1e-8)
            if s_dft > best_score:
                best_score = s_dft
                best_h = h_dft_opt.copy()
                
        # If we found a good Sidon set, we can stop searching for smaller ones
        # unless we have lots of time, because smaller sets have lower theoretical max
        if size >= 7 and best_score > 3.0: 
             break

    # 2. Fallback / Diversify with other initializations if time remains
    if _time_left(start_time) > 20.0:
        # Keep the diverse initializations from the original program as a backup
        initializations = []
        
        # Uniform
        initializations.append(np.ones(N) / N)
        
        # Single Gaussians
        for center in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            for width in [0.03, 0.05, 0.08]:
                x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
                h = np.exp(-((x - center) ** 2) / (2 * width ** 2))
                h = h / np.sum(h)
                initializations.append(h)
        
        # Random
        rng = np.random.RandomState(42)
        for _ in range(5):
            h_rand = rng.random(N)
            h_rand = h_rand / np.sum(h_rand)
            initializations.append(h_rand)
            
        scored_inits = []
        for h_init in initializations:
            s = _ratio(h_init)
            scored_inits.append((s, h_init))
        scored_inits.sort(key=lambda x: -x[0])
        
        top_k = min(5, len(scored_inits))
        for i in range(top_k):
            if _time_left(start_time) < 5.0:
                break
            s_init, h_init = scored_inits[i]
            if s_init < best_score * 0.9: # Only try if promising
                continue
                
            h_opt, s_opt = _local_optimize(h_init, start_time, max_iter=100, tol=1e-9)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Final polish
    if _time_left(start_time) > 5.0:
        h_polish, s_polish = _local_optimize(best_h, start_time, max_iter=500, tol=1e-12)
        if s_polish > best_score:
            best_score = s_polish
            best_h = h_polish.copy()

    return best_h.tolist()
