import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    """Compute discrete autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')

def _ratio(h, beta=100.0):
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

def _ratio_neg(h, beta=100.0):
    return -_ratio(h, beta)

def _compute_ratio_exact(h):
    """Compute exact ratio without soft-max."""
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
    return l2_sq / max_val

def _generate_sidon_sets(K, N, rng):
    """Generate Sidon sets of size K from range [0, N).
    A Sidon set has all pairwise differences distinct."""
    sidon_sets = []
    for _ in range(50):
        # Greedy construction of Sidon set
        S = set()
        diffs = set()
        valid = True
        for _ in range(K):
            candidates = [i for i in range(N) if i not in S]
            if not candidates:
                valid = False
                break
            # Try to find an element that doesn't create duplicate differences
            found = False
            for i in candidates:
                new_diffs = set()
                conflict = False
                for j in S:
                    d = abs(i - j)
                    if d in diffs or d in new_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                if not conflict:
                    S.add(i)
                    diffs.update(new_diffs)
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid and len(S) == K:
            sidon_set = sorted(S)
            if sidon_set not in sidon_sets:
                sidon_sets.append(sidon_set)
        if len(sidon_sets) >= 20:
            break
    # If we didn't get enough, add some random sparse sets
    while len(sidon_sets) < 5:
        S = sorted(rng.choice(N, size=K, replace=False).tolist())
        if S not in sidon_sets:
            sidon_sets.append(S)
    return sidon_sets

def _sidon_heights(exponents, amplitudes, N):
    """Compute heights as squared magnitudes of DFT of sparse polynomial.
    
    P(z) = sum_k a_k * z^{e_k}
    h[n] = |sum_k a_k * exp(2*pi*i*n*e_k/N)|^2 / N
    """
    h = np.zeros(N)
    for n in range(N):
        val = 0.0
        for k, e in enumerate(exponents):
            val += amplitudes[k] * np.exp(2j * np.pi * n * e / N)
        h[n] = (np.abs(val) ** 2) / N
    return h

def _sidon_ratio(params, beta=100.0):
    """Compute ratio for Sidon-parameterized heights.
    params: [log_amplitudes..., exponents...] where exponents are floats we round"""
    K = len(params) // 2
    log_amps = params[:K]
    exps_float = params[K:]
    exps = np.round(exps_float).astype(int)
    amps = np.exp(log_amps)
    h = _sidon_heights(exps, amps, N)
    return _ratio(h, beta)

def _sidon_ratio_neg(params, beta=100.0):
    return -_sidon_ratio(params, beta)

def _optimize_sidon(exponents, K, start_time, max_iter=200, beta=100.0):
    """Optimize amplitudes for a fixed Sidon set of exponents."""
    # Initialize amplitudes
    rng = np.random.RandomState(hash(tuple(exponents)) % 2**31)
    log_amps0 = rng.randn(K) * 0.5
    params0 = np.concatenate([log_amps0, np.array(exponents, dtype=float)])
    
    bounds = [(-5, 5)] * K + [(0, N - 1)] * K
    
    def objective(params):
        return _sidon_ratio_neg(params, beta)
    
    result = minimize(objective, params0, method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': max_iter, 'gtol': 1e-10, 'maxls': 20})
    
    params_opt = result.x
    exps_opt = np.round(params_opt[K:]).astype(int)
    amps_opt = np.exp(params_opt[:K])
    h_opt = _sidon_heights(exps_opt, amps_opt, N)
    score = _compute_ratio_exact(h_opt)
    return h_opt, score

def _local_optimize(h0, start_time, max_iter=500, tol=1e-10, beta=100.0):
    """Direct optimization of h with non-negativity constraint."""
    h = np.maximum(h0, 0.0).copy()
    
    def objective(h_flat):
        return _ratio_neg(h_flat, beta)
    
    result = minimize(objective, h, method='L-BFGS-B',
                      bounds=[(0, None)] * N,
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 30})
    
    h_opt = np.maximum(result.x, 0.0)
    return h_opt, -result.fun

def _gaussian_profile(center, width, height):
    """Create a Gaussian-like profile on the 50 bins."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h = height * np.exp(-((x - center) ** 2) / (2 * width ** 2))
    return h

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _compute_ratio_exact(best_h)
    
    # Phase 1: Sidon-set sparse polynomial optimization
    rng = np.random.RandomState(42)
    
    for K in [3, 4, 5, 6]:
        if _time_left(start_time) < 15.0:
            break
        
        sidon_sets = _generate_sidon_sets(K, N, rng)
        
        for sidon_set in sidon_sets[:10]:
            if _time_left(start_time) < 5.0:
                break
            
            h_opt, s_opt = _optimize_sidon(sidon_set, K, start_time, max_iter=150, beta=100.0)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
    
    # Phase 2: Diverse initializations and local optimization
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Single Gaussians at various positions and widths
    for center in [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]:
        for width in [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]:
            initializations.append(_gaussian_profile(center, width, 1.0))
    
    # Random initializations
    for _ in range(15):
        h_rand = rng.random(N)
        initializations.append(h_rand)
    
    # Sparse random
    for _ in range(10):
        h_sparse = np.zeros(N)
        idx = rng.choice(N, size=rng.randint(5, 15), replace=False)
        h_sparse[idx] = rng.random(len(idx))
        initializations.append(h_sparse)
    
    # Sort initializations by score, keep top ones for optimization
    scored_inits = []
    for h_init in initializations:
        h_init = np.maximum(h_init, 0.0)
        s = _compute_ratio_exact(h_init)
        scored_inits.append((s, h_init))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # Update best from initializations
    if scored_inits[0][0] > best_score:
        best_score = scored_inits[0][0]
        best_h = scored_inits[0][1].copy()
    
    # Optimize from top initializations
    top_k = min(15, len(scored_inits))
    for i in range(top_k):
        if _time_left(start_time) < 5.0:
            break
        
        s_init, h_init = scored_inits[i]
        
        # Direct optimization with high beta
        h_opt, s_opt = _local_optimize(h_init, start_time, max_iter=300, tol=1e-10, beta=100.0)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    # Phase 3: Final polish with high-precision optimization
    if _time_left(start_time) > 10.0:
        for beta in [100.0, 200.0, 500.0]:
            if _time_left(start_time) < 5.0:
                break
            h_polish, s_polish = _local_optimize(best_h, start_time, max_iter=500, tol=1e-12, beta=beta)
            if s_polish > best_score:
                best_score = s_polish
                best_h = h_polish.copy()
    
    # One more polish with very high beta for near-exact max
    if _time_left(start_time) > 5.0:
        def objective_final(h_flat):
            return -_ratio(h_flat, beta=1000.0)
        
        result = minimize(objective_final, best_h, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 500, 'gtol': 1e-12, 'maxls': 30})
        h_final = np.maximum(result.x, 0.0)
        s_final = _compute_ratio_exact(h_final)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final.copy()
    
    # Final exact evaluation
    h_final = np.maximum(best_h, 0.0)
    l1 = np.sum(h_final)
    if l1 > 0:
        h_norm = h_final / l1
        l2_sq = np.sum(h_norm ** 2)
        conv = _autoconv(h_norm)
        max_val = np.max(conv)
        if max_val > 0:
            exact_score = l2_sq / max_val
            if exact_score > best_score:
                best_score = exact_score
    
    return best_h.tolist()
