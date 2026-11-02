import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution of step function with heights h (vectorized)."""
    n = len(h)
    num_points = 2 * n - 1
    
    # Precompute overlap matrix: overlap[i, j, k] = length of intersection
    # of [i, i+1] and [k-j, k-j+1] for i,j in [0,n-1], k in [0,2n-2]
    # overlap = max(0, min(i+1, k-j+1) - max(i, k-j))
    
    i_idx = np.arange(n)
    j_idx = np.arange(n)
    k_idx = np.arange(num_points)
    
    # Build 3D overlap array
    I, J, K = np.meshgrid(i_idx, j_idx, k_idx, indexing='ij')
    lo = np.maximum(I, K - J)
    hi = np.minimum(I + 1, K - J + 1)
    overlaps = np.maximum(0, hi - lo)  # shape (n, n, 2n-1)
    
    # ff_vals[k] = DELTA * sum_{i,j} h[i]*h[j]*overlaps[i,j,k]
    # = DELTA * einsum('i,j,ijk->k', h, h, overlaps)
    ff_vals = DELTA * np.einsum('i,j,ijk->k', h, h, overlaps)
    
    return np.max(ff_vals)

def _autoconv_all(h):
    """Compute all autoconv values."""
    n = len(h)
    num_points = 2 * n - 1
    i_idx = np.arange(n)
    j_idx = np.arange(n)
    k_idx = np.arange(num_points)
    I, J, K = np.meshgrid(i_idx, j_idx, k_idx, indexing='ij')
    lo = np.maximum(I, K - J)
    hi = np.minimum(I + 1, K - J + 1)
    overlaps = np.maximum(0, hi - lo)
    ff_vals = DELTA * np.einsum('i,j,ijk->k', h, h, overlaps)
    return ff_vals

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _smooth_autoconv_sup(h, beta=100.0):
    """Compute smooth approximation of sup norm using log-sum-exp."""
    ff_vals = _autoconv_all(h)
    max_val = np.max(ff_vals)
    if max_val == 0:
        return 0.0
    return max_val + np.log(np.sum(np.exp((ff_vals - max_val) * beta))) / beta

def _smooth_ratio(h, beta=100.0):
    """Compute smooth ratio."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _smooth_autoconv_sup(h, beta)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def generate_mian_chowla(n, max_size):
    """Generate a Mian-Chowla (Sidon) set."""
    sidon = [0]
    used_sums = {0}
    for candidate in range(1, n):
        valid = True
        for i in sidon:
            if i + candidate in used_sums:
                valid = False
                break
        if valid:
            sidon.append(candidate)
            used_sums.add(candidate + candidate)
            for i in sidon[:-1]:
                used_sums.add(i + candidate)
        if len(sidon) >= max_size:
            break
    return sidon

def _epigraph_optimize(h0, start_time, deadline):
    """
    Epigraph formulation: maximize sum_h2/t subject to ff_vals <= t, h >= 0, sum_h = 1.
    Using smooth constraints via log-sum-exp for the max.
    """
    h0 = np.maximum(h0, 0).copy()
    s0 = np.sum(h0)
    if s0 < 1e-15:
        return h0, _ratio(h0)
    h0 = h0 / s0
    
    best_h = h0.copy()
    best_score = _ratio(h0)
    
    # Epigraph: minimize -log(sum_h2) + log(t) + penalty*logsumexp(ff_vals/t)
    # Or: maximize sum_h2/t - mu * max(0, max(ff_vals) - t)
    # Use smooth version: maximize sum_h2/t - mu * logsumexp((ff_vals - t)/beta)
    
    beta = 50.0
    
    def neg_objective(x):
        h_part = np.maximum(x[:N], 0)
        t_val = max(x[N], 1e-10)
        s = np.sum(h_part)
        if s < 1e-15:
            return 0.0
        h_norm = h_part / s
        
        ff_vals = _autoconv_all(h_norm)
        sum_h2 = np.sum(h_norm**2)
        
        # Smooth constraint violation: logsumexp((ff_vals - t_val)/beta) * beta
        shifted = (ff_vals - t_val) / beta
        max_shifted = np.max(shifted)
        if max_shifted > 50:  # overflow guard
            return 1e10
        lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        penalty = lse * beta
        
        ratio_val = sum_h2 / t_val
        return -(ratio_val - penalty)
    
    t_init = _autoconv_sup(h0) * 1.05
    x0 = np.concatenate([h0, [t_init]])
    bounds = [(0, None)] * N + [(1e-8, None)]
    
    try:
        result = minimize(neg_objective, x0, method='L-BFGS-B',
                         bounds=bounds,
                         options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-9})
        
        h_candidate = np.maximum(result.x[:N], 0)
        s = np.sum(h_candidate)
        if s > 1e-15:
            h_candidate = h_candidate / s
            sc = _ratio(h_candidate)
            if sc > best_score:
                best_score = sc
                best_h = h_candidate.copy()
    except Exception:
        pass
    
    return best_h, best_score

def _coordinate_descent_polish(h, start_time, deadline, max_rounds=50):
    """Coordinate descent with golden section search per coordinate."""
    best_h = h.copy()
    best_score = _ratio(best_h)
    
    for round_num in range(max_rounds):
        if time.time() - start_time > deadline - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > deadline - 5:
                break
            
            old_val = best_h[i]
            
            # Try to optimize coordinate i via a few trial values
            # Use local search around current value
            candidates = []
            if old_val > 0:
                for frac in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
                    candidates.append(old_val * frac)
            candidates.extend([0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
            
            best_local = old_val
            best_local_score = best_score
            
            for cv in candidates:
                cv = max(0.0, cv)
                test_h = best_h.copy()
                test_h[i] = cv
                s = np.sum(test_h)
                if s < 1e-15:
                    continue
                test_h = test_h / s
                sc = _ratio(test_h)
                if sc > best_local_score + 1e-12:
                    best_local_score = sc
                    best_local = cv
            
            if best_local != old_val:
                best_h[i] = best_local
                s = np.sum(best_h)
                if s > 1e-15:
                    best_h = best_h / s
                    best_score = best_local_score
                improved = True
        
        if not improved:
            break
    
    return best_h, best_score

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    np.random.seed(42)
    
    initializations = []
    
    # 1. Mian-Chowla sets of various sizes
    for size in [8, 10, 12, 14, 16]:
        s = generate_mian_chowla(N, size)
        h_mc = np.zeros(N)
        for idx in s:
            h_mc[idx] = 1.0
        initializations.append(h_mc)
        
    # 2. Random sparse vectors
    for _ in range(20):
        h_sp = np.zeros(N)
        num_ones = np.random.randint(5, 15)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sp[indices] = np.random.uniform(0.5, 1.5, num_ones)
        initializations.append(h_sp)
        
    # 3. Random dense vectors
    for _ in range(10):
        h_rd = np.random.uniform(0, 1, N)
        initializations.append(h_rd)
        
    # 4. Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    initializations.append(h_alt)
    
    # 5. Every 3
    h_sp3 = np.zeros(N)
    h_sp3[::3] = 1.0
    initializations.append(h_sp3)
    
    # 6. Gaussian
    t = np.linspace(-1, 1, N)
    initializations.append(np.exp(-t**2/0.3))
    
    # 7. Triangular
    h_tri = np.zeros(N)
    center = N // 2
    for i in range(N):
        h_tri[i] = max(0, 1 - abs(i - center) / (N/2))
    initializations.append(h_tri)
    
    # 8. Two peaks
    h_2p = np.zeros(N)
    h_2p[10] = 1.0
    h_2p[40] = 1.0
    initializations.append(h_2p)
    
    # 9. Cosine bump
    t = np.linspace(-1, 1, N)
    initializations.append(0.5 + 0.5*np.cos(np.pi*t))
    
    # 10. Power law
    t = np.linspace(-1, 1, N)
    initializations.append(np.abs(t)**(-0.5 + 1e-9))
    
    for h0 in initializations:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            continue
        h0 = h0 / s0
        
        # Phase 1: Smooth surrogate optimization with multiple beta values
        for beta in [30.0, 50.0, 80.0]:
            if time.time() - start_time > DEADLINE - 20:
                break
            
            def neg_smooth_ratio(x, b=beta):
                x = np.maximum(x, 0)
                s = np.sum(x)
                if s < 1e-15:
                    return 0.0
                return -_smooth_ratio(x / s, beta=b)
            
            try:
                result = minimize(neg_smooth_ratio, h0.copy(), method='L-BFGS-B',
                                 bounds=[(0, None)] * N,
                                 options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9})
                
                h_candidate = np.maximum(result.x, 0)
                s = np.sum(h_candidate)
                if s > 1e-15:
                    h_candidate = h_candidate / s
                    sc = _ratio(h_candidate)
                    if sc > best_score:
                        best_score = sc
                        best_h = h_candidate.copy()
            except Exception:
                continue
        
        # Phase 2: Epigraph optimization
        if time.time() - start_time < DEADLINE - 15:
            try:
                h_epi, sc_epi = _epigraph_optimize(best_h.copy(), start_time, DEADLINE)
                if sc_epi > best_score:
                    best_score = sc_epi
                    best_h = h_epi.copy()
            except Exception:
                pass
    
    # Final polish: Nelder-Mead on exact ratio
    if time.time() - start_time < DEADLINE - 10:
        try:
            def neg_ratio(x):
                x = np.maximum(x, 0)
                s = np.sum(x)
                if s < 1e-15:
                    return 0.0
                return -_ratio(x / s)
            
            result = minimize(neg_ratio, best_h.copy(), method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            h_candidate = np.maximum(result.x, 0)
            s = np.sum(h_candidate)
            if s > 1e-15:
                h_candidate = h_candidate / s
                sc = _ratio(h_candidate)
                if sc > best_score:
                    best_score = sc
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Coordinate descent polish with more diverse perturbations
    if time.time() - start_time < DEADLINE - 5:
        try:
            h_cd, sc_cd = _coordinate_descent_polish(best_h.copy(), start_time, DEADLINE)
            if sc_cd > best_score:
                best_score = sc_cd
                best_h = h_cd.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead polish after coordinate descent
    if time.time() - start_time < DEADLINE - 3:
        try:
            def neg_ratio2(x):
                x = np.maximum(x, 0)
                s = np.sum(x)
                if s < 1e-15:
                    return 0.0
                return -_ratio(x / s)
            
            result = minimize(neg_ratio2, best_h.copy(), method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            
            h_candidate = np.maximum(result.x, 0)
            s = np.sum(h_candidate)
            if s > 1e-15:
                h_candidate = h_candidate / s
                sc = _ratio(h_candidate)
                if sc > best_score:
                    best_score = sc
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h.tolist()
