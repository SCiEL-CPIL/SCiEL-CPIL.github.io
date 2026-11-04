import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_ratio(h):
    """Compute ||f||_2^2 / (||f||_1 * ||f*f||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = float(np.sum(h * h))
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(F * F.conj(), n=n_fft)
    conv_inf = float(np.max(np.abs(conv)))
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _get_sidon_set(n, max_val):
    """Generate a Sidon set of size n within [0, max_val]."""
    S = [0]
    sums = set([0])
    for i in range(1, max_val + 1):
        if len(S) >= n:
            break
        new_sums = set()
        valid = True
        for s in S:
            new_sum = i + s
            if new_sum in sums or new_sum in new_sums:
                valid = False
                break
            new_sums.add(new_sum)
        two_i = 2 * i
        if two_i in sums or two_i in new_sums:
            valid = False
        if valid:
            S.append(i)
            sums.update(new_sums)
            sums.add(two_i)
    return S

def _optimize_amplitudes_simplex(indices, deadline, maxiter=300):
    """Optimize amplitudes on a fixed support using simplex parameterization."""
    k = len(indices)
    if k == 0:
        return np.zeros(N), 0.0
    
    def simplex_obj(y):
        # Softmax reparameterization: x_i = exp(y_i) / sum(exp(y_j))
        y_max = np.max(y)
        exp_y = np.exp(y - y_max)
        x = exp_y / np.sum(exp_y)
        h = np.zeros(N)
        h[indices] = x
        return -_compute_ratio(h)
    
    best_score = 0.0
    best_x = np.ones(k) / k
    
    rng = np.random.RandomState(42 + len(indices))
    for attempt in range(8):
        if time.time() > deadline:
            break
        if attempt == 0:
            y_init = np.zeros(k)
        else:
            y_init = rng.randn(k) * 0.8
        
        try:
            res = minimize(simplex_obj, y_init, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < -best_score:
                best_score = -res.fun
                y_opt = res.x
                y_max = np.max(y_opt)
                exp_y = np.exp(y_opt - y_max)
                best_x = exp_y / np.sum(exp_y)
        except Exception:
            pass
    
    h_final = np.zeros(N)
    h_final[indices] = best_x
    s = np.sum(h_final)
    if s > 0:
        h_final = h_final / s
    return h_final, best_score

def _optimize_amplitudes_dft(indices, deadline, maxiter=200):
    """Optimize using DFT spectral parameterization: h_k = |z_k|^2 where z is complex.
    We optimize the phases and magnitudes of z on the support indices."""
    k = len(indices)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Parameterize: for each index in support, z_i = r_i * exp(i*theta_i)
    # h_i = r_i^2
    # We optimize log(r_i) and theta_i jointly
    # But simpler: just optimize the complex z directly, then h = |z|^2 normalized
    
    def dft_obj(params):
        # params: [log_r_0, theta_0, log_r_1, theta_1, ...]
        log_rs = params[0::2]
        thetas = params[1::2]
        rs = np.exp(log_rs)
        # h_i = r_i^2, then normalize to L1=1
        h_vals = rs * rs
        s = np.sum(h_vals)
        if s < 1e-15:
            return 0.0
        h_vals = h_vals / s
        h = np.zeros(N)
        h[indices] = h_vals
        return -_compute_ratio(h)
    
    best_score = 0.0
    best_params = None
    
    rng = np.random.RandomState(123 + len(indices))
    for attempt in range(6):
        if time.time() > deadline:
            break
        if attempt == 0:
            # Uniform magnitudes, random phases
            log_r0 = np.zeros(k)
            theta0 = rng.uniform(0, 2*np.pi, k)
        else:
            log_r0 = rng.randn(k) * 0.5
            theta0 = rng.uniform(0, 2*np.pi, k)
        
        params0 = np.zeros(2*k)
        params0[0::2] = log_r0
        params0[1::2] = theta0
        
        try:
            res = minimize(dft_obj, params0, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to uniform
        h_final = np.zeros(N)
        h_final[indices] = 1.0 / k
        return h_final, 0.0
    
    log_rs = best_params[0::2]
    thetas = best_params[1::2]
    rs = np.exp(log_rs)
    h_vals = rs * rs
    s = np.sum(h_vals)
    if s > 0:
        h_vals = h_vals / s
    h_final = np.zeros(N)
    h_final[indices] = h_vals
    return h_final, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Generate candidate Sidon sets of various sizes
    candidate_sets = []
    for k in range(4, 14):
        indices = _get_sidon_set(k, N - 1)
        if len(indices) >= 4:
            candidate_sets.append(indices)
    
    # Also add a few random sparse sets for diversity
    rng = np.random.RandomState(999)
    for _ in range(3):
        size = rng.randint(5, 10)
        indices = sorted(rng.choice(N, size=size, replace=False).tolist())
        candidate_sets.append(indices)
    
    # For each candidate set, try both optimization methods
    for indices in candidate_sets:
        if time.time() > deadline:
            break
        
        # Method 1: Simplex (softmax) optimization
        h1, score1 = _optimize_amplitudes_simplex(indices, deadline)
        if score1 > best_score:
            best_score = score1
            best_h = h1.copy()
        
        # Method 2: DFT spectral optimization
        if time.time() > deadline:
            break
        h2, score2 = _optimize_amplitudes_dft(indices, deadline)
        if score2 > best_score:
            best_score = score2
            best_h = h2.copy()
    
    # Final polish: if we have time, do a local refinement on the best solution
    if time.time() < deadline - 10.0:
        # Extract support from best_h
        support = np.where(best_h > 1e-10)[0]
        if len(support) >= 2:
            h_polished, score_polished = _optimize_amplitudes_simplex(support, deadline, maxiter=500)
            if score_polished > best_score:
                best_score = score_polished
                best_h = h_polished.copy()
    
    # Ensure non-negativity and return as list of floats
    best_h = np.maximum(best_h, 0.0)
    return [float(x) for x in best_h]
