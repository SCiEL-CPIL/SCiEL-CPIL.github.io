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

def _neg_ratio(params):
    h = np.exp(params)
    return -_compute_ratio(h)

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

def construct():
    deadline = time.time() + 140.0
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Single spike initialization
    spike = np.zeros(N)
    spike[N // 2] = 1.0
    spike_score = _compute_ratio(spike)
    if spike_score > best_score:
        best_score = spike_score
        best_h = spike.copy()
    
    # Sidon-set based initialization
    # A Sidon set has the property that all pairwise sums are distinct,
    # which helps minimize autoconvolution peaks.
    sidon_set = _get_sidon_set(8, 25)
    h_sidon = np.zeros(N)
    center = N // 2
    for s in sidon_set:
        idx = center + s - sidon_set[len(sidon_set)//2]
        if 0 <= idx < N:
            h_sidon[idx] = 1.0
    if np.sum(h_sidon) > 0:
        sidon_score = _compute_ratio(h_sidon)
        if sidon_score > best_score:
            best_score = sidon_score
            best_h = h_sidon.copy()
    
    # Try various structured initializations
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Single spike
    spike = np.zeros(N)
    spike[N // 2] = 1.0
    inits.append(spike)
    
    # Two spikes
    for d in [1, 2, 3, 5, 7, 10, 12]:
        h2 = np.zeros(N)
        h2[N // 2 - d] = 1.0
        h2[N // 2 + d] = 1.0
        inits.append(h2)
    
    # Three spikes (Sidon-like)
    for d1, d2 in [(1, 3), (2, 5), (1, 4), (2, 7), (3, 8)]:
        h3 = np.zeros(N)
        h3[N // 2] = 1.0
        h3[N // 2 - d1] = 1.0
        h3[N // 2 + d2] = 1.0
        inits.append(h3)
    
    # Gaussian-like
    for sigma in [2, 3, 4, 5]:
        x = np.arange(N) - N // 2
        h_gauss = np.exp(-x**2 / (2 * sigma**2))
        inits.append(h_gauss)
    
    # Cosine window
    for width in [5, 8, 10]:
        x = np.arange(N) - N // 2
        mask = np.abs(x) <= width
        h_cos = np.zeros(N)
        h_cos[mask] = np.cos(np.pi * x[mask] / (2 * width + 1))
        inits.append(h_cos)
    
    # Sparse random
    for _ in range(5):
        h_sparse = np.zeros(N)
        indices = np.random.choice(N, size=np.random.randint(3, 8), replace=False)
        h_sparse[indices] = 1.0
        inits.append(h_sparse)
    
    # Evaluate all initializations
    for h_init in inits:
        if time.time() > deadline:
            break
        h_init = np.maximum(h_init, 1e-8)
        score = _compute_ratio(h_init)
        if score > best_score:
            best_score = score
            best_h = h_init.copy()
    
    # Local optimization from best initializations
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            h0 = best_h.copy()
        elif r == 1:
            h0 = inits[0].copy()  # Uniform
        elif r == 2:
            h0 = inits[1].copy()  # Single spike
        elif r == 3:
            h0 = inits[2].copy()  # Two spikes d=1
        elif r == 4:
            h0 = inits[3].copy()  # Two spikes d=2
        elif r == 5:
            h0 = inits[4].copy()  # Two spikes d=3
        elif r == 6:
            h0 = inits[5].copy()  # Two spikes d=5
        elif r == 7:
            h0 = inits[6].copy()  # Two spikes d=7
        elif r == 8:
            h0 = inits[7].copy()  # Three spikes
        elif r == 9:
            h0 = inits[8].copy()  # Three spikes
        elif r == 10:
            h0 = inits[9].copy()  # Gaussian
        elif r == 11:
            h0 = inits[10].copy()  # Gaussian
        elif r == 12:
            h0 = inits[11].copy()  # Cosine
        elif r == 13:
            h0 = inits[12].copy()  # Cosine
        elif r == 14:
            h0 = inits[13].copy()  # Sparse
        elif r == 15:
            h0 = inits[14].copy()  # Sparse
        else:
            h0 = best_h * (1.0 + 0.3 * np.random.randn(N))
        
        h0 = np.maximum(h0, 1e-8)
        log_h0 = np.log(h0)
        
        try:
            result = minimize(
                _neg_ratio,
                log_h0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = np.exp(result.x)
            score = _compute_ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
        except Exception:
            pass
    
    # Final refinement with more iterations from best
    if time.time() < deadline - 10:
        h0 = np.maximum(best_h, 1e-8)
        log_h0 = np.log(h0)
        try:
            result = minimize(
                _neg_ratio,
                log_h0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_opt = np.exp(result.x)
            score = _compute_ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
        except Exception:
            pass
    
    return [float(x) for x in best_h]
