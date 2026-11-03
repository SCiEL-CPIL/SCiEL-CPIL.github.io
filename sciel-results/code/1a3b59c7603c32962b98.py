"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses SDP relaxation with barrier method and structured optimization.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 115


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """
    Project h onto {x | sum(x) = target_sum, 0 <= x_i <= 1}.
    """
    h = np.asarray(h, dtype=np.float64)
    
    def sum_proj(tau):
        return np.sum(np.clip(h + tau, 0.0, 1.0))
    
    lo = -max(np.max(h), 1.0) - 1.0
    hi = 1.0 - min(np.min(h), 0.0) + 1.0
    
    if sum_proj(lo) > target_sum:
        lo -= 10.0
    if sum_proj(hi) < target_sum:
        hi += 10.0
        
    for _ in range(50):
        mid = (lo + hi) / 2.0
        s = sum_proj(mid)
        if s < target_sum:
            lo = mid
        else:
            hi = mid
            
    tau = (lo + hi) / 2.0
    h_proj = np.clip(h + tau, 0.0, 1.0)
    
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-12:
        mask = (h_proj > 1e-9) & (h_proj < 1.0 - 1e-9)
        if np.any(mask):
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
            
    return h_proj


def _sdp_relaxation(n_small, max_iter=100, mu_init=0.1):
    """
    Solve SDP relaxation for smaller n using barrier method.
    Minimize t subject to:
      G >= 0 (PSD)
      (2/n) * (n/2 - sum_i G[i, (i+k)%n]) <= t for all k
      sum_i G[i,i] <= n/2 (since h_i^2 <= h_i and sum h_i = n/2)
      G[i,i] >= 0
    """
    G = np.eye(n_small) * 0.25
    t = 0.6
    mu = mu_init
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 25:
            break
            
        # Compute constraint violations
        cs = np.zeros(n_small)
        for k in range(n_small):
            s = 0.0
            for i in range(n_small):
                j = (i + k) % n_small
                s += G[i, j]
            cs[k] = (2.0 / n_small) * (n_small / 2.0 - s)
        
        # Barrier term for PSD: -mu * sum(log(eigenvalues))
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals_clipped = np.clip(eigvals, 1e-10, None)
        
        # Gradient of barrier
        grad_barrier = np.zeros((n_small, n_small))
        # d/dG of -mu * sum log(eig) = -mu * eigvecs @ diag(1/eig) @ eigvecs.T
        inv_eig = 1.0 / eigvals_clipped
        grad_barrier = -mu * (eigvecs * inv_eig) @ eigvecs.T
        
        # Gradient of objective: minimize t + barrier
        # dL/dt = 1
        # dL/dG from overlap constraints: use smooth max
        beta = 30.0
        m = np.max(cs)
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
        
        grad_G = np.zeros((n_small, n_small))
        for k in range(n_small):
            if w[k] < 1e-10:
                continue
            factor = -(2.0 / n_small) * w[k]
            for i in range(n_small):
                j = (i + k) % n_small
                grad_G[i, j] += factor
                grad_G[j, i] += factor
        
        grad_G = (grad_G + grad_G.T) / 2.0
        grad_G += grad_barrier
        
        # Newton step: solve for delta G and delta t
        # Simplified: use gradient descent with adaptive step
        grad_norm = np.linalg.norm(grad_G, 'fro')
        if grad_norm < 1e-8:
            break
            
        step_size = 0.01 / (1.0 + grad_norm * 0.1)
        
        G_new = G - step_size * grad_G
        G_new = (G_new + G_new.T) / 2.0
        
        # Project to PSD
        eigvals, eigvecs = np.linalg.eigh(G_new)
        eigvals = np.clip(eigvals, 1e-10, None)
        G_new = (eigvecs * eigvals) @ eigvecs.T
        
        # Update t
        t = max(0.0, t - step_size * 0.5)
        
        G = G_new
        mu *= 0.8
        
    # Extract h from G
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argmax(eigvals)
    h_small = np.abs(eigvecs[:, idx]) * np.sqrt(max(eigvals[idx], 0))
    
    # Normalize
    if np.sum(h_small) > 1e-6:
        h_small = h_small * (n_small / 2.0) / np.sum(h_small)
    h_small = np.clip(h_small, 0, 1)
    h_small = _project(h_small, n_small / 2.0)
    
    return h_small


def _optimize_structured(h0, max_iter=500):
    """
    Optimize using projected gradient with line search, exploiting
    the circulant structure of the overlap constraints.
    """
    h = _project(h0.copy(), N / 2.0)
    h_prev = h.copy()
    g_prev = None
    alpha = 0.05
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
            
        cs = _linear_overlap_values(h)
        m = np.max(cs)
        
        # Adaptive beta for smooth max
        beta = 50.0 + 50.0 * i / max_iter
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
        
        # Compute gradient efficiently
        grad = np.zeros(N)
        for k in range(N):
            if w[k] < 1e-10:
                continue
            factor = -(2.0 / N) * w[k]
            if k > 0:
                grad[k:] += factor * h[:N-k]
            if k < N:
                grad[:N-k] += factor * h[k:]
        
        # Line search
        best_alpha = alpha
        best_score = _score(h)
        for ls in range(5):
            h_try = h - best_alpha * grad
            h_try = _project(h_try, N / 2.0)
            s_try = _score(h_try)
            if s_try < best_score:
                best_score = s_try
            else:
                best_alpha *= 0.5
                break
        
        h_new = h - best_alpha * grad
        h_new = _project(h_new, N / 2.0)
        
        # Barzilai-Borwein step update
        if g_prev is not None:
            s_vec = h_new - h
            y_vec = grad - g_prev
            s_norm_sq = np.dot(s_vec, s_vec)
            y_norm_sq = np.dot(y_vec, y_vec)
            sy = np.dot(s_vec, y_vec)
            if s_norm_sq > 1e-12 and y_norm_sq > 1e-12 and sy > 1e-12:
                alpha_bb = s_norm_sq / sy
                alpha = 0.5 * (alpha + alpha_bb)
                alpha = np.clip(alpha, 1e-4, 0.5)
        
        h_prev = h
        h = h_new
        g_prev = grad
        
        if np.linalg.norm(h - h_prev) < 1e-9:
            break
            
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    target_sum = N / 2.0

    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h, target_sum)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # 1. SDP relaxation for small n, then interpolate
    if time.time() - t0 < _DEADLINE - 40:
        for n_small in [10, 20]:
            if time.time() - t0 > _DEADLINE - 30:
                break
            h_sdp = _sdp_relaxation(n_small)
            # Interpolate to N
            h_fft = np.fft.rfft(h_sdp)
            n_fft = 2 * N
            h_interp = np.fft.irfft(h_fft, n=n_fft)[:N]
            h_interp = _project(h_interp, target_sum)
            _update_best(h_interp)
            h_opt = _optimize_structured(h_interp)
            _update_best(h_opt)

    # 2. Structured frequency-based starts
    i = np.arange(N)
    if time.time() - t0 < _DEADLINE - 20:
        for freq in [1, 2, 3, 5]:
            if time.time() - t0 > _DEADLINE - 15:
                break
            # Multiple amplitudes
            for amp in [0.3, 0.5, 0.7]:
                h_cos = 0.5 + amp * np.cos(2.0 * np.pi * freq * i / N)
                h_cos = _project(h_cos, target_sum)
                h_opt = _optimize_structured(h_cos)
                _update_best(h_opt)

    # 3. Random starts with refinement
    if time.time() - t0 < _DEADLINE - 15:
        for _ in range(4):
            if time.time() - t0 > _DEADLINE - 10:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _project(h_rand, target_sum)
            h_opt = _optimize_structured(h_rand)
            _update_best(h_opt)

    # 4. Final polish on best solution
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_polished = _optimize_structured(best_h, max_iter=200)
        _update_best(h_polished)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project(best_h, target_sum)
    return best_h
