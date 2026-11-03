"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses SDP relaxation with Gram matrix and interior-point-inspired solver.
"""
import time
import numpy as np

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
        
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = sum_proj(mid)
        if s < target_sum:
            lo = mid
        else:
            hi = mid
            
    tau = (lo + hi) / 2.0
    h_proj = np.clip(h + tau, 0.0, 1.0)
    
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        mask = (h_proj > 1e-10) & (h_proj < 1.0 - 1e-10)
        if np.any(mask):
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
            
    return h_proj


def _solve_sdp_gram(n_small, max_iter=500):
    """
    Solve SDP relaxation using Gram matrix G with PSD constraint.
    G = h h^T, minimize t subject to:
      G >= 0 (PSD)
      For each k: (2/n) * (n/2 - sum_i G[i, (i+k)%n]) <= t
      trace(G) = sum h_i^2 <= sum h_i = n/2
    """
    G = np.eye(n_small) * 0.25
    t = 1.0
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 25:
            break
            
        # Compute overlap values from G
        cs = np.zeros(n_small)
        for k in range(n_small):
            s = 0.0
            for i in range(n_small):
                j = (i + k) % n_small
                s += G[i, j]
            cs[k] = (2.0 / n_small) * (n_small / 2.0 - s)
        
        max_c = np.max(cs)
        if max_c <= t + 1e-8:
            break
            
        # Smooth max approximation
        beta = 30.0 + 20.0 * (iteration / max_iter)
        m = np.max(cs)
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
        
        # Gradient with respect to G
        dG = np.zeros((n_small, n_small))
        for k in range(n_small):
            if w[k] < 1e-10:
                continue
            factor = -(2.0 / n_small) * w[k]
            for i in range(n_small):
                j = (i + k) % n_small
                dG[i, j] += factor
                dG[j, i] += factor
        
        dG = (dG + dG.T) / 2.0
        
        # Line search for step size
        alpha = 0.05
        for _ in range(10):
            G_trial = G - alpha * dG
            G_trial = (G_trial + G_trial.T) / 2.0
            
            # Project to PSD
            eigvals, eigvecs = np.linalg.eigh(G_trial)
            eigvals = np.clip(eigvals, 0, None)
            G_trial = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Check if this improves the objective
            cs_trial = np.zeros(n_small)
            for k in range(n_small):
                s = 0.0
                for i in range(n_small):
                    j = (i + k) % n_small
                    s += G_trial[i, j]
                cs_trial[k] = (2.0 / n_small) * (n_small / 2.0 - s)
            
            if np.max(cs_trial) < max_c * 0.999 or alpha < 0.001:
                break
            alpha *= 0.5
        
        G = G - alpha * dG
        G = (G + G.T) / 2.0
        
        # Project to PSD
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = np.clip(eigvals, 0, None)
        G = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        t = max(t, val)
        
        if np.linalg.norm(dG) < 1e-10:
            break
    
    # Extract h from G using largest eigenvector
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argmax(eigvals)
    h_small = np.abs(eigvecs[:, idx]) * np.sqrt(max(eigvals[idx], 0))
    
    # Normalize
    if np.sum(h_small) > 1e-6:
        h_small = h_small * (n_small / 2.0) / np.sum(h_small)
    h_small = np.clip(h_small, 0, 1)
    h_small = _project(h_small, n_small / 2.0)
    
    return h_small


def _interpolate_to_n(h_small, n_target):
    """Interpolate h_small to n_target using FFT."""
    n_small_len = len(h_small)
    if n_small_len == n_target:
        return h_small.copy()
    
    # Zero-pad and interpolate
    h_fft = np.fft.rfft(h_small, n=2 * n_target)
    h_interp = np.fft.irfft(h_fft, n=2 * n_target)[:n_target]
    h_interp = _project(h_interp, n_target / 2.0)
    
    return h_interp


def _optimize(h0, max_iter=600, beta_start=5.0, beta_end=300.0):
    """
    Optimize using smooth max with adaptive step sizes.
    """
    h = _project(h0.copy(), N / 2.0)
    h_prev = h.copy()
    g_prev = None
    best_local = _score(h)
    best_h_local = h.copy()
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 12:
            break
            
        beta = beta_start * (beta_end / beta_start) ** (i / max_iter)
        
        cs = _linear_overlap_values(h)
        m = np.max(cs)
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
        
        # Compute gradient
        grad = np.zeros(N)
        for k in range(N):
            if w[k] < 1e-10:
                continue
            factor = -(2.0 / N) * w[k]
            if k > 0:
                grad[k:] += factor * h[:N-k]
            if k < N:
                grad[:N-k] += factor * h[k:]
        
        # Adaptive step size
        if g_prev is not None:
            s_vec = h - h_prev
            y_vec = grad - g_prev
            s_norm_sq = np.dot(s_vec, s_vec)
            y_norm_sq = np.dot(y_vec, y_vec)
            sy = np.dot(s_vec, y_vec)
            
            if s_norm_sq > 1e-12 and y_norm_sq > 1e-12 and sy > 1e-12:
                alpha_bb1 = s_norm_sq / sy
                alpha_bb2 = sy / y_norm_sq
                alpha = 0.5 * (alpha_bb1 + alpha_bb2)
                alpha = np.clip(alpha, 1e-6, 5.0)
            else:
                alpha = 0.1 / (np.linalg.norm(grad) + 1e-6)
        else:
            alpha = 0.1 / (np.linalg.norm(grad) + 1e-6)

        # Line search
        for _ in range(8):
            h_new = h - alpha * grad
            h_new = _project(h_new, N / 2.0)
            
            score_new = _score(h_new)
            if score_new < best_local * 1.001:
                break
            alpha *= 0.5
        
        h_new = h - alpha * grad
        h_new = _project(h_new, N / 2.0)
        score_new = _score(h_new)
        
        if score_new < best_local:
            best_local = score_new
            best_h_local = h_new.copy()
        
        h_prev = h
        h = h_new
        g_prev = grad
        
        if np.linalg.norm(h - h_prev) < 1e-9:
            break
            
    return best_h_local


def _polish(h0, max_iter=200):
    """
    Fine-tuning step with very small perturbations.
    """
    h = _project(h0.copy(), N / 2.0)
    best = _score(h)
    best_h = h.copy()
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        # Try small random perturbations
        for _ in range(5):
            delta = np.random.randn(N) * 0.01
            h_trial = _project(h + delta, N / 2.0)
            s = _score(h_trial)
            if s < best - 1e-10:
                best = s
                best_h = h_trial.copy()
                h = h_trial.copy()
                break
    
    return best_h


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

    # 1. SDP-based initialization
    if time.time() - t0 < _DEADLINE - 40:
        h_sdp_small = _solve_sdp_gram(25)
        h_sdp = _interpolate_to_n(h_sdp_small, N)
        _update_best(h_sdp)
        
        h_opt = _optimize(h_sdp, max_iter=400)
        _update_best(h_opt)
        
        h_polished = _polish(h_opt)
        _update_best(h_polished)
        
    # 2. Multiple cosine starts with different frequencies
    i = np.arange(N)
    for freq in [1, 2, 3, 4]:
        if time.time() - t0 > _DEADLINE - 20:
            break
        h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N)
        h_opt = _optimize(h_cos, max_iter=300)
        _update_best(h_opt)
        
    # 3. Sine-based starts
    if time.time() - t0 < _DEADLINE - 15:
        for freq in [1, 2, 3]:
            h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * freq * i / N)
            h_opt = _optimize(h_sin, max_iter=200)
            _update_best(h_opt)
        
    # 4. Random starts
    if time.time() - t0 < _DEADLINE - 10:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 8:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_opt = _optimize(h_rand, max_iter=200)
            _update_best(h_opt)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Final polish
    if time.time() - t0 < _DEADLINE - 3:
        best_h = _polish(best_h, max_iter=100)
    
    best_h = _project(best_h, target_sum)
    return best_h
