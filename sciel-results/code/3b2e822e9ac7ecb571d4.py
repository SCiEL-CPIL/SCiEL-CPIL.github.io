"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses SDP relaxation insights with custom interior-point-inspired solver.
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


def _solve_sdp_small(n_small):
    """
    Solve SDP relaxation for smaller n, then interpolate to N.
    This is a practical approximation of the full SDP.
    """
    G = np.eye(n_small) * 0.25
    t = 1.0
    
    for iteration in range(300):
        if time.time() > _DEADLINE - 20:
            break
            
        cs = np.zeros(n_small)
        for k in range(n_small):
            s = 0.0
            for i in range(n_small):
                j = (i + k) % n_small
                s += G[i, j]
            cs[k] = (2.0 / n_small) * (n_small / 2.0 - s)
        
        max_c = np.max(cs)
        if max_c <= t + 1e-6:
            break
            
        beta = 50.0
        m = np.max(cs)
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
        
        dG = np.zeros((n_small, n_small))
        for k in range(n_small):
            if w[k] == 0:
                continue
            factor = -(2.0 / n_small) * w[k]
            for i in range(n_small):
                j = (i + k) % n_small
                dG[i, j] += factor
                dG[j, i] += factor
        
        dG = (dG + dG.T) / 2.0
        
        alpha = 0.01
        G_new = G - alpha * dG
        G_new = (G_new + G_new.T) / 2.0
        
        eigvals, eigvecs = np.linalg.eigh(G_new)
        eigvals = np.clip(eigvals, 0, None)
        G_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        t = max(t, val)
        G = G_new
        
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argmax(eigvals)
    h_small = np.abs(eigvecs[:, idx]) * np.sqrt(eigvals[idx])
    
    if np.sum(h_small) > 1e-6:
        h_small = h_small * (n_small / 2.0) / np.sum(h_small)
    h_small = np.clip(h_small, 0, 1)
    h_small = _project(h_small, n_small / 2.0)
    
    if n_small == N:
        return h_small
    
    h_fft = np.fft.rfft(h_small)
    n_fft = 2 * N
    h_interp = np.fft.irfft(h_fft, n=n_fft)[:N]
    h_interp = _project(h_interp, N / 2.0)
    
    return h_interp


def _compute_gradient(h, weights):
    """
    Compute gradient of smooth max of C_k with respect to h.
    Uses FFT-based correlation for efficiency.
    """
    n = len(h)
    grad = np.zeros(n)
    
    for k in range(n):
        if weights[k] == 0.0:
            continue
        factor = -(2.0 / n) * weights[k]
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
    
    return grad


def _optimize(h0, max_iter=600, beta_start=5.0, beta_end=1000.0):
    """
    Optimize using smooth max with adaptive step sizing and momentum.
    """
    h = _project(h0.copy(), N / 2.0)
    h_prev = h.copy()
    g_prev = None
    momentum = np.zeros(N)
    momentum_coeff = 0.5
    alpha_history = []
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
            
        beta = beta_start * (beta_end / beta_start) ** (i / max_iter)
        
        cs = _linear_overlap_values(h)
        m = np.max(cs)
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
        
        grad = _compute_gradient(h, w)
        
        # Adaptive step size with momentum
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
                alpha = np.clip(alpha, 1e-6, 10.0)
            else:
                alpha = 1.0 / (np.linalg.norm(grad) + 1e-6)
                alpha = min(alpha, 0.1)
        else:
            alpha = 1.0 / (np.linalg.norm(grad) + 1e-6)
            alpha = min(alpha, 0.05)
        
        # Add momentum
        momentum = momentum_coeff * momentum + (1 - momentum_coeff) * (-grad)
        h_new = h + alpha * momentum
        h_new = _project(h_new, N / 2.0)
        
        # Line search with backtracking
        score_curr = _score(h)
        for ls_iter in range(10):
            score_new = _score(h_new)
            if score_new <= score_curr + 1e-8:
                break
            alpha *= 0.5
            h_new = h + alpha * momentum
            h_new = _project(h_new, N / 2.0)
        
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

    # 1. SDP-inspired start from small problem
    if time.time() - t0 < _DEADLINE - 40:
        h_sdp = _solve_sdp_small(20)
        _update_best(h_sdp)
        h_opt = _optimize(h_sdp)
        _update_best(h_opt)
        
    # 2. Multiple cosine starts with different frequencies and phases
    i = np.arange(N)
    for freq in [1, 2, 3, 4, 5]:
        if time.time() - t0 > _DEADLINE - 20:
            break
        for phase in [0, np.pi/2, np.pi]:
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N + phase)
            h_opt = _optimize(h_cos)
            _update_best(h_opt)
            
    # 3. Random starts with various distributions
    if time.time() - t0 < _DEADLINE - 20:
        for trial in range(5):
            if time.time() - t0 > _DEADLINE - 10:
                break
            if trial == 0:
                h_rand = np.random.uniform(0, 1, N)
            elif trial == 1:
                h_rand = np.random.beta(2, 2, N)
            elif trial == 2:
                h_rand = np.random.exponential(0.5, N)
                h_rand = h_rand / h_rand.max()
            elif trial == 3:
                h_rand = np.sin(np.linspace(0, 4*np.pi, N)) ** 2
            else:
                h_rand = 0.5 + 0.3 * np.sin(2*np.pi*i/N) + 0.2 * np.cos(4*np.pi*i/N)
                h_rand = np.clip(h_rand, 0, 1)
            h_opt = _optimize(h_rand)
            _update_best(h_opt)

    # 4. Final polish on best solution
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_polished = _optimize(best_h, max_iter=200)
        _update_best(h_polished)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project(best_h, target_sum)
    return best_h
