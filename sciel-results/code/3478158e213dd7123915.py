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
    # For small n, we can use scipy to solve the SDP approximately
    # by converting to a QP with PSD constraint handled via eigenvalue clipping
    
    # The SDP:
    # Minimize t
    # Subject to:
    #   G >= 0 (PSD)
    #   For each k: (2/n) * (n/2 - sum_{i=k}^{n-1} G[i, (i+k)%n]) <= t
    #   sum_i G[i,i] <= n (since h_i^2 <= h_i <= 1, sum h_i^2 <= sum h_i = n/2)
    #   Actually: sum h_i = n/2, and G = h h^T means trace(G) = sum h_i^2
    
    # For small n, let's use a simple iterative method
    # Initialize G as identity * 0.5
    G = np.eye(n_small) * 0.25  # G[i,i] = h_i^2, if h_i=0.5, G[i,i]=0.25
    t = 1.0
    
    for iteration in range(200):
        if time.time() > _DEADLINE - 20:
            break
            
        # Compute violations
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
            
        # Gradient descent on G and t
        # dL/dt = 1
        # dL/dG[i,j] from constraint k: -(2/n) * w_k where w_k is weight
        
        # Use smooth max
        beta = 50.0
        m = np.max(cs)
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
        
        # Gradient with respect to G
        dG = np.zeros((n_small, n_small))
        for k in range(n_small):
            if w[k] == 0:
                continue
            factor = -(2.0 / n_small) * w[k]
            for i in range(n_small):
                j = (i + k) % n_small
                dG[i, j] += factor
                dG[j, i] += factor  # Symmetric
        
        # Ensure symmetry
        dG = (dG + dG.T) / 2.0
        
        # Update G: G_new = G - alpha * dG, then project to PSD
        alpha = 0.01
        G_new = G - alpha * dG
        G_new = (G_new + G_new.T) / 2.0
        
        # Project to PSD via eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(G_new)
        eigvals = np.clip(eigvals, 0, None)
        G_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Update t
        t = max(t, val)
        
        G = G_new
        
    # Extract h from G: if G = h h^T, then h_i = sqrt(G[i,i])
    # But G may not be rank-1, so use the largest eigenvalue
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argmax(eigvals)
    h_small = np.abs(eigvecs[:, idx]) * np.sqrt(eigvals[idx])
    
    # Normalize to sum = n_small/2
    if np.sum(h_small) > 1e-6:
        h_small = h_small * (n_small / 2.0) / np.sum(h_small)
    h_small = np.clip(h_small, 0, 1)
    h_small = _project(h_small, n_small / 2.0)
    
    # Interpolate to size N
    if n_small == N:
        return h_small
    
    # Use FFT-based interpolation
    h_fft = np.fft.rfft(h_small)
    n_fft = 2 * N
    h_interp = np.fft.irfft(h_fft, n=n_fft)[:N]
    h_interp = _project(h_interp, N / 2.0)
    
    return h_interp


def _optimize(h0, max_iter=600, beta_start=20.0, beta_end=1000.0):
    """
    Optimize using smooth max with adaptive step size and momentum.
    """
    h = _project(h0.copy(), N / 2.0)
    h_prev = h.copy()
    g_prev = None
    momentum = np.zeros(N)
    mu = 0.9  # momentum coefficient
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
            
        # Adaptive beta: increase as we converge
        progress = i / max_iter
        beta = beta_start * (beta_end / beta_start) ** progress
        
        cs = _linear_overlap_values(h)
        m = np.max(cs)
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
        
        grad = np.zeros(N)
        for k in range(N):
            if w[k] == 0.0:
                continue
            factor = -(2.0 / N) * w[k]
            if k > 0:
                grad[k:] += factor * h[:N-k]
            if k < N:
                grad[:N-k] += factor * h[k:]
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Compute step size
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
                alpha = 1.0 / (grad_norm + 1e-6)
                alpha = min(alpha, 0.1)
        else:
            alpha = 1.0 / (grad_norm + 1e-6)
            alpha = min(alpha, 0.1)
        
        # Apply momentum
        momentum = mu * momentum + (1 - mu) * alpha * grad
        h_new = h - momentum
        h_new = _project(h_new, N / 2.0)
        
        # Backtracking if score worsens
        score_new = _score(h_new)
        score_curr = _score(h)
        backtrack = 0
        while score_new > score_curr * 1.005 + 1e-8 and backtrack < 10:
            momentum = momentum * 0.5
            h_new = h - momentum
            h_new = _project(h_new, N / 2.0)
            score_new = _score(h_new)
            backtrack += 1
        
        h_prev = h
        h = h_new
        g_prev = grad
        
        if np.linalg.norm(h - h_prev) < 1e-9:
            break
            
    return h


def _local_refine(h, max_iter=200):
    """
    Local refinement using coordinate descent on the smooth max.
    """
    h = _project(h.copy(), N / 2.0)
    beta = 500.0
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        cs = _linear_overlap_values(h)
        m = np.max(cs)
        
        best_improvement = 1e-10
        best_new_h = None
        
        # Try updating each coordinate with small perturbations
        for j in range(N):
            for delta in [-0.005, -0.002, 0.002, 0.005]:
                h_trial = h.copy()
                h_trial[j] += delta
                h_trial = _project(h_trial, N / 2.0)
                if not np.all(np.isfinite(h_trial)):
                    continue
                s = _score(h_trial)
                if s < m - best_improvement:
                    best_improvement = m - s
                    best_new_h = h_trial.copy()
        
        if best_new_h is None:
            # Try even smaller steps
            for j in range(N):
                for delta in [-0.001, -0.0005, 0.0005, 0.001]:
                    h_trial = h.copy()
                    h_trial[j] += delta
                    h_trial = _project(h_trial, N / 2.0)
                    if not np.all(np.isfinite(h_trial)):
                        continue
                    s = _score(h_trial)
                    if s < m - 1e-10:
                        best_new_h = h_trial.copy()
                        break
                if best_new_h is not None:
                    break
        
        if best_new_h is None:
            break
        
        h = best_new_h
    
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
    if time.time() - t0 < _DEADLINE - 30:
        h_sdp = _solve_sdp_small(20)
        _update_best(h_sdp)
        h_opt = _optimize(h_sdp)
        _update_best(h_opt)
        
    # 2. Cosine starts
    i = np.arange(N)
    for freq in [1, 2, 3]:
        if time.time() - t0 > _DEADLINE - 15:
            break
        h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N)
        h_opt = _optimize(h_cos)
        _update_best(h_opt)
        
    # 3. Random starts
    if time.time() - t0 < _DEADLINE - 15:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 10:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_opt = _optimize(h_rand)
            _update_best(h_opt)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Final local refinement
    if time.time() - t0 < _DEADLINE - 5:
        best_h = _local_refine(best_h, max_iter=100)
    
    best_h = _project(best_h, target_sum)
    return best_h
