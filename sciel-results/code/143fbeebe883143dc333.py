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


def _project_psd(G):
    """
    Project matrix G onto the PSD cone.
    """
    G = (G + G.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.clip(eigvals, 0, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _solve_sdp_direct(n, max_iter=200):
    """
    Solve the SDP relaxation for size n directly.
    Variables: G (n x n), t (scalar)
    Objective: Minimize t
    Constraints:
      G >= 0
      For each k: sum_{i} G[i, (i+k)%n] >= n/2 - n*t/2
      Tr(G) <= n/2 (since h_i^2 <= h_i and sum h_i = n/2)
      Tr(G) >= n/4 (since sum h_i^2 >= (sum h_i)^2/n)
      
    We use Projected Gradient Descent on G and t.
    """
    # Initialize G as identity * 0.5 (corresponds to h_i=0.5 roughly, but rank n)
    # Better init: G = h0 h0^T where h0 is a good guess (e.g., sinusoid)
    i = np.arange(n)
    h_init = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / n)
    h_init = _project(h_init, n / 2.0)
    G = np.outer(h_init, h_init)
    G = _project_psd(G)
    
    t = 1.0
    
    # Precompute index pairs for constraints
    pairs = []
    for k in range(n):
        idx_i = np.arange(n)
        idx_j = (idx_i + k) % n
        pairs.append((idx_i, idx_j))
        
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 20:
            break
            
        # Compute current overlap sums S_k = sum_i G[i, j_k(i)]
        S = np.zeros(n)
        for k, (idx_i, idx_j) in enumerate(pairs):
            S[k] = np.sum(G[idx_i, idx_j])
            
        # Constraints: S_k >= n/2 - n*t/2  =>  n*t/2 >= n/2 - S_k => t >= 1 - 2*S_k/n
        # Minimize t subject to t >= max_k(1 - 2*S_k/n) and t >= 0
        t_lower = 1.0 - 2.0 * S / n
        t_required = np.max(t_lower)
        
        # Also t should be at least 0
        t_target = max(0.0, t_required)
        
        # Gradient of objective w.r.t G:
        # We want to decrease t. t is driven by the most violated constraint.
        # Let k* be the index of max t_lower.
        # d t / d G[i, j] for the active constraint k*:
        # t = 1 - 2 * S_k*/n
        # d S_k*/d G[i, j] = 1 if (i, j) is a pair in k* (or j, i if k* != 0 and we consider symmetry)
        # Actually S_k = sum_i G[i, (i+k)%n].
        # If we update G, we affect S_k for all k.
        # But in gradient descent for min t, we look at the subgradient of the max function.
        
        # Let's use a smooth approximation or just pick the worst constraint.
        # Subgradient of t w.r.t G:
        # grad_G = - (2/n) * sum_{k in active_set} e_{k}
        # where e_k is the matrix with 1s at positions (i, (i+k)%n).
        
        # To be safe, we use the "worst" constraint k_max.
        k_max = np.argmax(t_lower)
        
        # Gradient of t w.r.t G
        # t = 1 - (2/n) * S_{k_max}
        # d t / d G[i, j] = -(2/n) if G[i, j] is part of S_{k_max}
        # S_{k_max} = sum_i G[i, (i+k_max)%n]
        # So for each i, G[i, (i+k_max)%n] contributes.
        # Also, since G is symmetric, G[j, i] is the same element.
        # If k_max != 0, the pair (i, j) and (j, i) might be counted separately in S_k?
        # S_k = sum_i G[i, (i+k)%n]. This is the sum of specific entries.
        # If k=0, it's the trace.
        # If k!=0, it's the sum of off-diagonals.
        # The matrix G is symmetric.
        # The term G[i, j] appears in S_k if j = (i+k)%n.
        # It also appears in S_{n-k} if i = (j+(n-k))%n => j = (i+k)%n. Same condition.
        # So each off-diagonal element G[i, j] (i!=j) appears in S_k and S_{n-k}.
        # Wait, S_k sums G[i, i+k].
        # If we change G[i, j] by delta, S_k changes by delta if j=i+k.
        # Does S_{n-k} change? S_{n-k} sums G[i, i+n-k].
        # If j = i+k, then i = j-k = j + n - k. So G[j, i] is in S_{n-k}.
        # Since G is symmetric, G[i, j] = G[j, i].
        # So changing G[i, j] changes both S_k and S_{n-k} by delta?
        # No, S_k is defined as sum of specific matrix entries.
        # Let's be precise.
        # S_k = \sum_{i=0}^{n-1} G_{i, (i+k) \bmod n}
        # If we perturb G_{i, j}, it contributes to S_k if j = (i+k) mod n.
        # It contributes to S_{k'} if j = (i+k') mod n.
        # For a fixed pair (i, j), let k = (j-i) mod n.
        # Then it contributes to S_k.
        # Does it contribute to S_{n-k}?
        # S_{n-k} = \sum_{m} G_{m, (m+n-k) \bmod n}.
        # Let m = j. Then term is G_{j, (j+n-k) \bmod n} = G_{j, (j-i) \bmod n + i + n - k}?
        # (j+n-k) mod n = (j-i + i + n - k) mod n = (k + i + n - k) mod n = i.
        # So G_{j, i} is in S_{n-k}.
        # Since G is symmetric, G_{j, i} = G_{i, j}.
        # So the entry G_{i, j} appears in S_k and S_{n-k}.
        
        # So d S_k / d G_{i, j} = 1 if k = (j-i) mod n.
        # d S_{n-k} / d G_{i, j} = 1 if n-k = (j-i) mod n => k = (i-j) mod n.
        
        # In our gradient step for t (driven by k_max):
        # grad t = - (2/n) * (d S_{k_max} / d G)
        # So we subtract from G the positions where k = (j-i) mod n == k_max.
        
        grad_G = np.zeros((n, n))
        # Positions where (j - i) % n == k_max
        for i in range(n):
            j = (i + k_max) % n
            grad_G[i, j] -= 2.0 / n
            # Since G is symmetric, we must maintain symmetry in the update.
            # The projection will handle it, but it's better to be symmetric in grad.
            # Actually, the constraint S_k is not symmetric in G if we treat G as a general matrix.
            # But we project to symmetric PSD.
            # Let's just use the gradient of the constraint function.
            
        # Update G
        alpha = 0.05
        G_new = G - alpha * grad_G
        G_new = _project_psd(G_new)
        
        # Update t
        t = t_target
        
        G = G_new
        
    # Extract h from G
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argmax(eigvals)
    h_small = np.abs(eigvecs[:, idx]) * np.sqrt(eigvals[idx])
    
    # Normalize
    if np.sum(h_small) > 1e-6:
        h_small = h_small * (n / 2.0) / np.sum(h_small)
    h_small = np.clip(h_small, 0, 1)
    h_small = _project(h_small, n / 2.0)
    
    return h_small


def _optimize(h0, max_iter=400, beta_start=10.0, beta_end=500.0):
    """
    Optimize using smooth max with Barzilai-Borwein steps.
    """
    h = _project(h0.copy(), N / 2.0)
    h_prev = h.copy()
    g_prev = None
    beta = beta_start
    
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
        
        grad = np.zeros(N)
        for k in range(N):
            if w[k] == 0.0:
                continue
            factor = -(2.0 / N) * w[k]
            if k > 0:
                grad[k:] += factor * h[:N-k]
            if k < N:
                grad[:N-k] += factor * h[k:]
                
        if g_prev is not None:
            s_vec = h - h_prev
            y_vec = grad - g_prev
            s_norm_sq = np.dot(s_vec, s_vec)
            y_norm_sq = np.dot(y_vec, y_vec)
            sy = np.dot(s_vec, y_vec)
            
            if s_norm_sq > 1e-12 and y_norm_sq > 1e-12:
                alpha_bb1 = s_norm_sq / sy if sy > 1e-12 else 1.0
                alpha_bb2 = sy / y_norm_sq if y_norm_sq > 1e-12 else 1.0
                alpha = 0.5 * (alpha_bb1 + alpha_bb2)
                if alpha < 1e-6:
                    alpha = 1e-6
                if alpha > 10.0:
                    alpha = 10.0
            else:
                alpha = 1.0
        else:
            alpha = 1.0 / (np.linalg.norm(grad) + 1e-6)
            alpha = min(alpha, 0.1)

        h_new = h - alpha * grad
        h_new = _project(h_new, N / 2.0)
        
        score_new = _score(h_new)
        score_curr = _score(h)
        if score_new > score_curr * 1.01 + 1e-6:
            alpha = alpha * 0.5
            h_new = h - alpha * grad
            h_new = _project(h_new, N / 2.0)
            
        h_prev = h
        h = h_new
        g_prev = grad
        
        if np.linalg.norm(h - h_prev) < 1e-8:
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

    # 1. Direct SDP solution for N
    if time.time() - t0 < _DEADLINE - 30:
        h_sdp = _solve_sdp_direct(N, max_iter=100)
        _update_best(h_sdp)
        
        # Refine with smooth max
        h_opt = _optimize(h_sdp)
        _update_best(h_opt)
        
    # 2. Cosine starts
    i = np.arange(N)
    for freq in [1, 2, 3, 5]:
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
    
    best_h = _project(best_h, target_sum)
    return best_h
