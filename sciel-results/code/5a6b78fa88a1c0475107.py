"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses SDP relaxation with a custom interior-point (barrier) solver.
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


def _sdp_overlap_constraints(G, t):
    """
    Compute slack variables for the overlap constraints.
    Constraint: S_k >= N/2 - N*t/2
    Slack s_k = S_k - (N/2 - N*t/2)
    """
    n = G.shape[0]
    slacks = np.zeros(n)
    for k in range(n):
        # Sum of G[i, (i+k)%n]
        # This is the trace of G @ circulant_shift(k) or similar
        # Efficiently: sum of elements at offset k
        # For k=0, it's trace(G)
        # For k>0, it's sum of G[i, i+k] and G[i+k, i] ? No, G is symmetric.
        # S_k = sum_{i} G[i, (i+k)%n]
        # Note: for k=0, S_0 = sum G[i,i] = Tr(G).
        # For k != 0, S_k = sum_{i=0}^{n-1} G[i, (i+k)%n]
        # Since G is symmetric, sum_{i} G[i, i+k] = sum_{i} G[i+k, i] = sum_{j} G[j, j-k].
        # Actually, the overlap definition uses sum h_i h_{i+k}.
        # If G = h h^T, G[i, j] = h_i h_j.
        # sum_i G[i, i+k] = sum_i h_i h_{i+k}. This matches.
        
        # Vectorized computation for all k is possible via FFT or direct sum.
        # For n=100, direct sum is fine.
        idx1 = np.arange(n)
        idx2 = (idx1 + k) % n
        s_val = np.sum(G[idx1, idx2])
        slacks[k] = s_val - (n / 2.0 - n * t / 2.0)
    return slacks


def _solve_sdp_ipm(n, max_iter=100, mu_init=1.0, mu_red=0.5):
    """
    Solve the SDP relaxation using a barrier method.
    Variables: G (symmetric n x n), t (scalar).
    Objective: Minimize t - (1/mu) * sum(log(eig(G))) - (1/mu) * sum(log(slacks))
    """
    # Initialize G as positive definite
    # G = 0.5 * I + small noise to break symmetry?
    # We want G to be close to h h^T where sum h = n/2.
    # If h = 0.5 * 1, G = 0.25 * 11^T.
    # Let's start with G = 0.25 * np.ones((n,n))
    G = 0.25 * np.ones((n, n))
    t = 1.0
    mu = mu_init

    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
            
        # Compute eigenvalues of G for barrier
        eigvals, eigvecs = np.linalg.eigh(G)
        # Ensure positive eigenvalues for log
        eigvals = np.clip(eigvals, 1e-10, None)
        
        # Compute slacks
        slacks = _sdp_overlap_constraints(G, t)
        slacks = np.clip(slacks, 1e-10, None)
        
        # Objective value (for monitoring)
        obj = t - (1.0 / mu) * (np.sum(np.log(eigvals)) + np.sum(np.log(slacks)))
        
        # Gradient computation
        # d/dG [ -1/mu sum log(eig(G)) ] = -1/mu G^{-1}
        # d/dt [ -1/mu sum log(slacks_k) ]
        # slack_k = S_k(G) - n/2 + n*t/2
        # d slack_k / dt = n/2
        # d/dt [ -1/mu sum log(slack_k) ] = -1/mu sum (1/slack_k) * (n/2)
        
        # d/dG [ -1/mu sum log(slack_k) ]
        # d slack_k / dG = E_k (matrix with 1s at (i, i+k) and (i+k, i) if k!=0, 2 at (i,i) if k=0? No.)
        # S_k = sum_i G[i, i+k].
        # d S_k / d G[a,b] is 1 if (a,b) = (i, i+k) for some i.
        # So d S_k / d G = M_k where M_k[i, i+k] = 1.
        # Since G is symmetric, we treat G as the variable.
        # The gradient with respect to the matrix G (in the symmetric space) is:
        # grad_G = -1/mu * (G^{-1} - sum_k (1/slack_k) * M_k)
        
        # Compute G^{-1}
        # Using eigendecomposition: G^{-1} = V diag(1/lambda) V^T
        G_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        
        # Compute sum_k (1/slack_k) * M_k
        # M_k has 1s at positions (i, i+k mod n).
        # Let's construct the gradient term from slacks.
        grad_slack = np.zeros((n, n))
        for k in range(n):
            coeff = 1.0 / slacks[k]
            idx1 = np.arange(n)
            idx2 = (idx1 + k) % n
            grad_slack[idx1, idx2] += coeff
            # Note: since G is symmetric, and S_k = sum G[i, i+k],
            # if we vary G[a,b], it affects S_k if (a,b) is a shifted pair.
            # The term in the objective is sum log(slack_k).
            # The derivative w.r.t G is sum_k (1/slack_k) * dS_k/dG.
            # dS_k/dG is the matrix with 1s at (i, i+k).
            # Because G is symmetric, G[a,b] = G[b,a].
            # If we use the full matrix G as variable, the gradient is just the sum of M_k.
            # However, we must be careful with the factor of 2 if we consider symmetric updates.
            # Let's stick to full matrix gradient.
            
        grad_G = -1.0 / mu * (G_inv - grad_slack)
        
        # Gradient w.r.t t
        grad_t = 1.0 - 1.0 / mu * np.sum((1.0 / slacks) * (n / 2.0))
        
        # Newton Step
        # We need the Hessian or approximate it.
        # For simplicity, we use a scaled gradient step with line search.
        # This is a "barrier method with gradient descent" which is robust.
        
        # Scale gradient
        norm_G = np.linalg.norm(grad_G)
        norm_t = abs(grad_t)
        
        # Step size
        alpha = 1.0 / (norm_G + norm_t + 1e-6)
        alpha = min(alpha, 0.5) # Cap step size
        
        G_new = G - alpha * grad_G
        t_new = t - alpha * grad_t
        
        # Ensure G is symmetric
        G_new = (G_new + G_new.T) / 2.0
        
        # Project G to PSD (if eigenvalues went negative)
        eigvals_new, eigvecs_new = np.linalg.eigh(G_new)
        if np.min(eigvals_new) < 1e-8:
             eigvals_new = np.clip(eigvals_new, 1e-8, None)
             G_new = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T
             
        # Check slacks for new G, t
        slacks_new = _sdp_overlap_constraints(G_new, t_new)
        
        # If slacks are invalid, reduce step size
        if np.min(slacks_new) < 1e-8:
             # Backtracking line search
             alpha = 0.5
             for _ in range(20):
                 G_new = G - alpha * grad_G
                 t_new = t - alpha * grad_t
                 G_new = (G_new + G_new.T) / 2.0
                 slacks_new = _sdp_overlap_constraints(G_new, t_new)
                 if np.min(slacks_new) > 1e-8 and np.min(np.linalg.eigvalsh(G_new)) > 1e-8:
                     break
                 alpha *= 0.5
             else:
                 # If still invalid, keep old G, t and reduce mu
                 mu *= mu_red
                 continue
                 
        G = G_new
        t = t_new
        
        # Update mu
        if it % 5 == 0:
            mu *= mu_red
        if mu < 1e-4:
            break
            
    # Extract h from G
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argmax(eigvals)
    h = np.abs(eigvecs[:, idx]) * np.sqrt(eigvals[idx])
    
    # Normalize and project
    h = _project(h, n / 2.0)
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

    # 1. SDP Interior Point Method
    if time.time() - t0 < _DEADLINE - 30:
        h_sdp = _solve_sdp_ipm(N)
        _update_best(h_sdp)
        
    # 2. Cosine starts (fallback/improvement)
    i = np.arange(N)
    for freq in [1, 2, 3]:
        if time.time() - t0 > _DEADLINE - 15:
            break
        h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N)
        # Use a simple gradient descent for polish
        # Re-using the logic from previous _optimize but simpler
        h_curr = h_cos.copy()
        for _ in range(50):
            cs = _linear_overlap_values(h_curr)
            m = np.max(cs)
            beta = 100.0
            exp_terms = np.exp(beta * (cs - m))
            w = exp_terms / np.sum(exp_terms)
            
            grad = np.zeros(N)
            for k in range(N):
                if w[k] == 0.0: continue
                factor = -(2.0 / N) * w[k]
                if k > 0:
                    grad[k:] += factor * h_curr[:N-k]
                if k < N:
                    grad[:N-k] += factor * h_curr[k:]
                    
            h_curr = h_curr - 0.1 * grad
            h_curr = _project(h_curr, target_sum)
            
        _update_best(h_curr)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project(best_h, target_sum)
    return best_h
