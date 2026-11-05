"""Optimized program for third autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_value(h):
    """Compute the objective value for a given h."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s ** 2))

def _soft_max(x, beta=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    shifted = beta * (x - x_max)
    shifted = np.clip(shifted, -500, 500)
    return x_max + np.log(np.sum(np.exp(shifted))) / beta

def _objective_soft(x, beta=100.0):
    """Differentiable objective using soft-max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    sm = _soft_max(conv, beta)
    return 2.0 * N * sm / (s ** 2)

def _objective_hard(x):
    """Exact objective with hard max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s ** 2))

def _grad_soft(x, beta=100.0):
    """Gradient of the soft-max objective."""
    h = x
    n = len(h)
    s = np.sum(h)
    conv = np.convolve(h, h, mode='full')
    x_max = np.max(conv)
    shifted = beta * (conv - x_max)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Compute gradient of soft_max w.r.t. h efficiently
    # grad_h[j] = sum_k weights[k] * (h[k-j] if valid + h[j-k] if valid)
    # This is equivalent to: for each j, sum over k where k-j is in [0,n-1] of weights[k]*h[k-j]
    #                    + sum over k where j-k is in [0,n-1] of weights[k]*h[j-k]
    # The first sum is (weights * h)[j] with appropriate alignment
    # The second sum is similar but with reversed indices
    
    # Efficient computation using convolution/correlation
    # Let's use the fact that:
    # sum_k w[k]*h[k-j] for k-j in [0,n-1] is the cross-correlation of w and h
    # Specifically, it's the j-th element of the full cross-correlation
    
    # Actually, let's just do it directly but efficiently using vectorized ops
    # grad_h[j] = sum_{k} w[k] * (h[k-j] * I(0<=k-j<n) + h[j-k] * I(0<=j-k<n))
    
    # First term: for each j, sum_k w[k]*h[k-j] where k-j in [0,n-1]
    # Let m = k-j, so k = m+j, m in [0,n-1]
    # = sum_{m=0}^{n-1} w[m+j] * h[m]
    # This is the dot product of h with w[j:j+n]
    
    # Second term: for each j, sum_k w[k]*h[j-k] where j-k in [0,n-1]
    # Let m = j-k, so k = j-m, m in [0,n-1]
    # = sum_{m=0}^{n-1} w[j-m] * h[m]
    # This is the dot product of h with w[j-n+1:j+1] (reversed)
    
    # Let's compute this using matrix multiplication for efficiency
    # Create a matrix W where W[j,k] = w[k] if 0<=k-j<n else 0
    # Then first term is W @ h
    # Similarly for second term
    
    # Actually, for n=400, O(n^2) is fine. Let's do it directly.
    grad_h = np.zeros(n)
    
    # Precompute which weights are significant
    significant = np.where(weights > 1e-15)[0]
    
    for j in range(n):
        # First term: sum_{m=0}^{n-1} w[m+j] * h[m]
        # m+j must be in [0, 2n-2], so m in [max(0, -j), min(n-1, 2n-2-j)]
        m_start = max(0, 0 - j)  # 0
        m_end = min(n - 1, 2 * n - 2 - j)
        if m_start <= m_end:
            k_indices = m_start + j
            grad_h[j] += np.dot(weights[k_indices], h[m_start:m_end + 1])
        
        # Second term: sum_{m=0}^{n-1} w[j-m] * h[m]
        # j-m must be in [0, 2n-2], so m in [max(0, j-(2n-2)), min(n-1, j)]
        m_start2 = max(0, j - (2 * n - 2))
        m_end2 = min(n - 1, j)
        if m_start2 <= m_end2:
            k_indices2 = j - np.arange(m_start2, m_end2 + 1)
            grad_h[j] += np.dot(weights[k_indices2], h[m_start2:m_end2 + 1])
    
    sm = _soft_max(conv, beta)
    factor = 2.0 * N / (s ** 3)
    grad_obj = factor * (s * grad_h - 2.0 * sm * np.ones(n))
    
    return grad_obj

def construct():
    """Construct step heights to minimize the autocorrelation inequality constant."""
    deadline = time.time() + 115.0
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    def try_candidate(h):
        nonlocal best_h, best_val
        h = np.asarray(h, dtype=np.float64)
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-15:
            return
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Phase 1: Generate diverse initializations using eigenvectors of random Toeplitz matrices
    rng = np.random.default_rng(42)
    initializations = []
    
    # Method 1: Eigenvectors of random symmetric Toeplitz matrices
    for trial in range(10):
        if time.time() > deadline - 60:
            break
        # Generate random symmetric Toeplitz matrix
        toeplitz_diag = rng.standard_normal(N)
        toeplitz_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                toeplitz_matrix[i, j] = toeplitz_diag[abs(i - j)]
        
        # Get top K eigenvectors (most negative eigenvalues for decorrelation)
        eigenvalues, eigenvectors = np.linalg.eigh(toeplitz_matrix)
        K = 20
        top_eigvecs = eigenvectors[:, :K]
        
        # Construct h as linear combination of these eigenvectors with random coefficients
        coeffs = rng.standard_normal(K)
        h = top_eigvecs @ coeffs
        # Add a small constant to ensure sum is not zero
        h += 0.1 * np.mean(h)
        initializations.append(h)
    
    # Method 2: Comb patterns with varying period
    for period in [2, 3, 4, 5, 7]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # Method 3: Higher frequency waves
    for freq in [2, 5, 10, 20]:
        t = np.linspace(0, 2 * np.pi * freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2 * t)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # Method 4: Random smooth signals
    for _ in range(10):
        if time.time() > deadline - 50:
            break
        freqs = rng.uniform(0.1, 3.0, 3)
        amps = rng.uniform(0.1, 1.0, 3)
        phases = rng.uniform(0, 2 * np.pi, 3)
        h = np.ones(N) * 0.5
        for f, a, p in zip(freqs, amps, phases):
            h += a * np.sin(2 * np.pi * f * np.arange(N) / N + p)
        initializations.append(h)
    
    # Method 5: Block patterns
    for block_size in [10, 20, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b * block_size:(b + 1) * block_size] = rng.standard_normal()
        initializations.append(h0)
    
    # Phase 2: Optimize each initialization with L-BFGS-B
    def run_optimization(start_h, max_iter=3000, beta=100.0):
        nonlocal best_h, best_val
        start_h = np.asarray(start_h, dtype=np.float64).copy()
        s = np.sum(start_h)
        if abs(s) < 1e-10:
            start_h[0] += 1.0
        
        def obj(x):
            return _objective_soft(x, beta)
        
        def grad(x):
            return _grad_soft(x, beta)
        
        try:
            res = minimize(obj, start_h, jac=grad, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'gtol': 1e-9, 'ftol': 1e-12,
                                   'maxls': 50})
            hard_val = _objective_hard(res.x)
            if hard_val < best_val:
                best_val = hard_val
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Optimize each initialization
    for h0 in initializations:
        if time.time() > deadline - 20:
            break
        run_optimization(h0, max_iter=2000, beta=100.0)
    
    # Phase 3: Local refinement with higher beta
    if time.time() < deadline - 15:
        for beta in [200.0, 500.0]:
            if time.time() > deadline - 10:
                break
            run_optimization(best_h, max_iter=1000, beta=beta)
    
    # Phase 4: Nelder-Mead polish
    if time.time() < deadline - 5:
        def obj_nm(x):
            return _objective_hard(x)
        
        try:
            res = minimize(obj_nm, best_h, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
            hard_val = _objective_hard(res.x)
            if hard_val < best_val:
                best_val = hard_val
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Try more aggressive structured patterns
    if time.time() < deadline - 5:
        # Two-block with optimized ratio
        for k in range(1, N, 10):
            if time.time() > deadline - 3:
                break
            for ratio in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                h = np.zeros(N)
                h[:k] = ratio
                h[k:] = -1.0
                try_candidate(h)
        
        # Three-block
        for k1 in range(1, N // 3, 20):
            if time.time() > deadline - 2:
                break
            for k2 in range(k1 + 1, 2 * N // 3, 20):
                if time.time() > deadline - 1:
                    break
                for v1 in [1.0, 2.0]:
                    for v2 in [-0.5, -1.0, -2.0]:
                        for v3 in [0.5, 1.0]:
                            h = np.zeros(N)
                            h[:k1] = v1
                            h[k1:k2] = v2
                            h[k2:] = v3
                            try_candidate(h)
    
    return best_h
