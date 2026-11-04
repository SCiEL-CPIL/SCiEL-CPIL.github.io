"""Direct L-BFGS-B optimization on low-frequency structured perturbations of h.

Strategy: parameterize h as 1 + sum of low-frequency cosine/sine terms,
optimize the smooth-max surrogate of the autoconvolution peak via L-BFGS-B
with analytic gradients. This is a fundamentally different representation
(direct h-space, low-freq basis) and optimizer (L-BFGS-B with gradients)
than the prior DFT-amplitude/phase Nelder-Mead approach.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
_t0 = time.time()
_DEADLINE = _t0 + 130.0  # safety margin inside the 150 s kill

# Precompute low-frequency basis vectors
# Use the first K frequency components (cos and sin for k=1..K)
_K = 12  # number of frequency components
# Basis: for each k in 1..K, we have cos(2*pi*k*n/N) and sin(2*pi*k*n/N)
_n_idx = np.arange(N)
_basis = np.zeros((2 * _K, N))
for k in range(1, _K + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
# _basis shape: (2*K, N), so h = 1 + _basis.T @ coeffs  (coeffs shape 2*K)
_BASIS_T = _basis.T  # (N, 2*K)


def _evaluate(h):
    """Raw task value (lower is better)."""
    s = float(np.sum(h))
    if s == 0.0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _smooth_max_surrogate(coeffs, beta=50.0):
    """
    Compute a smooth surrogate for the objective.
    h = 1 + _BASIS_T @ coeffs
    We want to minimize: 2*N*max(conv(h)) / sum(h)^2
    Since we fix h centered around 1 (mean=1), sum(h)=N, so denominator = N^2.
    Objective becomes: 2*N*max(conv) / N^2 = 2*max(conv)/N
    
    For the smooth max, we use log-sum-exp on the convolution values.
    But conv has 2N-1 entries; the max is what matters.
    
    Actually, let's work directly: the value is 2*N*max(conv)/sum(h)^2.
    With mean(h)=1 (our parameterization ensures this since basis has zero mean),
    sum(h) = N, so value = 2*N*max(conv)/N^2 = 2*max(conv)/N.
    
    We minimize max(conv). Use smooth approximation via logsumexp.
    """
    h = 1.0 + _BASIS_T @ coeffs
    conv = np.convolve(h, h)
    # Smooth max: (1/beta) * log(sum(exp(beta * conv)))
    # But conv can be negative; shift for numerical stability
    c_max = np.max(conv)
    # For the smooth surrogate, we use: c_max + (1/beta)*log(sum(exp(beta*(conv - c_max))))
    # This is the log-sum-exp trick
    shifted = beta * (conv - c_max)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50.0, 50.0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    smooth_max = c_max + log_sum / beta
    return 2.0 * smooth_max / N


def _smooth_max_grad(coeffs, beta=50.0):
    """Gradient of the smooth max surrogate w.r.t. coeffs."""
    h = 1.0 + _BASIS_T @ coeffs
    conv = np.convolve(h, h)
    c_max = np.max(conv)
    shifted = beta * (conv - c_max)
    shifted = np.clip(shifted, -50.0, 50.0)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted))
    # Weights for the smooth max: w_i = exp(beta*(conv_i - c_max)) / sum(...)
    w = exp_shifted / np.sum(exp_shifted)
    # d smooth_max / d conv_i = w_i
    # d conv_i / d h_j: conv is full convolution, conv[i] = sum_j h[j]*h[i-j] (with proper indexing)
    # For full convolution of length 2N-1: conv[k] = sum_{j=0}^{N-1} h[j] * h[k-j] where valid indices
    # Gradient of conv[k] w.r.t. h[j] = h[k-j] + h[j] (if we consider both terms in the product)
    # Actually for full conv: conv[k] = sum_j h[j]*h[k-j] for j where both indices valid
    # d conv[k]/d h[j] = h[k-j] (from first term) + h[j] (from second term, when k-j' = j i.e. j'=k-j)
    # More carefully: conv[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} h[j]*h[k-j]
    # d conv[k]/d h[j] = h[k-j] + (if j appears in second position: h[k-j] when k-j is valid and j is the "first" index)
    # Actually: if conv[k] = sum_j h[j]*h[k-j], then d conv[k]/d h[j] = h[k-j] + h[k-j] = 2*h[k-j] 
    # Wait no. Let me think again.
    # conv[k] = sum_{j} h[j] * h[k-j]
    # d conv[k] / d h[j] = h[k-j] + h[j] * (d h[k-j]/d h[j])
    # The second term: d h[k-j']/d h[j] = 1 if k-j'=j, i.e., j'=k-j. So when we vary h[j], 
    # the term h[j']*h[k-j'] with j'=j gives h[j]*h[k-j] -> derivative h[k-j]
    # And the term h[j']*h[k-j'] with k-j'=j, i.e., j'=k-j, gives h[k-j]*h[j] -> derivative h[j]
    # So d conv[k]/d h[j] = h[k-j] + h[k-j] = 2*h[k-j] IF both j and k-j are in [0,N-1]
    # But actually it's just h[k-j] from the first sum and h[j] from... 
    # Let me be more careful:
    # conv[k] = sum_{j: 0<=j<N, 0<=k-j<N} h[j]*h[k-j]
    # d/dh[j0] conv[k] = sum_j [delta_{j,j0}*h[k-j] + h[j]*delta_{k-j,j0}]
    # = h[k-j0] (if 0<=k-j0<N) + h[j0] (if 0<=k-j0<N, i.e., j0 in range and k-j0 in range)
    # Wait: the second term: h[j]*delta_{k-j,j0} -> this is nonzero when k-j=j0, i.e., j=k-j0
    # So second term = h[k-j0] if 0<=k-j0<N
    # So total: d conv[k]/d h[j0] = h[k-j0] + h[k-j0] = 2*h[k-j0] if 0<=k-j0<N, else just one term
    # Actually if 0<=k-j0<N: both terms contribute h[k-j0], so 2*h[k-j0]
    # If not: only the first term if 0<=j0<N (always true), so h[k-j0]... but k-j0 out of range means h[k-j0] doesn't exist
    # Let me just compute it numerically or use a different approach.
    
    # Alternative: use the fact that for the gradient, we can compute d conv / d h via:
    # d conv[k]/d h[j] = h[k-j] + h[j] where the indices are interpreted with the understanding
    # that out-of-range indices contribute 0. Actually for full convolution:
    # If we define h_ext with zeros outside [0,N-1], then conv[k] = sum_j h_ext[j]*h_ext[k-j]
    # d conv[k]/d h[j0] = h_ext[k-j0] + h_ext[k-j0] = 2*h_ext[k-j0] (symmetric)
    # Wait that's not right either. Let me just do it directly.
    
    # For efficiency, note that the gradient of the smooth max w.r.t. h is:
    # g_h[j] = sum_k w[k] * d conv[k]/d h[j]
    # And d conv[k]/d h[j] = h[k-j] + h[j] (with appropriate boundary handling)
    # Actually for full convolution, the Jacobian is:
    # d conv[k]/d h[j] = h[k-j] if 0<=k-j<N, plus h[j] if 0<=k-j<N... 
    # I'll compute it as: for each j, the contribution is sum_k w[k]*(h[k-j] + h[j]) for valid k,j
    # = sum_k w[k]*h[k-j] + h[j]*sum_k w[k]
    # The second term: h[j] * sum_k w[k] = h[j] * 1 (since w sums to 1)
    # The first term: sum_k w[k]*h[k-j] = (w * h)[j] (cross-correlation-like)
    # Actually sum_k w[k]*h[k-j] is the convolution of w and h, evaluated at j... 
    # Let's define: (w*h_full)[j] = sum_k w[k]*h[k-j] where h is zero-padded
    # This is the cross-correlation. In numpy: np.correlate(w, h, 'full')[N-1+j] or similar
    
    # Let me just compute d conv/d h matrix-free:
    # grad_h[j] = sum_k w[k] * (h[k-j] if valid + h[j] if valid)
    # = sum_k w[k]*h[k-j] (valid k-j) + h[j] * sum_k w[k] (valid k-j for the h[j] term... )
    # Actually the h[j] term: it appears when in conv[k] = sum_j' h[j']*h[k-j'], the term with j'=j 
    # contributes h[j]*h[k-j], and the term with k-j'=j (j'=k-j) contributes h[k-j]*h[j].
    # So d conv[k]/d h[j] = h[k-j] (from j'=j term) + h[j] (from j'=k-j term, if k-j is valid index)
    # Both require 0<=k-j<N for the respective h values to exist.
    # So: d conv[k]/d h[j] = (h[k-j] if 0<=k-j<N else 0) + (h[j] if 0<=k-j<N else 0)
    # = (h[k-j] + h[j]) if 0<=k-j<N, else 0
    
    # So grad_h[j] = sum_{k: 0<=k-j<N} w[k]*(h[k-j] + h[j])
    # = sum_{k: max(0,j) <= k < min(2N-1, j+N)} w[k]*h[k-j] + h[j]*sum_{k: 0<=k-j<N} w[k]
    
    # For the first term: let m = k-j, so k = m+j, m ranges over [0, N-1]
    # sum_m w[m+j]*h[m] for m where 0<=m+j<2N-1
    # This is like a weighted sum of h with weights shifted by j
    
    # This is getting complex. Let me just compute it directly for the gradient.
    # Since N=400 and conv has 799 entries, and we need grad_h of size 400:
    
    # grad_h[j] = sum_k w[k] * (h[k-j] + h[j]) * indicator(0<=k-j<N)
    # Let's compute using FFT or direct. For N=400, direct O(N^2) per gradient eval is 160K ops, fine.
    
    grad_h = np.zeros(N)
    # For each j, sum over k where 0<=k-j<N:
    # k ranges from j to j+N-1 (clamped to [0, 2N-2])
    k_min = np.maximum(0, 0)  # k starts at 0
    k_max = 2 * N - 2  # k ends at 2N-2
    # For each j: k in [max(0, j), min(2N-2, j+N-1)] such that 0<=k-j<N means k in [j, j+N-1]
    # Intersect with [0, 2N-2]
    for j in range(N):
        k_lo = max(0, j)
        k_hi = min(2 * N - 2, j + N - 1)
        if k_lo <= k_hi:
            wk = w[k_lo:k_hi + 1]
            # h[k-j] for k in [k_lo, k_hi]: indices k-j in [k_lo-j, k_hi-j] = [max(0,j)-j, min(2N-2,j+N-1)-j]
            # = [0 if j>=0 else ..., ...] since j>=0: [0, min(2N-2-j, N-1)]
            h_idx_lo = k_lo - j
            h_idx_hi = k_hi - j
            h_vals = h[h_idx_lo:h_idx_hi + 1]
            grad_h[j] += np.dot(wk, h_vals) + h[j] * np.sum(wk)
    
    # Now grad_coeffs = _BASIS @ grad_h (since h = 1 + _BASIS_T @ coeffs, dh/dcoeffs = _BASIS_T, 
    # so dobj/dcoeffs = _BASIS @ dobj/dh)
    # Wait: h = 1 + _BASIS_T @ coeffs, so dh[i]/dcoeffs[m] = _BASIS_T[i,m] = _basis[m,i]
    # dobj/dcoeffs[m] = sum_i dobj/dh[i] * dh[i]/dcoeffs[m] = sum_i grad_h[i] * _basis[m,i] = (_basis @ grad_h)[m]
    grad_coeffs = _basis @ grad_h
    # Scale by 2/N (from objective = 2*smooth_max/N)
    grad_coeffs *= 2.0 / N
    return grad_coeffs


def _optimize_from_start(coeffs0, beta, maxiter=200):
    """Run L-BFGS-B from a starting point."""
    def f_and_g(c):
        val = _smooth_max_surrogate(c, beta)
        grad = _smooth_max_grad(c, beta)
        return val, grad
    
    res = minimize(f_and_g, coeffs0, jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    return res.x, res.fun


def construct():
    best_h = np.ones(N)
    best_val = 2.0
    rng = np.random.default_rng(42)
    
    # Try multiple betas (smoothness parameters) and random restarts
    betas = [20.0, 50.0, 100.0]
    n_restarts = 5
    
    for beta in betas:
        if time.time() > _DEADLINE - 30:
            break
        for r in range(n_restarts):
            if time.time() > _DEADLINE - 25:
                break
            # Random starting point in coefficient space
            if r == 0:
                coeffs0 = np.zeros(2 * _K)
            else:
                coeffs0 = rng.normal(0, 0.1, 2 * _K)
            
            try:
                coeffs_opt, val = _optimize_from_start(coeffs0, beta, maxiter=150)
            except Exception:
                continue
            
            h = 1.0 + _BASIS_T @ coeffs_opt
            # Ensure sum(h) != 0 and finite
            s = np.sum(h)
            if not np.isfinite(s) or abs(s) < 1e-10:
                continue
            
            v = _evaluate(h)
            if np.isfinite(v) and v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Final: try a few more random restarts with the best beta if time allows
    if time.time() < _DEADLINE - 15:
        for r in range(10):
            if time.time() > _DEADLINE - 10:
                break
            coeffs0 = rng.normal(0, 0.05, 2 * _K)
            try:
                coeffs_opt, val = _optimize_from_start(coeffs0, 50.0, maxiter=100)
            except Exception:
                continue
            h = 1.0 + _BASIS_T @ coeffs_opt
            s = np.sum(h)
            if not np.isfinite(s) or abs(s) < 1e-10:
                continue
            v = _evaluate(h)
            if np.isfinite(v) and v < best_val:
                best_val = v
                best_h = h.copy()
    
    final = _evaluate(best_h)
    if not np.isfinite(final) or final >= 2.0:
        return np.ones(N)
    return best_h
