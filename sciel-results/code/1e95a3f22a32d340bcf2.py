"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

K = 12  # Number of Fourier modes (reduced from 20 for speed)
DIM = 2 * K  # Parameter dimension

# Precompute Fourier basis
_n = np.arange(N)
_cos_basis = np.array([np.cos(2 * np.pi * k * _n / N) for k in range(1, K)])  # (K-1, N)
_sin_basis = np.array([np.sin(2 * np.pi * k * _n / N) for k in range(1, K)])  # (K-1, N)

def _value(h):
    """Compute the evaluation metric."""
    s = np.sum(h)
    if s == 0:
        return np.inf
    ac = np.convolve(h, h)
    return abs(2 * N * np.max(ac) / (s ** 2))

def _reconstruct(x):
    """
    Reconstruct h from parameter vector x (vectorized).
    x = [a0, a1, ..., a_{K-1}, b1, ..., b_{K-1}]
    h[n] = a0 + sum_{k=1}^{K-1} (a_k * cos(2*pi*k*n/N) + b_k * sin(2*pi*k*n/N))
    """
    h = np.full(N, x[0])
    # x[1:K] are cosine coefficients, x[K:2*K] are sine coefficients
    h += _cos_basis.T @ x[1:K]
    h += _sin_basis.T @ x[K:2*K]
    return h

def _objective_grad(x):
    """
    Compute smooth objective and its analytic gradient.
    Uses log-sum-exp for smooth max of autocorrelation.
    """
    h = _reconstruct(x)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(x)
    
    ac = np.convolve(h, h)
    
    # Smooth max via log-sum-exp
    beta = 50.0  # Temperature for smoothing
    ac_scaled = ac / beta
    max_ac_scaled = np.max(ac_scaled)
    exp_terms = np.exp(ac_scaled - max_ac_scaled)
    sum_exp = np.sum(exp_terms)
    log_sum_exp = max_ac_scaled + np.log(sum_exp)
    max_ac_smooth = log_sum_exp * beta
    
    # Weights for gradient
    weights = exp_terms / sum_exp  # Softmax weights
    
    obj = abs(2 * N * max_ac_smooth / (s ** 2))
    
    # Gradient via chain rule
    # d(obj)/d(ac) = 2*N/s^2 * weights (with sign handled by abs)
    # Since max_ac_smooth >= 0 typically, abs just multiplies by sign
    sign = np.sign(2 * N * max_ac_smooth / (s ** 2))
    
    # d(max_ac_smooth)/d(ac_j) = beta * weights_j
    # d(ac)/d(h_i): ac = conv(h,h), ac[m] = sum_i h[i]*h[m-i]
    # d(ac[m])/d(h[i]) = h[m-i] + h[m-i] if m-i != i, else 2*h[i]
    # More precisely: ac[m] = sum_{i=0}^{N-1} h[i]*h[m-i] for valid indices
    # Using FFT for efficiency:
    # d(ac)/d(h) can be computed as 2*conv(h, h) but that's not quite right
    # Actually: ac[m] = sum_i h[i]*h[m-i], so d(ac[m])/d(h[j]) = h[m-j] + (h[j] if m-j==j else 0)
    # Simpler: use the fact that d(conv(h,h))/dh = 2*conv(h, .) but need care
    # Let's use: grad_ac = 2 * np.convolve(h, h, mode='full') but shifted...
    # Actually the cleanest way:
    # ac = fft2 approach or just:
    # d(ac[m])/d(h[j]) = h[m-j] for m-j in [0,N-1] plus h[j] if m-j==j
    # This is equivalent to: for each j, d(ac)/d(h[j]) = h shifted by j (circular-like)
    # Let's just do it directly with the convolution structure
    
    # d(ac)/d(h) shape: (2N-1, N) but we can compute grad w.r.t h as:
    # grad_h[m] = sum_j weights_j * d(ac[j])/d(h[m])
    # d(ac[j])/d(h[m]) = h[j-m] if 0<=j-m<N, plus h[m] if j-m==m
    # = h[j-m] (with boundary check) + delta(j-m, m)*h[m]
    # = h[j-m] + h[m]*delta_{j,2m}
    
    # Efficient computation:
    # For each lag j, the contribution to grad_h is weights[j] * (h shifted by j)
    # h_shifted_j[m] = h[j-m] if 0<=j-m<N else 0
    # This is a cross-correlation-like operation
    
    # Let's compute it as: grad_h = 0
    # for j in range(2N-1):
    #   if j < N:
    #     grad_h[0:N-j] += weights[j] * h[j:N]  # h[j-m] for m=0..N-j-1
    #     grad_h[j:N] += weights[j] * h[0:N-j]  # h[j-m] for m=j..N-1, j-m=0..N-j-1
    #   else:
    #     # j >= N, j-m in [0,N-1] means m in [j-N+1, N-1]... wait
    #     # j-m >= 0 => m <= j, j-m < N => m > j-N
    #     # So m in [max(0, j-N+1), min(N-1, j)]
    #     lo = max(0, j - N + 1)
    #     hi = min(N - 1, j)
    #     for m in range(lo, hi+1):
    #       grad_h[m] += weights[j] * h[j-m]
    #   # Plus the diagonal term: if j == 2*m, add weights[j]*h[m]
    #   if j % 2 == 0 and j//2 < N:
    #     grad_h[j//2] += weights[j] * h[j//2]
    
    # This is O(N^2) per gradient eval which is fine for N=400
    # But let's be smarter using FFT or just accept O(N^2)
    
    # Actually, let me reconsider. The autocorrelation is:
    # ac[k] = sum_{i} h[i] * h[i+k] for k=0..N-1 (one-sided)
    # But np.convolve gives full: ac_full[m] = sum_i h[i]*h[m-i]
    # For m=0..2N-2
    
    # d(ac_full[m])/d(h[j]) = h[m-j] if 0<=m-j<N, plus extra h[j] if m-j==j (i.e., m=2j)
    # So: d(ac_full[m])/d(h[j]) = h[m-j] * [0<=m-j<N] + h[j] * [m==2j]
    
    # grad_h[j] = sum_m weights[m] * (h[m-j]*[0<=m-j<N] + h[j]*[m==2j])
    #            = sum_{m: 0<=m-j<N} weights[m]*h[m-j] + h[j]*weights[2j] (if 2j<2N-1)
    # Let i = m-j, so m = i+j:
    # = sum_{i: 0<=i<N} weights[i+j]*h[i] + h[j]*weights[2j]
    # = dot(h, weights[j:j+N]) + h[j]*weights[2j] (with bounds checking)
    
    grad_h = np.zeros(N)
    for j in range(N):
        # First term: sum_{i=0}^{N-1} weights[i+j] * h[i], where i+j < 2N-1
        idx_start = j
        idx_end = min(2 * N - 1, j + N)
        if idx_start < idx_end:
            grad_h[j] += np.dot(h, weights[idx_start:idx_end])
        # Second term: h[j] * weights[2j] if 2j < 2N-1
        if 2 * j < 2 * N - 1:
            grad_h[j] += h[j] * weights[2 * j]
    
    # Now: obj = |2*N*max_ac_smooth/s^2|
    # d(obj)/d(h) = sign * 2*N/s^2 * d(max_ac_smooth)/d(h) - sign * 2*N*max_ac_smooth*2/s^3 * d(s)/d(h)
    # d(max_ac_smooth)/d(h) = beta * grad_h (since max_ac_smooth = beta*log_sum_exp, d/dh = beta * sum weights * d(ac)/dh)
    # Wait: max_ac_smooth = beta * (max_ac_scaled + log(sum_exp))
    # d(max_ac_smooth)/d(ac[m]) = beta * weights[m]
    # d(max_ac_smooth)/d(h[j]) = beta * sum_m weights[m] * d(ac[m])/d(h[j]) = beta * grad_h[j]
    
    # d(s)/d(h[j]) = 1
    
    factor = sign * 2 * N / (s ** 2)
    grad_h = factor * (beta * grad_h - 2 * max_ac_smooth / s * np.ones(N))
    
    # Now chain through x: h = _reconstruct(x)
    # h[0] = x[0] + sum_k cos_k(0)*x[k+1] + sum_k sin_k(0)*x[K+k]
    # h[n] = x[0] + sum_{k=1}^{K-1} x[k]*cos_k(n) + x[K+k-1]*sin_k(n)
    # dx_0 = sum_j grad_h[j] * dh_j/dx_0 = sum_j grad_h[j] * 1 = sum(grad_h)
    # dx_k (k=1..K-1) = sum_j grad_h[j] * cos_k(j)
    # dx_{K+k-1} = sum_j grad_h[j] * sin_k(j)
    
    grad_x = np.zeros(DIM)
    grad_x[0] = np.sum(grad_h)
    for k in range(1, K):
        grad_x[k] = np.dot(grad_h, _cos_basis[k-1])
        grad_x[K + k - 1] = np.dot(grad_h, _sin_basis[k-1])
    
    return obj, grad_x

def _objective(x):
    """Objective wrapper for minimize (no gradient)."""
    val, _ = _objective_grad(x)
    return val

def construct():
    """
    Construct the optimal step heights using optimization.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_x = None
    best_val = np.inf
    
    # Seed start: all ones corresponds to a0 = 1, others = 0
    x_seed = np.zeros(DIM)
    x_seed[0] = 1.0
    
    starts = [x_seed]
    
    # Add diverse random starts
    np.random.seed(42)
    n_starts = 15
    for i in range(n_starts):
        x_rand = np.random.randn(DIM) * (0.3 + 0.1 * i / n_starts)
        x_rand[0] += 1.0  # Bias toward positive DC
        starts.append(x_rand)
    
    # Also try some structured starts
    for freq in [3, 5, 7]:
        x_struct = np.zeros(DIM)
        x_struct[0] = 1.0
        x_struct[freq] = 0.3  # Add a cosine component
        starts.append(x_struct)
    
    for x0 in starts:
        if time.time() > deadline - 10:
            break
        
        try:
            res = minimize(
                _objective,
                x0,
                jac=lambda x: _objective_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            h_candidate = _reconstruct(res.x)
            val = _value(h_candidate)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if best_x is not None and time.time() < deadline - 10:
        try:
            res_final = minimize(
                _objective,
                best_x,
                jac=lambda x: _objective_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            h_candidate = _reconstruct(res_final.x)
            val = _value(h_candidate)
            if val < best_val:
                best_val = val
                best_x = res_final.x
        except Exception:
            pass
    
    if best_x is not None:
        return _reconstruct(best_x)
    
    # Fallback to seed
    return np.ones(N)

# EVOLVE-BLOCK-END
