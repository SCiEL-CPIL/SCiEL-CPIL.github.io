import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def fft_convolve(h):
    """FFT-based autoconvolution for speed."""
    n = len(h)
    size = 2 * n
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * f, size)[:2*n-1]
    return conv

def smooth_max_surrogate(h, beta=100.0):
    """
    Smooth approximation of max(conv) / (sum h)^2.
    Uses log-sum-exp for smooth max with analytical gradient.
    """
    s = np.sum(h)
    if s == 0:
        return 0.0, np.zeros_like(h)
    
    conv = fft_convolve(h)
    # Scale for numerical stability in log-sum-exp
    conv_scaled = conv / beta
    max_conv_scaled = np.max(conv_scaled)
    # log-sum-exp: max + log(sum(exp(x - max)))
    exp_terms = np.exp(conv_scaled - max_conv_scaled)
    log_sum_exp = max_conv_scaled + np.log(np.sum(exp_terms))
    smooth_max = beta * log_sum_exp
    
    score = smooth_max / (s * s)
    
    # Analytical gradient
    # w_j = exp(conv_j - max) / sum(exp(...))
    w = exp_terms / np.sum(exp_terms)
    
    # d_smooth_max/dh_i = beta * sum_j w_j * d_conv_j/dh_i
    # conv_j = sum_k h_k * h_{j-k} (with j from 0 to 2N-2)
    # d_conv_j/dh_i = h_{j-i} + h_{i-j} (if indices valid), else 0
    # This is: d_conv_j/dh_i = h_{j-i} if 0<=j-i<N, else 0, plus h_{i-j} if 0<=i-j<N, else 0
    # But actually conv_j = sum_k h_k * h_{j-k}, so d_conv_j/dh_i = h_{j-i} if 0<=j-i<N
    # Wait, let me think more carefully.
    # conv_j = sum_{k} h_k * h_{j-k} where k ranges such that both indices are in [0,N-1]
    # So conv_j = sum_{k=max(0,j-N+1)}^{min(j,N-1)} h_k * h_{j-k}
    # d_conv_j/dh_i = h_{j-i} if 0<=j-i<N, else 0
    # Plus: when k=i, we get h_i * h_{j-i}, so derivative w.r.t h_i is h_{j-i}
    # And when j-k=i, i.e., k=j-i, we get h_{j-i} * h_i, so derivative w.r.t h_i is h_{j-i}
    # But these are the same term when k=i and j-k=i, i.e., j=2i.
    # Actually, d_conv_j/dh_i = h_{j-i} if 0<=j-i<N, plus h_{j-i} if 0<=j-i<N (from the other factor)
    # No wait. conv_j = sum_k h_k * h_{j-k}. 
    # d/dh_i [sum_k h_k * h_{j-k}] = h_{j-i} (from the first factor when k=i) + h_{j-i} (from the second factor when j-k=i, i.e., k=j-i)
    # But these are the same term only when i = j-i, i.e., j=2i. In that case, the term is h_i^2, and derivative is 2*h_i.
    # So d_conv_j/dh_i = h_{j-i} if 0<=j-i<N, plus h_{j-i} if 0<=j-i<N.
    # Actually no. Let me be more careful.
    # conv_j = sum_k h_k * h_{j-k}
    # The term involving h_i appears when k=i (giving h_i * h_{j-i}) or when j-k=i i.e. k=j-i (giving h_{j-i} * h_i).
    # If i != j-i (i.e., j != 2i), these are two different terms, each contributing h_{j-i} to the derivative.
    # If i == j-i (i.e., j == 2i), it's the same term h_i^2, contributing 2*h_i to the derivative.
    # So: d_conv_j/dh_i = h_{j-i} if 0<=j-i<N, plus h_{j-i} if 0<=j-i<N.
    # Which simplifies to: 2*h_{j-i} if 0<=j-i<N and j!=2i, or 2*h_i if j=2i.
    # Actually, both cases give 2*h_{j-i} when 0<=j-i<N, because if j=2i then j-i=i.
    # So d_conv_j/dh_i = 2*h_{j-i} if 0<=j-i<N, else 0.
    # Wait, that's not right either. Let me redo.
    # conv_j = sum_k h_k * h_{j-k}
    # d_conv_j/dh_i = sum_k [delta_{k,i} * h_{j-k} + h_k * delta_{j-k,i}]
    # = h_{j-i} + h_{j-i} if both terms exist
    # = 2*h_{j-i} if 0<=j-i<N
    # But this double-counts when k=i and j-k=i simultaneously, i.e., j=2i.
    # When j=2i: d_conv_j/dh_i = h_i + h_i = 2*h_i, which is correct for d(h_i^2)/dh_i.
    # So the formula d_conv_j/dh_i = 2*h_{j-i} if 0<=j-i<N else 0 is correct.
    
    # Now, d_smooth_max/dh_i = beta * sum_j w_j * d_conv_j/dh_i = beta * sum_j w_j * 2*h_{j-i} (if 0<=j-i<N)
    # = 2*beta * sum_j w_j * h_{j-i} (if 0<=j-i<N)
    # Let m = j-i, so j = i+m. m ranges from -i to N-1-i.
    # d_smooth_max/dh_i = 2*beta * sum_{m: 0<=i+m<2N-1 and 0<=m<N} w_{i+m} * h_m
    # Wait, j ranges from 0 to 2N-2. m = j-i ranges from -i to 2N-2-i.
    # We need 0<=m<N, so m ranges from max(0, -i) to min(N-1, 2N-2-i).
    # Since i>=0, max(0,-i)=0. And 2N-2-i >= N-1 iff N-1-i >= 0 iff i <= N-1, which is true.
    # So m ranges from 0 to min(N-1, 2N-2-i).
    # For i <= N-1, 2N-2-i >= N-1, so m ranges from 0 to N-1.
    # So d_smooth_max/dh_i = 2*beta * sum_{m=0}^{N-1} w_{i+m} * h_m (if i+m < 2N-1)
    # For i from 0 to N-1, i+m ranges from i to i+N-1, which is <= 2N-2, so always valid.
    # So d_smooth_max/dh_i = 2*beta * sum_{m=0}^{N-1} w_{i+m} * h_m
    
    # This is a convolution of w and h, specifically (w * h)[i] but with w indexed from 0.
    # Actually, it's sum_m w_{i+m} * h_m = (w_reversed * h)[i] or something similar.
    # Let's just compute it directly using FFT or direct computation.
    
    # d_score/dh_i = (1/s^2) * d_smooth_max/dh_i - 2*smooth_max/s^3 * 1
    # = (1/s^2) * 2*beta * sum_m w_{i+m} * h_m - 2*smooth_max/s^3
    
    # Compute sum_m w_{i+m} * h_m for all i
    # This is a cross-correlation or convolution.
    # Let's define v = w (length 2N-1) and h (length N).
    # We want c[i] = sum_{m=0}^{N-1} v[i+m] * h[m] for i=0,...,N-1.
    # This is like (v * h_flipped)[something].
    # Alternatively, use FFT.
    
    # c[i] = sum_{m=0}^{N-1} w[i+m] * h[m]
    # Let's reverse h: h_rev[m] = h[N-1-m]
    # Then c[i] = sum_{m=0}^{N-1} w[i+m] * h_rev[N-1-m]
    # This is (w * h_rev)[i + N - 1]
    
    h_rev = h[::-1]
    conv_wh = np.convolve(w, h_rev, mode='full')
    # c[i] = conv_wh[i + N - 1] for i=0,...,N-1
    c = conv_wh[N-1 : 2*N-1]
    
    d_smooth_max_dh = 2 * beta * c
    
    grad = d_smooth_max_dh / (s * s) - 2 * smooth_max / (s * s * s)
    
    return score, grad

def objective_and_grad(x, beta):
    """Objective and gradient for L-BFGS-B."""
    h = np.maximum(x, 0)
    score, grad = smooth_max_surrogate(h, beta)
    # Chain rule: if x_i < 0, h_i = 0, so dh_i/dx_i = 0
    mask = x >= 0
    grad_out = np.where(mask, grad, 0)
    return score, grad_out

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Initialize with uniform distribution
    h_uniform = np.full(N, 1.0)
    
    best_h = h_uniform.copy()
    best_score = compute_score(h_uniform)
    
    # Try gradient-based optimization from multiple initializations
    initializations = [
        "uniform",
        "gaussian_centered",
        "bimodal",
        "random_sparse",
        "trapezoid"
    ]
    
    np.random.seed(42)
    
    for init_name in initializations:
        if time.time() > deadline - 25:
            break
        
        if init_name == "uniform":
            h0 = h_uniform.copy()
        elif init_name == "gaussian_centered":
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-x**2 / 0.3)
            h0 = np.maximum(h0, 0)
            if np.sum(h0) > 0:
                h0 = h0 / np.sum(h0) * N
        elif init_name == "bimodal":
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-(x-0.5)**2/0.1) + np.exp(-(x+0.5)**2/0.1)
            h0 = np.maximum(h0, 0)
            if np.sum(h0) > 0:
                h0 = h0 / np.sum(h0) * N
        elif init_name == "trapezoid":
            x = np.linspace(-1, 1, N)
            h0 = np.maximum(1 - np.abs(x), 0)
            if np.sum(h0) > 0:
                h0 = h0 / np.sum(h0) * N
        else:  # random_sparse
            h0 = np.zeros(N)
            num_nonzero = N // 4
            idx = np.random.choice(N, num_nonzero, replace=False)
            h0[idx] = 1.0
            h0 = h0 / np.sum(h0) * N if np.sum(h0) > 0 else h_uniform.copy()
        
        # Optimize with decreasing beta (annealing)
        betas = [30.0, 60.0, 120.0, 250.0, 500.0]
        current_h = h0.copy()
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
            
            # L-BFGS-B optimization with more iterations since gradient is now analytical
            try:
                result = minimize(
                    lambda x: objective_and_grad(x, beta)[0],
                    current_h,
                    jac=lambda x: objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                current_h = np.maximum(result.x, 0)
            except Exception:
                pass
            
            # Evaluate actual score
            actual_score = compute_score(current_h)
            if actual_score < best_score:
                best_score = actual_score
                best_h = current_h.copy()
        
        # Also check the final point
        final_score = compute_score(current_h)
        if final_score < best_score:
            best_score = final_score
            best_h = current_h.copy()
    
    # Final polish: run one more optimization from the best solution with high beta
    if time.time() < deadline - 5:
        try:
            result = minimize(
                lambda x: objective_and_grad(x, 1000.0)[0],
                best_h,
                jac=lambda x: objective_and_grad(x, 1000.0)[1],
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            polished_h = np.maximum(result.x, 0)
            polished_score = compute_score(polished_h)
            if polished_score < best_score:
                best_score = polished_score
                best_h = polished_h.copy()
        except Exception:
            pass
    
    # Final check: ensure non-negativity
    best_h = np.maximum(best_h, 0)
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
