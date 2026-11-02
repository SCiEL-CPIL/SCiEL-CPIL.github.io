"""
Program for the first autocorrelation inequality.
Strategy: Use a spectral/gradient-based approach with a smooth-max surrogate objective,
combined with FFT-accelerated convolution and gradient descent with momentum.
This is a different design family: instead of discrete two-block structures + coordinate descent,
we optimize in a continuous-like space using gradients and a differentiable surrogate.
"""
import time
import numpy as np
from scipy import optimize

N = 600
dx = 0.5 / N

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s < 1e-12:
        return 0.0
    # FFT-based convolution for speed
    n_conv = 2 * N - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = np.zeros(n_fft, dtype=np.complex128)
    H[:N] = h
    conv = np.real(np.fft.ifft(np.fft.fft(H) ** 2))[:n_conv]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_max_surrogate(conv, beta):
    """
    Smooth approximation of max(conv) using log-sum-exp.
    """
    # Numerical stability
    m = np.max(conv)
    return m + np.log(np.sum(np.exp(conv - m))) / beta

def objective_grad(h, beta=50.0):
    """
    Compute smooth-max surrogate objective and its gradient.
    Returns (value, grad).
    """
    s = np.sum(h)
    if s < 1e-12:
        return 0.0, np.zeros_like(h)
    
    n_conv = 2 * N - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = np.zeros(n_fft, dtype=np.complex128)
    H[:N] = h
    conv = np.real(np.fft.ifft(np.fft.fft(H) ** 2))[:n_conv]
    
    # Smooth max surrogate
    m = np.max(conv)
    exp_terms = np.exp(conv - m)
    sum_exp = np.sum(exp_terms)
    value = m + np.log(sum_exp) / beta
    score = value / (s * s)
    
    # Gradient of score with respect to h
    # d(score)/d(h[i]) = [1/(s^2) * d(value)/d(h[i]) - 2*s/(s^2) * value/s * ... ]
    # Actually: score = value / s^2
    # d(score)/dh[i] = (1/s^2) * d(value)/dh[i] - 2*value/s^3 * s
    # Wait: d(score)/dh[i] = (d(value)/dh[i] * s^2 - value * 2*s * dh[i]/dh[i]) / s^4
    # = d(value)/dh[i] / s^2 - 2 * value * h[i] / s^3
    
    # d(value)/dh[i]: value = smooth_max(conv)/1, but conv depends on h.
    # conv[k] = sum_j h[j] h[j+k] (for valid indices)
    # Using FFT: conv = ifft(fft(h)^2)
    # d(conv[k])/dh[i] = h[k-i] + h[k+i] (with boundary handling, but FFT handles circular, 
    # and since we zero-pad, it's equivalent to linear convolution with zeros outside)
    # Actually for linear conv via zero-padding: d(conv[k])/dh[i] = h[(k-i) mod N] if in range, else 0, 
    # plus h[(k+i) mod N]... no, it's simpler:
    # conv[k] = sum_{j=0}^{N-1} h[j] h[j+k] where h[j+k]=0 if j+k>=N
    # d(conv[k])/dh[i] = h[i+k] (if i+k < N) + h[i-k] (if i-k >= 0)
    #   [from j=i term: h[i]h[i+k] -> derivative w.r.t h[i] is h[i+k]]
    #   [from j=i-k term: h[i-k]h[i] -> derivative w.r.t h[i] is h[i-k]]
    
    # So d(conv)/dh[i] = h[i+k] + h[i-k] summed over k... no, for each k separately.
    # d(value)/dh[i] = sum_k (d(value)/d(conv[k])) * (d(conv[k])/dh[i])
    # d(value)/d(conv[k]) = exp(conv[k]-m) / sum_exp / beta  (for the log-sum-exp part)
    #                     + delta_{k, argmax} * 1  (if we had hard max, but smooth max doesn't have that term exactly)
    # Actually smooth_max = m + log(sum exp)/beta
    # d(smooth_max)/d(conv[k]) = delta_{k,argmax} + exp(conv[k]-m)/(sum_exp * beta)
    # But since m = max(conv), and we're using the soft version, the gradient is:
    # d(smooth_max)/d(conv[k]) = exp(conv[k]-m) / (sum_exp * beta) + delta_{k, argmax} * (1 - exp(conv[argmax]-m)/sum_exp)
    # This is getting complicated. Let's just use the standard log-sum-exp gradient:
    # If we define f(conv) = log(sum exp(conv/beta)) * beta, then f is a smooth max.
    # But I used m + log(sum exp(conv-m))/beta. Let's stick with a simpler form.
    
    # Let's redefine: surrogate = log(sum exp(conv / beta)) * beta
    # Then d(surrogate)/d(conv[k]) = exp(conv[k]/beta) / sum(exp(conv[j]/beta))
    
    # Let's recompute with this simpler form for gradient clarity.
    # But I already computed value with the other form. Let's just compute the gradient numerically 
    # or use the chain rule carefully.
    
    # For efficiency, let's compute the gradient of the smooth max using the probabilities.
    # p_k = exp(conv[k] - m) / sum_exp
    # d(smooth_max)/d(conv[k]) = p_k / beta  (for the log-sum-exp part) 
    # But smooth_max = m + log(sum_exp)/beta, and m = max(conv).
    # The derivative of m w.r.t conv[k] is 1 if k=argmax, else 0.
    # So d(smooth_max)/d(conv[k]) = delta_{k,argmax} + p_k / beta * (1 - p_argmax) ... no.
    
    # Let's just use the standard: for f(x) = log(sum exp(x_i/beta)), f is smooth max / beta.
    # Let g = beta * f = beta * log(sum exp(conv/beta)).
    # d(g)/d(conv[k]) = exp(conv[k]/beta) / sum(exp(conv[j]/beta))
    
    # I'll switch to this definition for cleaner gradients.
    # But I already have the code. Let's just compute the gradient of score using the 
    # chain rule with the current value definition.
    
    # Actually, let's just recompute everything with the beta*log(sum exp(conv/beta)) form.
    
    # Redo:
    conv_scaled = conv / beta
    max_cs = np.max(conv_scaled)
    exp_cs = np.exp(conv_scaled - max_cs)
    sum_exp_cs = np.sum(exp_cs)
    surrogate = beta * (max_cs + np.log(sum_exp_cs))
    score = surrogate / (s * s)
    
    # Gradient of surrogate w.r.t conv[k]:
    # d(surrogate)/d(conv[k]) = exp(conv[k]/beta) / sum(exp(conv[j]/beta)) = exp_cs[k] / sum_exp_cs
    p_k = exp_cs / sum_exp_cs  # shape (n_conv,)
    
    # Now d(surrogate)/d(h[i]) = sum_k p_k * d(conv[k])/d(h[i])
    # d(conv[k])/d(h[i]) = h[i+k] (if i+k < N) + h[i-k] (if i-k >= 0)
    # This is equivalent to: for fixed i, sum_k p_k * (h[i+k] + h[i-k])
    # = sum_k p_k * h[i+k] + sum_k p_k * h[i-k]
    # Let j = i+k, then k = j-i, sum_j p_{j-i} * h[j] (for valid k)
    # This is a convolution of p with h, but with shifted indices.
    
    # More precisely:
    # term1[i] = sum_{k} p_k * h[i+k] where i+k is in [0, N-1] and k in [0, n_conv-1]
    # term2[i] = sum_{k} p_k * h[i-k] where i-k is in [0, N-1] and k in [0, n_conv-1]
    
    # term1[i] = sum_j h[j] * p_{j-i}  (where j-i is valid index in p)
    # term2[i] = sum_j h[j] * p_{i-j}  (where i-j is valid index in p)
    
    # So term1 = convolve(h, p) and term2 = convolve(h, p_reversed) or something similar.
    # Actually, term1[i] = (h * p)[i] where * is cross-correlation or convolution?
    # Let's just compute it directly for clarity, or use FFT.
    
    # For N=600, direct computation is O(N^2) = 360000, which is fine.
    # But let's use FFT for speed.
    
    # term1[i] = sum_{k} p[k] * h[i+k]
    # This is like a cross-correlation. 
    # Let's just do it with numpy for simplicity since N is small enough.
    
    # Actually, let's just compute the gradient numerically for simplicity and correctness,
    # or use a simpler approach: just do gradient descent with finite differences for the 
    # surrogate, or use the fact that we can compute the gradient efficiently.
    
    # For now, let's just return the score and a zero gradient, and use a different optimization method.
    # Or better, let's use scipy.optimize.minimize with the smooth surrogate and no gradient,
    # using L-BFGS-B with numerical gradients.
    
    # Actually, let's just use the smooth surrogate as the objective and let scipy handle it.
    
    return score, None  # We'll use numerical gradients

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial best: uniform
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # Use smooth-max surrogate with gradient-based optimization
    def surrogate_objective(h):
        s = np.sum(h)
        if s < 1e-12:
            return 0.0
        n_conv = 2 * N - 1
        n_fft = 1
        while n_fft < n_conv:
            n_fft *= 2
        H = np.zeros(n_fft, dtype=np.complex128)
        H[:N] = h
        conv = np.real(np.fft.ifft(np.fft.fft(H) ** 2))[:n_conv]
        beta = 50.0
        conv_scaled = conv / beta
        max_cs = np.max(conv_scaled)
        exp_cs = np.exp(conv_scaled - max_cs)
        sum_exp_cs = np.sum(exp_cs)
        surrogate = beta * (max_cs + np.log(sum_exp_cs))
        return surrogate / (s * s)
    
    # Try different initializations and optimize
    initializations = [
        np.ones(N),
        # Two-block initialization
        np.zeros(N),
        # Cosine-like
        np.cos(np.linspace(0, np.pi, N)),
        # Random
        np.random.rand(N),
    ]
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        h0 = np.maximum(h0, 0)
        if np.sum(h0) < 1e-12:
            continue
        h0 = h0 / np.sum(h0) * N  # Normalize to have sum ~ N
        
        try:
            res = optimize.minimize(
                surrogate_objective, h0, method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
            )
            h_opt = np.maximum(res.x, 0)
            if np.sum(h_opt) > 1e-12:
                score = compute_score(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt
        except Exception:
            pass
    
    # Refine with coordinate descent on the actual max objective
    for _ in range(3):
        if time.time() > deadline:
            break
        conv = np.convolve(best_h, best_h, mode='full')
        k_peak = np.argmax(conv)
        peak_val = conv[k_peak]
        
        # Try reducing h[i] for indices that contribute to the peak
        for i in range(N):
            if time.time() > deadline:
                break
            if best_h[i] < 1e-10:
                continue
            # Try reducing h[i] by a small factor
            h_new = best_h.copy()
            h_new[i] *= 0.9
            score = compute_score(h_new)
            if score < best_score:
                best_score = score
                best_h = h_new
            # Also try setting to 0
            h_new2 = best_h.copy()
            h_new2[i] = 0
            score2 = compute_score(h_new2)
            if score2 < best_score:
                best_score = score2
                best_h = h_new2
    
    return best_h.tolist()
