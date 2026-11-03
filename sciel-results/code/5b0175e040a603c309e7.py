import numpy as np
import time
import scipy.optimize as opt

N = 600
TIME_LIMIT = 140

def fft_autoconv(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_autoconv(f)
    return np.max(conv) / (s * s)

def project_simplex(v, s=1.0):
    """Project v onto the simplex {x | x >= 0, sum(x) = s}."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def lse_objective_and_grad(f, beta):
    """
    Compute the Log-Sum-Exp approximation of max(autoconv(f))
    and its gradient.
    f is assumed to be on the simplex (sum=1, f>=0).
    """
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf, np.zeros_like(f)
    # Normalize f to sum=1 for the calculation, but we assume input is normalized
    # If not, we should normalize. The FISTA step projects to simplex, so f is normalized.
    
    conv = fft_autoconv(f)
    max_conv = np.max(conv)
    
    # LSE: max_conv + (1/beta) * log(sum(exp(beta * (conv - max_conv))))
    z = beta * (conv - max_conv)
    # Clip z to avoid underflow/overflow issues in exp, though beta*(conv-max) <= 0
    # exp(z) is in (0, 1]
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z)
    
    lse_val = max_conv + np.log(sum_exp_z) / beta
    
    # Gradient of LSE
    # p_k = exp(z_k) / sum(exp_z)
    p = exp_z / sum_exp_z
    
    # Gradient of conv_k wrt f
    # conv_k = sum_j f_j f_{j+k}
    # d conv_k / d f_i = f_{i+k} + f_{i-k}
    
    # We need g_i = sum_k p_k (f_{i+k} + f_{i-k})
    # This is a weighted sum of correlations.
    # g = correlation(p, f) + correlation(p, f_reversed)?
    # Let's use the loop method for clarity and correctness, as N=600 is small.
    # Or better, use FFT for the gradient too?
    # The operation is: g = (p * f) + (p * f) where * is some correlation.
    # Specifically, g_i = sum_k p_k f_{i+k} + sum_k p_k f_{i-k}.
    # Term 1: h1_i = sum_{k: i+k < N} p_k f_{i+k}. This is a cross-correlation.
    # Term 2: h2_i = sum_{k: i-k >= 0} p_k f_{i-k}.
    
    # We can compute h1 and h2 using np.correlate or FFT.
    # Let's use FFT for speed.
    # h1_i = (p * f)_i where * is cross correlation?
    # np.correlate(p, f, 'full') gives sum p_j f_{i+j}.
    # We need sum p_k f_{i+k}.
    # Let's stick to the O(N^2) loop for robustness, it's fast enough for N=600.
    # 600 * 600 = 360k ops. FISTA 1000 iters = 360M ops.
    # In NumPy, a Python loop is slow.
    # Let's vectorize the gradient computation.
    
    # g_i = sum_k p_k f_{i+k} + sum_k p_k f_{i-k}
    # Note p is length 2N-1. f is length N.
    
    # We can compute the first term using FFT.
    # Let P = FFT(p, size=2N-1). F = FFT(f, size=2N-1).
    # This is getting complicated.
    
    # Let's try a matrix-free approach or just the loop with N=600.
    # Actually, N=600 is small. A simple loop in Python might take ~10ms per iteration.
    # 1000 iterations = 10s. This is acceptable.
    
    g = np.zeros(N)
    for i in range(N):
        # p[0:N-i] corresponds to f[i:N]
        if N - i > 0:
            g[i] += np.dot(p[0:N-i], f[i:N])
        # p[0:i+1] corresponds to f[i::-1]
        if i >= 0:
            g[i] += np.dot(p[0:i+1], f[i::-1])
            
    return lse_val, g

def fista_solve(f_init, start_time, max_time=120.0, beta=20.0):
    """
    Solve min LSE(f*f, beta) subject to f in Simplex using FISTA.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    x = f.copy()
    y = f.copy()
    t = 1.0
    
    # Estimate Lipschitz constant L
    # We can start with a rough estimate and use backtracking if needed,
    # or just pick a large L.
    # The Hessian of the quadratic form is bounded by 2 * max(f) * something?
    # Let's start with L = 10.0 and adjust if it diverges.
    L = 10.0
    
    best_f = x.copy()
    best_val = compute_score(x)
    
    for k in range(2000):
        if time.time() - start_time > max_time:
            break
        
        obj, grad = lse_objective_and_grad(y, beta)
        
        # Backtracking line search for L
        # We want to ensure the quadratic approximation holds.
        # For simplicity, we can fix L or use a simple heuristic.
        # Let's try to estimate L from the gradient magnitude.
        # Or just use a fixed L that works for N=600.
        # The curvature of f*f is related to f.
        # max eigenvalue of Hessian of sum f_i f_j is bounded by 2 * max(f) * N?
        # Let's use L = 50.0 to be safe.
        
        z = y - (1.0 / L) * grad
        x_new = project_simplex(z, 1.0)
        
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        
        y_new = x_new + ((t - 1.0) / t_new) * (x_new - x)
        
        x = x_new
        t = t_new
        y = y_new
        
        # Check convergence or improvement
        val = compute_score(x)
        if val < best_val:
            best_val = val
            best_f = x.copy()
            
    return best_f

def construct():
    start_time = time.time()
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
    candidates = []
    
    # Uniform
    f = np.ones(N)
    candidates.append(f / np.sum(f))
    
    # Gaussian
    for alpha in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        f = np.exp(-alpha * x ** 2)
        candidates.append(f / np.sum(f))
        
    # Triangle
    for w in [0.1, 0.2, 0.25, 0.4, 0.5]:
        f = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f[mask] = 1.0 - 2.0 * np.abs(x[mask]) / w
        f = np.maximum(f, 0)
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))
            
    # Hann Window
    f = 0.5 * (1 - np.cos(2 * np.pi * (x + 0.25)))
    f = np.maximum(f, 0)
    if np.sum(f) > 1e-15:
        candidates.append(f / np.sum(f))
        
    # Blackman Window
    t = (x + 0.25) / 0.5
    t = np.clip(t, 0, 1)
    f = 0.42 - 0.5 * np.cos(2 * np.pi * t) + 0.08 * np.cos(4 * np.pi * t)
    f = np.maximum(f, 0)
    if np.sum(f) > 1e-15:
        candidates.append(f / np.sum(f))
        
    # Random candidates
    np.random.seed(42)
    for _ in range(5):
        f = np.random.rand(N)
        candidates.append(f / np.sum(f))
        
    # Score them
    scored = []
    for f in candidates:
        score = compute_score(f)
        scored.append((f, score))
    
    scored.sort(key=lambda x: x[1])
    best_f = scored[0][0].copy()
    best_score = scored[0][1]
    
    # Apply FISTA to the best initial candidates
    for f_init, score_init in scored[:5]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        # Try different betas
        for beta in [10.0, 20.0, 50.0]:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            refined = fista_solve(f_init, start_time, TIME_LIMIT - 2, beta=beta)
            refined_score = compute_score(refined)
            if refined_score < best_score:
                best_f = refined
                best_score = refined_score
                
    # Final normalization
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]
