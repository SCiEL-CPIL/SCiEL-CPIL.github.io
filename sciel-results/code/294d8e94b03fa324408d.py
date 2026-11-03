import numpy as np
import time
import random

N = 600
TIME_LIMIT = 115  # Safety margin

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for fast convolution
    conv = np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)
    return np.max(conv) / (s * s)

def project_simplex(v, s=1.0):
    """Project vector v onto the simplex sum(v)=s, v>=0."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if np.sum(v) <= 0:
        return np.ones(n) * (s / n)
    
    # Algorithm from "Euclidean Projection onto a Simplex"
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    r = np.arange(1, n + 1)
    cond = u - (cssv - 1) / r > 0
    rho = np.where(cond)[0][-1] if np.any(cond) else 0
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    if np.sum(w) < 1e-15:
        return np.ones(n) * (s / n)
    return w * (s / np.sum(w))

def gradient_step(f, step_size):
    """Perform one gradient descent step on M(f) subject to sum(f)=1, f>=0."""
    # Compute convolution
    conv = np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)
    max_val = np.max(conv)
    # Find indices of max (or near max for stability)
    threshold = max_val * 0.99
    max_indices = np.where(conv >= threshold)[0]
    
    if len(max_indices) == 0:
        return f
        
    grad = np.zeros(N)
    for k in max_indices:
        # d C_k / d f_j = 2 * f[k-j] if 0 <= k-j < N
        start_j = max(0, k - (N - 1))
        end_j = min(N, k + 1)
        if start_j < end_j:
            indices = k - np.arange(start_j, end_j)
            grad[start_j:end_j] += 2 * f[indices]
            
    # Normalize gradient
    grad /= len(max_indices)
    
    # Update
    f_new = f - step_size * grad
    # Project to simplex
    f_new = project_simplex(f_new, 1.0)
    return f_new

def optimize_from_start(f_start, max_iters=5000, time_limit=20):
    """Optimize starting from f_start."""
    f = f_start.copy()
    f = project_simplex(f, 1.0)
    current_score = compute_score(f)
    step_size = 0.1
    start_time = time.time()
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        f_new = gradient_step(f, step_size)
        new_score = compute_score(f_new)
        
        if new_score < current_score:
            f = f_new
            current_score = new_score
            step_size *= 1.05
            if step_size > 0.5: step_size = 0.5
        else:
            step_size *= 0.9
            if step_size < 1e-4:
                break
                
    return f, current_score

def construct():
    start_time = time.time()
    
    # Define diverse initial shapes
    starts = []
    
    # 1. Uniform
    starts.append(np.ones(N) / N)
    
    # 2. Triangular
    x = np.linspace(0, 1, N)
    tri = 1 - np.abs(2 * x - 1)
    starts.append(tri / np.sum(tri))
    
    # 3. Cosine taper
    cos = 0.5 * (1 + np.cos(2 * np.pi * x))
    starts.append(cos / np.sum(cos))
    
    # 4. Plateau (middle 50%)
    plat = np.zeros(N)
    plat[N//4 : 3*N//4] = 1.0
    starts.append(plat / np.sum(plat))
    
    # 5. Gaussian-like
    g = np.exp(-((x - 0.5)**2) / (2 * 0.1**2))
    starts.append(g / np.sum(g))
    
    # 6. Random smooth
    for _ in range(5):
        r = np.random.rand(N)
        r = np.convolve(r, np.ones(10)/10, mode='same')
        starts.append(r / np.sum(r))
        
    best_f = None
    best_score = np.inf
    
    for i, start in enumerate(starts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        f_opt, score = optimize_from_start(start, max_iters=2000, time_limit=15)
        if score < best_score:
            best_score = score
            best_f = f_opt
            
    # Final polish with smaller steps on the best found
    if best_f is not None and time.time() - start_time < TIME_LIMIT - 5:
        f_polished, score_polished = optimize_from_start(best_f, max_iters=5000, time_limit=TIME_LIMIT - 5 - (time.time() - start_time))
        if score_polished < best_score:
            best_f = f_polished
            best_score = score_polished

    # Ensure nonnegative and return
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
