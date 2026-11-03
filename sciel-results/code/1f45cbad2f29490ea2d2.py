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

def objective_and_gradient(g):
    """
    Objective function and gradient for log-heights g.
    f = exp(g)
    Minimize L = max_k C_k / (sum f)^2
    """
    f = np.exp(g)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10, np.zeros_like(g)
    
    conv = fft_autoconv(f)
    max_val = np.max(conv)
    L = max_val / (s * s)
    
    # Find indices where conv is max
    # Use a tolerance to handle numerical noise in ties
    max_idx = np.where(conv >= max_val - 1e-12)[0]
    
    # Gradient of L w.r.t f
    # dL/df_j = (1/s^2) * dM/df_j - (2M/s^3) * 1
    # dM/df_j is the subgradient of max(conv) at f
    # We approximate by averaging the gradients of the active lags
    grad_f = np.zeros(N)
    
    if len(max_idx) > 0:
        # For each active lag k, compute dC_k/df
        # dC_k/df_j = f[k-j] + f[j-k] (with bounds check)
        # We can vectorize this for all active k? 
        # Or just pick one if unique.
        # If multiple, average is a valid subgradient.
        
        # Efficient way:
        # dC_k/df is the cross-correlation of f with a delta at k?
        # Actually, C_k = sum_i f_i f_{k-i}.
        # dC_k/df_j = f_{k-j} + f_{j-k}.
        
        # Let's compute the gradient for all lags at once?
        # grad_C_k_j = f_{k-j} + f_{j-k}
        # This is essentially 2 * corr(f, f) shifted?
        # Note that conv = f * f.
        # The gradient of C_k with respect to f is the vector v where v_j = f_{k-j} + f_{j-k}.
        
        # We can compute the gradient for a specific k in O(N).
        # If there are few active lags, we can loop.
        # Usually the max is unique or few ties.
        
        active_lags = max_idx[:10] # Limit to 10 to avoid overhead if many ties due to noise
        
        for k in active_lags:
            grad_k = np.zeros(N)
            # j ranges 0..N-1
            # term 1: f[k-j] valid if 0 <= k-j < N => k-N+1 <= j <= k
            # term 2: f[j-k] valid if 0 <= j-k < N => k <= j < k+N
            
            # Vectorized index creation
            js = np.arange(N)
            
            # Term 1: f[k-j]
            idx1 = k - js
            mask1 = (idx1 >= 0) & (idx1 < N)
            grad_k[mask1] += f[idx1[mask1]]
            
            # Term 2: f[j-k]
            idx2 = js - k
            mask2 = (idx2 >= 0) & (idx2 < N)
            grad_k[mask2] += f[idx2[mask2]]
            
            grad_f += grad_k / len(active_lags)
            
    grad_L_f = (grad_f / (s * s)) - (2 * max_val / (s ** 3))
    
    # Chain rule: dL/dg_j = dL/df_j * f_j
    grad_L_g = grad_L_f * f
    
    return L, grad_L_g

def construct():
    start_time = time.time()
    
    # Initial candidates in log-space
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
    
    candidates_g = []
    
    # Gaussian in log-space: log(exp(-alpha x^2)) = -alpha x^2
    for alpha in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        g = -alpha * x ** 2
        # Shift to avoid underflow/overflow issues? 
        # L-BFGS-B works on g. f=exp(g).
        # If g is very negative, f is 0.
        # Let's normalize g so that max(g) is around 0 to keep f in a good range.
        g = g - np.max(g)
        candidates_g.append(g)
        
    # Uniform: g = 0
    candidates_g.append(np.zeros(N))
    
    # Triangle approx: log(1-|x|/w)
    for w in [0.1, 0.2, 0.25, 0.4, 0.5]:
        f_tri = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f_tri[mask] = 1.0 - 2.0 * np.abs(x[mask]) / w
        f_tri = np.maximum(f_tri, 1e-10)
        g = np.log(f_tri)
        g = g - np.max(g)
        candidates_g.append(g)
        
    best_f = None
    best_score = np.inf
    
    for i, g_init in enumerate(candidates_g):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        try:
            # Use L-BFGS-B
            res = opt.minimize(
                objective_and_gradient, 
                g_init, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            f_opt = np.exp(res.x)
            # Normalize f to sum 1 for score calculation consistency
            s = np.sum(f_opt)
            if s > 1e-15:
                f_opt = f_opt / s
            
            score = compute_score(f_opt)
            
            if score < best_score:
                best_score = score
                best_f = f_opt.copy()
                
        except Exception:
            continue
    
    # Fallback if L-BFGS-B fails for all
    if best_f is None:
        f = np.ones(N)
        best_f = f / np.sum(f)
        
    return [float(x) for x in best_f]
