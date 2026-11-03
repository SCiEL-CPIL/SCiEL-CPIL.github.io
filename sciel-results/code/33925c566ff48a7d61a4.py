import numpy as np
import time
import scipy.optimize as opt

N = 600
TIME_LIMIT = 135

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

def smooth_max(x, k=50.0):
    """Smooth approximation of max(x) using log-sum-exp."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(k * (x - x_max)))) / k

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

def spectral_optimize(f_init, start_time, max_time=120.0):
    """
    Spectral optimization using a lifted Gram matrix representation.
    We represent f via its DFT coefficients. The autoconvolution peak is related
    to the maximum eigenvalue of the Gram matrix G = F F^H (where F is the DFT of f).
    
    Strategy:
    1. Work in the frequency domain with F = FFT(f).
    2. The constraint f >= 0 becomes IFFT(F) >= 0.
    3. The objective is to minimize the peak of IFFT(|F|^2).
    
    We use a projected gradient method in the frequency domain, where the projection
    enforces the non-negativity constraint in the time domain.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    # Frequency domain representation
    F = np.fft.rfft(f, n=2*N)
    
    lr = 0.05
    momentum = 0.85
    v = np.zeros_like(F, dtype=np.complex128)
    
    for it in range(3000):
        if time.time() - start_time > max_time:
            break
        
        # Compute autoconvolution in time domain
        conv = fft_autoconv(f)
        m = np.argmax(conv)
        
        # Gradient of the peak autoconvolution with respect to f
        g_f = np.zeros(N)
        if m == 0:
            g_f = 2 * f
        else:
            g_f[:N-m] += f[m:]
            g_f[m:] += f[:N-m]
        
        # Transform gradient to frequency domain
        g_F = np.fft.rfft(g_f, n=2*N)
        
        # Update in frequency domain
        v = momentum * v - lr * g_F
        F_new = F + v
        
        # Transform back to time domain
        f_new = np.fft.irfft(F_new, n=2*N)[:N]
        
        # Project to simplex (non-negative, sum=1)
        f_new = project_simplex(f_new, 1.0)
        
        # Update F
        F = np.fft.rfft(f_new, n=2*N)
        
        val = compute_score(f_new)
        if val < best_val - 1e-12:
            best_val = val
            best_f = f_new.copy()
            f = f_new
        else:
            lr *= 0.7
            v = 0.0
    
    return best_f

def block_coordinate_descent(f_init, start_time, max_time=120.0):
    """
    Block coordinate descent: optimize groups of coordinates simultaneously.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    block_size = 8
    n_blocks = N // block_size
    
    for sweep in range(30):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for b in range(n_blocks):
            if time.time() - start_time > max_time:
                break
            
            start_idx = b * block_size
            end_idx = start_idx + block_size
            block_indices = np.arange(start_idx, end_idx)
            
            def block_objective(v_block):
                v_block = np.maximum(v_block, 0)
                f_new = f.copy()
                f_new[block_indices] = v_block
                s_new = np.sum(f_new)
                if s_new < 1e-15:
                    return np.inf
                conv = fft_autoconv(f_new)
                return np.max(conv) / (s_new * s_new)
            
            v_init = f[block_indices]
            try:
                res = opt.minimize(block_objective, v_init, method='Nelder-Mead', 
                                   options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
                v_opt = res.x
            except Exception:
                v_opt = v_init
            
            if np.any(np.abs(v_opt - v_init) > 1e-10):
                f[block_indices] = np.maximum(v_opt, 0)
                improved = True
        
        s = np.sum(f)
        if s > 1e-15:
            f = f / s
        
        current_val = compute_score(f)
        if current_val < best_val - 1e-12:
            best_val = current_val
            best_f = f.copy()
        
        if not improved:
            break
            
    return best_f

def exact_coordinate_descent(f_init, start_time, max_time=120.0):
    """
    Coordinate descent where each coordinate is optimized exactly.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    for sweep in range(80):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > max_time:
                break
            
            old_val = f[i]
            S = np.sum(f)
            
            conv_current = fft_autoconv(f)
            
            def objective(v):
                if v < 0:
                    return np.inf
                S_new = S - old_val + v
                if S_new < 1e-15:
                    return np.inf
                m_start = max(0, i)
                m_end = min(2 * N - 2, i + N - 1)
                
                conv_vals = conv_current[m_start:m_end + 1].copy()
                
                js = np.arange(m_start, m_end + 1) - i
                
                old_contribs = np.zeros(len(js))
                mask_diag = (js == i)
                mask_offdiag = ~mask_diag
                old_contribs[mask_diag] = old_val * old_val
                old_contribs[mask_offdiag] = 2 * old_val * f[js[mask_offdiag]]
                
                conv_bg = conv_vals - old_contribs
                
                new_contribs = np.zeros(len(js))
                new_contribs[mask_diag] = v * v
                new_contribs[mask_offdiag] = 2 * v * f[js[mask_offdiag]]
                
                new_conv_vals = conv_bg + new_contribs
                
                max_conv = np.max(new_conv_vals)
                return max_conv / (S_new * S_new)
            
            v_upper = max(1.0, 2 * old_val + 0.1)
            
            try:
                res = opt.minimize_scalar(objective, bounds=(0, v_upper), method='bounded')
                v_opt = res.x
            except Exception:
                v_opt = old_val
            
            if abs(v_opt - old_val) > 1e-10:
                f[i] = v_opt
                improved = True
        
        s = np.sum(f)
        if s > 1e-15:
            f = f / s
        
        current_val = compute_score(f)
        if current_val < best_val - 1e-12:
            best_val = current_val
            best_f = f.copy()
        
        if not improved:
            break
            
    return best_f

def construct():
    start_time = time.time()
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
    candidates = []
    
    # Uniform
    f = np.ones(N)
    candidates.append(f / np.sum(f))
    
    # Gaussian
    for alpha in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
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
        
    # Raised cosine variants
    for beta in [0.5, 1.0, 2.0, 3.0]:
        f = np.cos(np.pi * np.clip(x / beta, -1, 1))
        f = np.maximum(f, 0)
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))
        
    # Random candidates
    np.random.seed(42)
    for _ in range(8):
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
    
    # Apply exact coordinate descent to the best initial candidates
    for f_init, score_init in scored[:4]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        refined = exact_coordinate_descent(f_init, start_time, TIME_LIMIT - 5)
        refined_score = compute_score(refined)
        if refined_score < best_score:
            best_f = refined
            best_score = refined_score
            
    # Try block coordinate descent on the best result
    if time.time() - start_time < TIME_LIMIT - 15:
        block_refined = block_coordinate_descent(best_f, start_time, TIME_LIMIT - 5)
        block_score = compute_score(block_refined)
        if block_score < best_score:
            best_f = block_refined
            best_score = block_score
            
    # Try spectral optimization on the best result
    if time.time() - start_time < TIME_LIMIT - 15:
        spectral_refined = spectral_optimize(best_f, start_time, TIME_LIMIT - 5)
        spectral_score = compute_score(spectral_refined)
        if spectral_score < best_score:
            best_f = spectral_refined
            best_score = spectral_score
            
    # Final coordinate descent polish
    if time.time() - start_time < TIME_LIMIT - 5:
        final_refined = exact_coordinate_descent(best_f, start_time, TIME_LIMIT - 2)
        final_score = compute_score(final_refined)
        if final_score < best_score:
            best_f = final_refined
            best_score = final_score
            
    # Final normalization
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]
