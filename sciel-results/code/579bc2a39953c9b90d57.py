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

def spectral_projected_gradient(f_init, start_time, max_time=120.0):
    """
    Spectral approach: minimize max(autoconv(f)) subject to sum(f)=1, f>=0.
    We parameterize by the amplitude spectrum A_k >= 0 of the DFT of f.
    The autoconvolution is IFFT(|FFT(f)|^2) = IFFT(FFT(f) * conj(FFT(f))).
    If we fix the phases and only adjust amplitudes, or vice versa, we can iterate.
    However, a simpler and robust spectral method is to work in the frequency domain 
    using the fact that the peak of the autoconvolution is related to the L1 norm 
    of the power spectrum? No, peak is IFFT(PowerSpectrum).
    
    Let's use a "soft" spectral update:
    1. Compute FFT of f.
    2. Update the phase/amplitude to reduce the peak.
    Actually, the current best approach is time-domain coordinate descent which is very effective.
    The prompt asks for a structural change involving spectral representation.
    
    Let's try a different spectral strategy:
    Minimize the peak of the autoconvolution by adjusting the signal in the frequency domain.
    We can use the fact that the gradient of the peak w.r.t. f_i can be computed.
    But we want to use the spectral domain.
    
    Here is a known technique for minimizing the peak-to-average power ratio (PAPR) 
    which is similar: iterative projection.
    1. Compute FFT of f.
    2. For each frequency bin, adjust the magnitude to reduce the time-domain peak.
    
    Alternatively, we can use the "spectral coordinate descent" idea from the inspiration:
    Minimize the objective with respect to the frequency coefficients.
    
    Let's stick to a robust hybrid: 
    Use the time-domain coordinate descent (which is known to be strong) but enhance it 
    with a spectral initialization or a spectral polishing step.
    
    Actually, the inspiration 1 (score 1.1290) was very close to the current (1.1312).
    The current program uses `exact_coordinate_descent`.
    
    Let's implement a spectral method based on the following:
    The objective is max_k (f*f)_k.
    This is equivalent to finding the largest eigenvalue of a certain matrix? No.
    
    Let's try a "spectral descent" where we update the DFT coefficients.
    Let F_k = FFT(f)_k.
    The autoconvolution is IFFT(F_k * conj(F_k)).
    We want to minimize the max of this IFFT.
    
    A simple spectral update:
    1. Compute f.
    2. Compute conv = fft_autoconv(f).
    3. Identify the peak lag m*.
    4. The gradient of conv[m*] w.r.t. f is:
       d/d f_i conv[m*] = f_{i+m*} + f_{i-m*} (with bounds).
    5. This is a correlation.
    
    Let's combine the best of both:
    1. Use the `exact_coordinate_descent` from the current program as the primary engine, as it is proven to work.
    2. Add a spectral-based initialization or a different initial candidate generator.
    3. Add a "spectral polishing" step that uses the frequency domain to make small adjustments.
    
    However, the directive says: "Make a STRUCTURAL change... a different optimization or search method".
    The current program uses time-domain coordinate descent.
    The inspiration suggests spectral representation.
    
    Let's implement a **Spectral Projected Gradient Descent**:
    1. Work with the DFT coefficients F.
    2. The constraint is f >= 0 and sum(f)=1.
    3. The objective is max(autoconv(f)).
    
    This is difficult because the constraint f>=0 is non-trivial in the frequency domain.
    
    Alternative Structural Change: **Alternating Minimization in Frequency and Time**
    1. Fix the frequency spectrum magnitudes, optimize phases to minimize peak? (Hard)
    2. Fix the time-domain signal, optimize something else?
    
    Let's look at the "First autocorrelation inequality" context.
    This is related to the "minimum peak autocorrelation" problem.
    
    A powerful method for this is the **Weinberger algorithm** or similar iterative spectral methods.
    But for non-negative signals, it's more constrained.
    
    Let's try a different structural approach: **Block Coordinate Descent**.
    Instead of updating one bin at a time (which is slow), update blocks of bins.
    Or, use **Quasi-Newton** methods?
    
    Let's stick to the most promising structural change: **Using the FFT to accelerate the gradient computation for a projected gradient descent method with a better line search**.
    The current `exact_coordinate_descent` is O(N^2) per coordinate update (due to the inner loop or dot product).
    With N=600, it's okay, but maybe we can do better.
    
    Actually, the `exact_coordinate_descent` in the current program is quite strong.
    The score is 1.1312.
    
    Let's try to improve the initial candidates.
    The current candidates are Uniform, Gaussian, Triangle.
    We can add **Cosine-bell** (Hann window) and **Blackman** windows.
    
    Also, we can try **Random Initializations** with random seeds to escape local minima.
    
    Let's combine:
    1. Better initial candidates (Hann, Blackman, random).
    2. The `exact_coordinate_descent` refinement.
    3. A new spectral polishing step: 
       - Compute FFT of f.
       - Apply a soft-thresholding or phase adjustment?
       - Actually, let's use the "spectral coordinate descent" idea from the inspiration but implemented correctly.
       
    Here is a specific spectral update:
    The gradient of the peak autoconvolution at lag m is g_i = f_{i+m} + f_{i-m}.
    We can compute this gradient for all i efficiently using FFT?
    g = IFFT(FFT(f) * conj(FFT(f_shifted)))?
    
    Let's implement a **Projected Gradient Descent** with **FFT-accelerated gradients** and **Adaptive Step Size**.
    This is structurally different from coordinate descent.
    
    Algorithm:
    1. Initialize f.
    2. Loop:
       a. Compute conv = fft_autoconv(f).
       b. Find peak lag m.
       c. Compute gradient g = d conv[m] / d f.
          g_i = f_{i+m} + f_{i-m} (with 0 padding).
          This can be computed as:
          g = np.roll(f, m) + np.roll(f, -m)
          But we need to handle boundaries correctly (linear convolution, not circular).
          For linear convolution, the gradient is a correlation.
          g = np.correlate(f, f, mode='full')[m] ... no.
          
          Let's derive:
          conv[m] = sum_j f_j f_{j+m}
          d/d f_i conv[m] = f_{i+m} (if i+m < N) + f_{i-m} (if i-m >= 0)
          
          We can compute this vector g of length N:
          g = np.zeros(N)
          g[:N-m] += f[m:]
          g[m:] += f[:N-m]
          
       d. Projected gradient step:
          f_new = f - lr * g
          f_new = project_simplex(f_new, 1.0)
       e. Check if improvement, else reduce lr.
    
    This is O(N) per step (plus FFT for conv).
    Coordinate descent is O(N^2) per sweep (N coordinates * O(N) each).
    Gradient descent might be faster per iteration, but might take more iterations.
    
    Let's implement this and compare with coordinate descent.
    
    We will run both and keep the best.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(f)
    
    for it in range(2000):
        if time.time() - start_time > max_time:
            break
        
        conv = fft_autoconv(f)
        m = np.argmax(conv)
        
        # Gradient of conv[m] w.r.t. f
        g = np.zeros(N)
        if m == 0:
            g = 2 * f
        else:
            g[:N-m] += f[m:]
            g[m:] += f[:N-m]
        
        # Normalize gradient to prevent large steps
        g_norm = np.linalg.norm(g)
        if g_norm > 1e-10:
            g = g / g_norm
        
        # Momentum step
        v = momentum * v - lr * g
        f_new = f + v
        
        # Project to simplex
        f_new = project_simplex(f_new, 1.0)
        
        val = compute_score(f_new)
        if val < best_val:
            best_val = val
            best_f = f_new.copy()
            f = f_new
            # Increase lr slightly if we made a big improvement?
            # lr *= 1.05
        else:
            # Reduce lr
            lr *= 0.5
            v = 0.0  # Reset momentum
    
    return best_f

def exact_coordinate_descent(f_init, start_time, max_time=120.0):
    """
    Coordinate descent where each coordinate is optimized exactly
    by minimizing the normalized max autoconvolution over that coordinate.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s  # Normalize
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    for sweep in range(100):
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
            
            if v_opt != old_val:
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
    # Blackman: a0 - a1*cos(2*pi*t) + a2*cos(4*pi*t)
    # t in [0, 1]
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
    
    # Apply exact coordinate descent to the best initial candidates
    for f_init, score_init in scored[:5]:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        refined = exact_coordinate_descent(f_init, start_time, TIME_LIMIT - 2)
        refined_score = compute_score(refined)
        if refined_score < best_score:
            best_f = refined
            best_score = refined_score
            
    # Try spectral projected gradient on the best result
    if time.time() - start_time < TIME_LIMIT - 10:
        spectral_refined = spectral_projected_gradient(best_f, start_time, TIME_LIMIT - 2)
        spectral_score = compute_score(spectral_refined)
        if spectral_score < best_score:
            best_f = spectral_refined
            best_score = spectral_score
            
    # Final normalization
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]
