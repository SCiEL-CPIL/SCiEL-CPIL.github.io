"""Spectral-design + FFT-accelerated minimax for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _grad_minimax_fft(x, q, ac):
    """
    Compute gradient of E_q[|ac|]/(sum x)^2 w.r.t. x using FFT.
    Returns (grad, expected_abs, s, s_sq).
    """
    L = 2 * N - 1
    s = np.sum(x)
    if abs(s) < 1e-10:
        s = 1e-10
    s_sq = s * s
    
    abs_ac = np.abs(ac)
    w = q * np.sign(ac)  # length L, w[k] = q[k]*sign(ac[k])
    
    # grad_x[j] = sum_k w[k] * (x[j+k]*I(j+k<N) + x[j-k]*I(j-k>=0))
    # This equals: (w * x)[j] + (w_rev * x)[j] restricted to [0,N)
    # where * denotes cross-correlation.
    # 
    # More precisely:
    # Part A: sum_{k=0}^{N-1-j} w[k] * x[j+k]  -- this is like a correlation
    # Part B: sum_{k=0}^{j} w[k] * x[j-k]       -- this is another correlation
    #
    # Using FFT: Let's compute via the fact that
    # sum_k w[k] * x[j+k] for valid j+k is a valid cross-correlation.
    #
    # Approach: pad both to length >= 2N, use FFT.
    # Define: f[k] = w[k] for k in [0, L), 0 otherwise. Length L.
    # Define: g[k] = x[k] for k in [0, N), 0 otherwise. Length N.
    # 
    # Part A[j] = sum_{k} f[k] * g[j+k] = (f * g)[j] in correlation sense
    #   = sum_k f[k] g[j+k]
    #   This is the cross-correlation (f ⋆ g)[j]
    #
    # Part B[j] = sum_{k} f[k] * g[j-k] = (f * g)[j] in convolution sense
    #   = sum_k f[k] g[j-k]
    #   This is the convolution (f * g)[j]
    #
    # So grad_x_raw[j] = (f ⋆ g)[j] + (f * g)[j] for j in [0, N)
    
    # Compute via FFT. Use length >= N + L - 1 for convolution.
    L_fft = 1
    while L_fft < N + L:
        L_fft *= 2
    
    F = np.fft.rfft(w, n=L_fft)
    G = np.fft.rfft(x, n=L_fft)
    
    # Convolution: (f * g)[j] = IFFT(F * G)[j]
    conv = np.fft.irfft(F * G, n=L_fft)[:N]
    
    # Cross-correlation: (f ⋆ g)[j] = sum_k f[k] g[j+k]
    # = sum_k f[k] g[j+k] = IFFT(F * conj(G))[j] ... let me verify.
    # Cross-correlation R_fg[j] = sum_k f[k] g[k+j]
    # In FFT: R_fg = IFFT(conj(FFT(f)) * FFT(g)) ... 
    # Actually: R_fg[j] = sum_k f[k] g[k+j]
    # Let's define h[k] = f[-k] (reversed). Then R_fg[j] = sum_k h[-k] g[k+j] ... 
    # Simpler: R_fg[j] = sum_k f[k] g[k+j] = (f * g_flipped)[j + N - 1] where g_flipped[k] = g[N-1-k]
    # Or: R_fg[j] = IFFT(conj(F) * G)[j]
    
    # Let me verify with small example mentally:
    # f = [a, b], g = [c, d]
    # R_fg[0] = f[0]*g[0] + f[1]*g[1] = a*c + b*d
    # conj(F)*G: F = FFT([a,b]), G = FFT([c,d])
    # IFFT(conj(F)*G) = [a*c+b*d, a*d+b*c] (for length 2, circular)
    # So R_fg[0] = IFFT(conj(F)*G)[0]. Good.
    
    corr = np.fft.irfft(np.conj(F) * G, n=L_fft)[:N]
    
    grad_x_raw = corr + conv
    
    # Account for 1/s_sq: d(E_q|ac| / s_sq)/dx[j] = grad_x_raw[j]/s_sq - 2*E_q|ac|*x[j]/s^3
    expected_abs = np.sum(q * abs_ac)
    grad_x = grad_x_raw / s_sq - 2.0 * expected_abs * x / (s_sq * s)
    
    return grad_x, expected_abs, s, s_sq

def _minimax_optimize(x_init, max_time=100):
    """
    Run the minimax game from initial x_init with FFT-accelerated gradients.
    """
    L = 2 * N - 1
    q = np.ones(L) / L
    lr_x = 0.005
    
    x = x_init.copy()
    best_x = x.copy()
    best_val = _objective(x)
    
    start = time.time()
    
    while time.time() - start < max_time:
        if _check_deadline():
            break
        
        # Run several iterations
        for it in range(30):
            if _check_deadline():
                break
            
            ac = _autoconv_fft(x)
            abs_ac = np.abs(ac)
            
            # Adversary update
            tau = 5.0
            log_q = abs_ac / tau
            log_q -= np.max(log_q)
            q_new = np.exp(log_q)
            q_new /= np.sum(q_new)
            q = 0.7 * q + 0.3 * q_new
            
            # Designer update
            grad_x, _, _, _ = _grad_minimax_fft(x, q, ac)
            x = x - lr_x * grad_x
        
        val = _objective(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Adaptive LR: reduce slightly each round to help convergence
        lr_x *= 0.95
    
    return best_x, best_val

def _spectral_design(seed=0, beta=50.0):
    """
    Design x via PSD optimization: minimize soft-max(|autoconv|) / (sum x)^2
    by optimizing the log-power-spectrum.
    """
    half = N // 2
    rng = np.random.RandomState(seed)
    
    # Start with flat PSD (white noise)
    params = np.zeros(half + 1)
    
    best_params = params.copy()
    best_val = 1e10
    
    # Simple gradient descent on log-PSD
    lr = 0.01
    for it in range(200):
        if _check_deadline():
            break
        
        P_half = np.exp(params)
        P_full = np.zeros(N)
        P_full[:half+1] = P_half
        P_full[half+1:] = P_half[:half][::-1]
        
        # c = IFFT(P_full) gives autoconvolution (circular)
        c = np.fft.irfft(P_full, n=N)
        
        # Use circular autoconv as proxy (close to linear for smooth PSDs)
        abs_c = np.abs(c)
        s = np.sum(c)  # sum of autoconv = (sum x)^2 for circular
        
        if abs(s) < 1e-10:
            continue
        
        # Soft-max surrogate for max|c|
        m = np.max(abs_c)
        exp_c = np.exp((abs_c - m) / beta)
        softmax_val = m + np.log(np.sum(exp_c)) / beta
        
        obj = softmax_val / (s * s)
        
        if obj < best_val:
            best_val = obj
            best_params = params.copy()
        
        # Gradient w.r.t. params: d(obj)/d(params[k]) = d(obj)/d(P_half[k]) * P_half[k]
        # d(softmax_val)/d(P_half[k]) = (1/beta) * exp_c[k] / sum(exp_c) * (1/s^2) * s^2 + ... 
        # This gets complex. Use finite differences or a simpler approach.
        
        # Simpler: use Nelder-Mead or just random perturbations
        # Let's try coordinate ascent on params
        improved = False
        for k in range(half + 1):
            for delta in [0.1, -0.1, 0.05, -0.05]:
                trial = params.copy()
                trial[k] += delta
                P_half_t = np.exp(trial)
                P_full_t = np.zeros(N)
                P_full_t[:half+1] = P_half_t
                P_full_t[half+1:] = P_half_t[:half][::-1]
                c_t = np.fft.irfft(P_full_t, n=N)
                abs_c_t = np.abs(c_t)
                s_t = np.sum(c_t)
                if abs(s_t) < 1e-10:
                    continue
                m_t = np.max(abs_c_t)
                exp_c_t = np.exp((abs_c_t - m_t) / beta)
                sm_t = m_t + np.log(np.sum(exp_c_t)) / beta
                obj_t = sm_t / (s_t * s_t)
                if obj_t < obj:
                    params = trial
                    obj = obj_t
                    improved = True
                    break
        if not improved:
            break
    
    # Recover x from best PSD: x = IFFT(sqrt(P_full))
    P_half_best = np.exp(best_params)
    P_full_best = np.zeros(N)
    P_full_best[:half+1] = P_half_best
    P_full_best[half+1:] = P_half_best[:half][::-1]
    
    # x such that |FFT(x)|^2 = P_full, so x = IFFT(sqrt(P_full) * phase)
    # For simplicity, take x = IFFT(sqrt(P_full)) (real, zero-phase)
    x = np.fft.irfft(np.sqrt(np.maximum(P_full_best, 0)), n=N)
    
    return x

def _coordinate_descent_polish(x, max_iters=150):
    """Fast coordinate descent to polish the solution."""
    best_x = x.copy()
    best_val = _objective(best_x)
    
    deltas = [0.05, -0.05, 0.1, -0.1, 0.02, -0.02]
    
    for it in range(max_iters):
        if _check_deadline():
            break
        improved = False
        for j in range(N):
            for d in deltas:
                trial = best_x.copy()
                trial[j] += d
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Generate diverse initializations including spectral designs
    inits = []
    
    # Spectral designs with different seeds
    for seed in [0, 1, 2]:
        if _check_deadline():
            break
        try:
            x_spec = _spectral_design(seed=seed, beta=50.0)
            inits.append(x_spec)
        except Exception:
            pass
    
    # Standard initializations
    inits.append(np.ones(N))
    inits.append(np.random.RandomState(42).randn(N))
    inits.append(np.sin(2 * np.pi * np.arange(N) / 50.0))
    inits.append(np.sin(2 * np.pi * np.arange(N) / 20.0))
    inits.append(np.random.RandomState(7).rand(N) * 2 - 1)
    
    # Allocate time budget among restarts
    time_per_restart = 12
    
    for x0 in inits:
        if _check_deadline():
            break
        
        try:
            x_opt, val_opt = _minimax_optimize(x0, max_time=time_per_restart)
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Final polish with coordinate descent
    if not _check_deadline():
        best_x, best_val = _coordinate_descent_polish(best_x, max_iters=100)
    
    return best_x
# EVOLVE-BLOCK-END
