"""Model-and-optimize: Alternating minimization with spectral amplitude constraint for autoconvolution minimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 138  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute the raw objective: max|ac| / (sum(x))^2."""
    ac = _autoconv_fft(x)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient using pure FFT.
    Returns (soft_max_value, gradient_array)
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    g = weights * signs  # length 2N-1, this is d(soft_max)/d(ac_k)
    
    # Gradient w.r.t. x: grad_x[i] = sum_k g[k] * d(ac[k])/dx[i]
    # d(ac[k])/dx[i] = x[k-i] + x[i-k] (for valid indices)
    # grad_x[i] = sum_k g[k] * (x[k-i] + x[i-k])
    # = A[i] + B[i] where A[i] = sum_m x[m]*g[m+i] and B[i] = sum_m x[m]*g[i-m]
    
    L_fft2 = 1
    while L_fft2 < 2 * (2*N - 1):
        L_fft2 *= 2
    
    X_f2 = np.fft.rfft(x, n=L_fft2)
    G_f2 = np.fft.rfft(g, n=L_fft2)
    
    # A[i] = sum_m x[m]*g[m+i] = cross-correlation
    A = np.fft.irfft(X_f2 * np.conj(G_f2), n=L_fft2)[:N]
    # B[i] = sum_m x[m]*g[i-m] = convolution
    B = np.fft.irfft(X_f2 * G_f2, n=L_fft2)[:N]
    
    grad_x = A + B
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_projection(x, target_amplitude=None):
    """
    Project x onto the set of sequences with a given spectral amplitude profile.
    Uses iterative soft-thresholding in frequency domain to enforce amplitude constraint.
    """
    if target_amplitude is None:
        # Default: flat-top (rectangular) window in frequency
        L_fft = 1
        while L_fft < 2 * N:
            L_fft *= 2
        k = np.arange(L_fft // 2 + 1)
        # Use a raised-cosine window for smoother transition
        t = k / (L_fft // 2)
        target_amplitude = np.ones(L_fft // 2 + 1)
        # Apply a gentle taper to avoid sharp edges
        taper = 0.5 * (1 + np.cos(np.pi * t))
        target_amplitude *= taper
    
    # Iterative projection: alternate between time-domain (identity) and freq-domain amplitude
    x_proj = x.copy()
    for _ in range(20):
        if _check_deadline():
            break
        # FFT
        L_fft = 1
        while L_fft < 2 * N:
            L_fft *= 2
        X = np.fft.rfft(x_proj, n=L_fft)
        
        # Enforce amplitude constraint: scale each frequency bin to target amplitude
        mag = np.abs(X)
        mag[mag < 1e-10] = 1e-10
        phase = X / mag
        X_new = target_amplitude * phase
        x_proj = np.fft.irfft(X_new, n=L_fft)[:N]
    
    return x_proj

def _alternating_minimize(x_init, beta, n_iters=50, lr=0.01, proj_strength=0.3):
    """
    Alternating minimization:
    1. Gradient descent step on soft-max objective
    2. Soft projection step toward spectral amplitude constraint
    """
    x = x_init.copy()
    best_x = x.copy()
    best_val = _objective_only(x)
    
    # Compute target amplitude: use a smoothed version of the initial spectrum
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    X_init = np.fft.rfft(x_init, n=L_fft)
    target_amp = np.abs(X_init)
    # Smooth the target amplitude
    from scipy.ndimage import uniform_filter1d
    target_amp = uniform_filter1d(target_amp, size=10)
    target_amp = np.maximum(target_amp, 1e-5)
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        # Step 1: Gradient descent
        soft_val, grad_x = _soft_max_and_grad(x, beta)
        s = np.sum(x)
        if abs(s) < 1e-10:
            s = 1e-10
        grad_obj = grad_x / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
        x = x - lr * grad_obj
        
        # Step 2: Soft projection toward spectral constraint
        x_proj = _spectral_projection(x, target_amp)
        x = (1 - proj_strength) * x + proj_strength * x_proj
        
        # Track best
        current_val = _objective_only(x)
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
        
        # Adaptive learning rate
        if it % 20 == 0 and it > 0:
            lr *= 0.95
            proj_strength = min(0.5, proj_strength * 1.05)
    
    return best_x, best_val

def _phase_retrieval(x_init, n_iters=100):
    """
    Spectral phase retrieval: fix amplitude, optimize phase to minimize peak autoconvolution.
    """
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    
    # Get initial amplitude
    X = np.fft.rfft(x_init, n=L_fft)
    target_amp = np.abs(X)
    target_amp = np.maximum(target_amp, 1e-5)
    
    x = x_init.copy()
    best_x = x.copy()
    best_val = _objective_only(x)
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        # Compute current spectrum
        X_cur = np.fft.rfft(x, n=L_fft)
        phase_cur = X_cur / np.maximum(np.abs(X_cur), 1e-10)
        
        # Enforce target amplitude
        X_new = target_amp * phase_cur
        x = np.fft.irfft(X_new, n=L_fft)[:N]
        
        # Gradient step to reduce peak autoconvolution
        soft_val, grad_x = _soft_max_and_grad(x, beta=50.0)
        s = np.sum(x)
        if abs(s) < 1e-10:
            s = 1e-10
        grad_obj = grad_x / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
        x = x - 0.01 * grad_obj
        
        # Track best
        current_val = _objective_only(x)
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
    
    return best_x, best_val

def _spectral_init(rng):
    """
    Generate initial points via spectral amplitude shaping.
    """
    L_fft = 1
    while L_fft < 2*N:
        L_fft *= 2
    
    k = np.arange(L_fft // 2 + 1)
    t = k / (L_fft // 2)
    
    windows = [
        np.ones(L_fft // 2 + 1),
        np.exp(-0.5 * (t - 0.5)**2 / (0.15**2)),
        np.exp(-0.5 * (t - 0.5)**2 / (0.25**2)),
        1.0 / (1.0 + (t / 0.3)**2),
        np.sinc(0.5 * t) ** 2,
        np.exp(-t),
        1.0 / (1.0 + t),
    ]
    
    inits = []
    for w in windows:
        w = np.maximum(w, 0)
        phase = rng.uniform(0, 2*np.pi, size=L_fft // 2 + 1)
        spec = w * np.exp(1j * phase)
        spec_full = np.zeros(L_fft, dtype=complex)
        spec_full[:L_fft//2+1] = spec
        spec_full[L_fft//2+1:] = np.conj(spec[1:L_fft//2][::-1])
        x = np.fft.ifft(spec_full).real[:N]
        x = x - np.min(x) + 0.1
        inits.append(x)
    
    t_arr = np.linspace(0, 1, N)
    inits.append(np.ones(N))
    inits.append(1.0 + 0.5 * np.sin(2 * np.pi * t_arr))
    inits.append(1.0 + 0.5 * np.cos(2 * np.pi * t_arr))
    inits.append(np.exp(t_arr))
    inits.append(np.exp(-t_arr))
    inits.append(1.0 + 0.8 * (t_arr - 0.5)**2)
    inits.append(1.0 + 0.5 * (t_arr - 0.5))
    
    for _ in range(8):
        inits.append(rng.normal(1.0, 0.3, size=N))
    
    for _ in range(4):
        x = np.ones(N) * 0.1
        n_spikes = rng.integers(3, 10)
        idx = rng.choice(N, size=n_spikes, replace=False)
        x[idx] = 1.0
        inits.append(x)
    
    for _ in range(4):
        x = np.ones(N)
        for freq in rng.integers(1, 5, size=3):
            x += 0.3 * np.sin(2 * np.pi * freq * t_arr + rng.uniform(0, 2*np.pi))
        inits.append(x)
    
    return inits

def _polish(x, beta, max_iter=300):
    """
    Polish a solution using L-BFGS-B with soft-max surrogate.
    """
    try:
        def obj_and_grad_func(x):
            return _objective_and_grad(x, beta)
        
        result = minimize(obj_and_grad_func, x, method='L-BFGS-B',
                          jac=True, options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12})
        
        if result.fun < _objective_only(x):
            return result.x.copy(), result.fun
    except Exception:
        pass
    
    return x, _objective_only(x)

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Multi-start with alternating minimization
    initial_points = _spectral_init(rng)
    
    for x0 in initial_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        # Try alternating minimization
        x_alt, val_alt = _alternating_minimize(x_start, beta=50.0, n_iters=80, lr=0.01, proj_strength=0.3)
        if val_alt < best_val:
            best_val = val_alt
            best_x = x_alt.copy()
        
        # Also try phase retrieval
        if not _check_deadline():
            x_pr, val_pr = _phase_retrieval(x_start, n_iters=60)
            if val_pr < best_val:
                best_val = val_pr
                best_x = x_pr.copy()
        
        # Also try direct L-BFGS
        if not _check_deadline():
            x_lf, val_lf = _polish(x_start, beta=50.0, max_iter=150)
            if val_lf < best_val:
                best_val = val_lf
                best_x = x_lf.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Polish with higher beta
    for beta in [100.0, 200.0, 500.0]:
        if _check_deadline():
            break
        best_x, best_val = _polish(best_x, beta, max_iter=250)
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Alternating minimization refinement on the polished solution
    if not _check_deadline():
        best_x, best_val = _alternating_minimize(best_x, beta=100.0, n_iters=100, lr=0.005, proj_strength=0.2)
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Final L-BFGS polish
    if not _check_deadline():
        best_x, best_val = _polish(best_x, beta=500.0, max_iter=350)
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Structured random polish
    step_size = 0.01
    n_steps = 0
    max_steps = 150000
    
    t_arr = np.linspace(0, 1, N)
    freqs = [1, 2, 3, 5, 8]
    
    while n_steps < max_steps:
        if _check_deadline():
            break
        
        r = rng.random()
        if r < 0.3:
            f = freqs[rng.integers(0, len(freqs))]
            perturbation = step_size * np.sin(2 * np.pi * f * t_arr + rng.uniform(0, 2*np.pi))
        elif r < 0.5:
            perturbation = np.zeros(N)
            n_spikes = rng.integers(1, 4)
            idx = rng.choice(N, size=n_spikes, replace=False)
            perturbation[idx] = rng.normal(0, step_size * 3, size=n_spikes)
        else:
            perturbation = rng.normal(0, step_size, size=N)
        
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size *= 0.999
        else:
            step_size = min(0.1, step_size * 1.01)
        n_steps += 1
    
    # Phase 6: Coordinate descent polish
    if not _check_deadline():
        coord_steps = 0
        max_coord_steps = 40000
        while coord_steps < max_coord_steps:
            if _check_deadline():
                break
            improved = False
            for i in range(0, N, max(1, N // 100)):
                if _check_deadline():
                    break
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    x_trial = best_x.copy()
                    x_trial[i] += delta
                    val = _objective_only(x_trial)
                    if val < best_val:
                        best_val = val
                        best_x = x_trial
                        improved = True
                        break
            coord_steps += 1
            if not improved:
                break
    
    return best_x
# EVOLVE-BLOCK-END
