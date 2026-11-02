import numpy as np
import time
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 145
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

def _objective_grad_smooth(x, beta=50.0):
    """
    Compute soft-max surrogate objective and its gradient.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Soft-max via log-sum-exp
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient computation
    # d(soft_max)/dx[j] = sum_n weights[n] * sign(ac[n]) * (x[j+n] + x[j-n])
    
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # term1[j] = sum_n g[n] * x[j+n]
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    
    # term2[j] = sum_n g[n] * x[j-n]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def _generate_random_phase_init(rng, N):
    """
    Generate an initial guess based on random Fourier phases.
    This tends to produce sequences with flatter autoconvolutions.
    """
    # We want x to be real.
    # Construct X such that X[k] = conj(X[N-k])
    L_fft = 1
    while L_fft < 2*N:
        L_fft *= 2
    
    # Random phases
    phases = rng.uniform(0, 2*np.pi, size=L_fft//2 + 1)
    
    # Amplitude: Try a few shapes
    # 1. Flat
    # 2. Gaussian
    k = np.arange(L_fft//2 + 1)
    t = k / (L_fft/2)
    
    # Randomly choose amplitude profile
    profile_type = rng.integers(0, 3)
    if profile_type == 0:
        amps = np.ones(L_fft//2 + 1)
    elif profile_type == 1:
        sigma = rng.uniform(0.2, 0.5)
        amps = np.exp(-t**2 / (2*sigma**2))
    else:
        # Linear taper
        amps = 1.0 - 0.5 * t
    
    # Combine
    X_half = amps * np.exp(1j * phases)
    
    # Construct full spectrum
    X = np.zeros(L_fft, dtype=complex)
    X[:L_fft//2 + 1] = X_half
    X[L_fft//2 + 1:] = np.conj(X_half[L_fft//2-1:0:-1])
    
    # IFFT
    x = np.fft.irfft(X, n=L_fft)[:N]
    
    # Ensure non-negative? Or just real?
    # The problem allows signed step functions.
    # However, negative values might reduce the sum S, hurting the score.
    # Let's try shifting to make it non-negative if possible, or just normalize.
    # Actually, if we allow negative, we might get better local minima.
    # But for "step functions" usually implies heights, often positive.
    # Let's normalize to sum 1.
    s = np.sum(x)
    if abs(s) > 1e-10:
        x = x / s
    else:
        x = np.ones(N) / N
        
    return x

def _coordinate_descent_polish(x, max_iters=500, step_scale=0.05):
    """
    Fast coordinate descent targeting the current max peak.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    
    for it in range(max_iters):
        if _check_deadline():
            break
            
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        max_idx = np.argmax(abs_ac)
        
        # Try to reduce the peak at max_idx by adjusting x
        # The peak value is ac[max_idx] = sum_i x[i] x[max_idx-i]
        # We can do a line search on a small subset of x or just random perturbations.
        # Better: Gradient step towards reducing ac[max_idx]
        
        # Compute gradient of ac[max_idx] w.r.t x
        # d(ac[k])/dx[j] = x[k-j] + x[j-k] (with bounds)
        k = max_idx
        grad_peak = np.zeros(N)
        for j in range(N):
            val = 0.0
            if k - j >= 0 and k - j < N:
                val += best_x[k-j]
            if j - k >= 0 and j - k < N:
                val += best_x[j-k]
            grad_peak[j] = val
            
        # Normalize gradient
        norm_g = np.linalg.norm(grad_peak)
        if norm_g > 1e-10:
            grad_peak /= norm_g
            
        # Try moving in negative gradient direction
        for step in [step_scale, step_scale*0.5, step_scale*0.1]:
            trial = best_x - step * grad_peak
            # Ensure sum doesn't change too much? No, objective handles normalization.
            val = _objective(trial)
            if val < best_val:
                best_val = val
                best_x = trial.copy()
                break
        else:
            # Try positive direction?
            for step in [step_scale, step_scale*0.5]:
                trial = best_x + step * grad_peak
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    break
            else:
                break
                
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # List of initializations
    inits = []
    
    # 1. Standard smooth
    t = np.linspace(0, 1, N)
    inits.append(np.ones(N))
    inits.append(np.exp(-((t-0.5)**2)/0.05))
    inits.append(1.0 + 0.5*np.cos(2*np.pi*t))
    inits.append(1.0 + 0.5*np.sin(2*np.pi*t))
    inits.append(np.linspace(1, 0.5, N))
    
    # 2. Random Phase Initializations (Key structural addition)
    for _ in range(30):
        inits.append(_generate_random_phase_init(rng, N))
    
    # 3. Random Smooth
    for _ in range(10):
        freqs = rng.uniform(0.1, 5.0, size=3)
        phases = rng.uniform(0, 2*np.pi, size=3)
        amps = rng.uniform(0.1, 0.5, size=3)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
        # Ensure positive for stability
        sig = np.maximum(sig, 0.1)
        inits.append(sig / np.sum(sig))
        
    # Optimization
    for x0 in inits:
        if _check_deadline():
            break
            
        # Normalize to sum 1
        s0 = np.sum(x0)
        if abs(s0) < 1e-10:
            continue
        x0 = x0 / s0
        
        # Phase 1: L-BFGS with moderate beta
        def _obj_smooth(x, b=50.0):
            val, grad = _objective_grad_smooth(x, beta=b)
            return val, grad
            
        try:
            res = minimize(_obj_smooth, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 100, 'gtol': 1e-5, 'ftol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
            
        if _check_deadline():
            break
            
        # Phase 2: Coordinate Descent Polish
        best_x, best_val = _coordinate_descent_polish(best_x, max_iters=100)
        
        # Phase 3: Refine with higher beta if time permits
        if not _check_deadline():
            def _obj_sharp(x, b=100.0):
                val, grad = _objective_grad_smooth(x, beta=b)
                return val, grad
                
            try:
                res = minimize(_obj_sharp, best_x, method='L-BFGS-B', jac=True,
                               options={'maxiter': 50, 'gtol': 1e-6, 'ftol': 1e-12})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
            if not _check_deadline():
                best_x, best_val = _coordinate_descent_polish(best_x, max_iters=50)

    return best_x
