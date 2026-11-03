"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _objective_grad_smooth(x, beta=30.0):
    """
    Compute soft-max surrogate objective and its gradient.
    Uses log-sum-exp for numerical stability.
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
    
    # Gradient:
    # val = soft_max / s^2
    # d(val)/dx[j] = (1/s^2) * d(soft_max)/dx[j] - (2*soft_max/s^3) * 1
    
    # d(soft_max)/dx[j] = sum_n weights[n] * sign(ac[n]) * d(ac[n])/dx[j]
    # d(ac[n])/dx[j] = x[j+n] + x[j-n] (with boundary handling)
    
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    # Compute d(soft_max)/dx efficiently using FFT
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # term1[j] = sum_n g[n] * x[j+n] (correlation)
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    
    # term2[j] = sum_n g[n] * x[j-n] 
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def _epigraph_objective(params, beta=20.0):
    """
    Epigraph formulation: minimize t subject to soft_max(|ac|)/s^2 <= t
    params = [x_0, ..., x_{N-1}, t]
    """
    x = params[:N]
    t = params[N]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return t
    
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    
    val = soft_max / (s * s)
    
    # Penalty for constraint violation
    penalty = 1e4 * max(0, val - t)
    
    return t + penalty

def _epigraph_grad(params, beta=20.0):
    """Gradient for epigraph formulation."""
    x = params[:N]
    t = params[N]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.concatenate([np.zeros(N), [1e10]])
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return np.concatenate([np.zeros(N), [1.0]])
    
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient of t + penalty
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    # Gradient of val w.r.t. x
    d_val_dx = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    # Penalty gradient
    if val > t:
        d_val_dx *= 1e4
    
    grad_x = d_val_dx
    grad_t = 1.0 + (1e4 if val > t else 0.0)
    
    return np.concatenate([grad_x, [grad_t]])

def _peak_targeted_polish(x, max_iters=500, step_scale=0.1):
    """
    Peak-targeted polish: identify the dominant lag in autoconvolution
    and apply targeted adjustments to reduce it.
    """
    best_val = _objective(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = abs_ac[peak_idx]
        
        if peak_val < 1e-10:
            break
        
        # The peak at lag k means ac[k] is large.
        # ac[k] = sum_j x[j]*x[j+k] (for k>=0)
        # To reduce |ac[k]|, we can adjust x to make the correlation at lag k smaller.
        # Use gradient descent on just the peak term with a small step.
        
        s = np.sum(best_x)
        if abs(s) < 1e-10:
            break
        
        # Compute gradient of ac[peak_idx] w.r.t. x
        k = peak_idx
        grad_peak = np.zeros(N)
        if k == 0:
            grad_peak = 2.0 * best_x
        else:
            for j in range(N - k):
                grad_peak[j] += best_x[j + k]
                grad_peak[j + k] += best_x[j]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_peak)
        if grad_norm < 1e-10:
            break
        
        # Try moving in negative gradient direction (to reduce |ac[k]|)
        # Also consider the effect on the denominator (sum x)^2
        sign_factor = np.sign(ac[peak_idx])
        
        # Try several step sizes
        improved = False
        for scale in [step_scale, step_scale * 0.5, step_scale * 0.25, step_scale * 0.1]:
            x_trial = best_x - sign_factor * scale * grad_peak / grad_norm
            val_trial = _objective(x_trial)
            if val_trial < best_val:
                best_val = val_trial
                best_x = x_trial
                improved = True
                break
        
        if not improved:
            # Try the opposite direction (in case we need to increase the sum)
            for scale in [step_scale, step_scale * 0.5]:
                x_trial = best_x + sign_factor * scale * grad_peak / grad_norm
                val_trial = _objective(x_trial)
                if val_trial < best_val:
                    best_val = val_trial
                    best_x = x_trial
                    improved = True
                    break
        
        if not improved:
            # Decay step size
            step_scale *= 0.8
            if step_scale < 0.001:
                break
    
    return best_x, best_val

def _spectral_design(rng, n_harmonics=8):
    """
    Generate a signal with low sidelobe autoconvolution using
    a windowed harmonic construction.
    """
    t = np.linspace(0, 1, N, endpoint=False)
    base = np.ones(N)
    
    # Add low-frequency components to shape the spectrum
    for h in range(1, n_harmonics + 1):
        amp = rng.uniform(0.01, 0.15) / h
        phase = rng.uniform(0, 2 * np.pi)
        base += amp * np.cos(2 * np.pi * h * t + phase)
    
    # Apply a smooth window to reduce edge effects
    window = 0.5 + 0.5 * np.cos(np.pi * np.linspace(-1, 1, N))
    sig = base * window
    
    # Normalize to have unit mean
    sig = sig / np.mean(sig)
    
    return sig

def _coordinate_descent(x, max_iters=5000, step_scale=0.1):
    """
    Coordinate descent polish: adjust individual coordinates to reduce objective.
    Uses in-place modification for speed.
    """
    best_val = _objective(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            # Try small adjustments to x[j] using in-place modification
            orig_val = best_x[j]
            for delta in [-step_scale, -step_scale*0.5, step_scale*0.5, step_scale]:
                best_x[j] = orig_val + delta
                val = _objective(best_x)
                if val < best_val:
                    best_val = val
                    improved = True
                    break
                else:
                    best_x[j] = orig_val
        
        if not improved:
            break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Structured initializations
    t = np.linspace(0, 1, N)
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
        1.0 + 0.5 * np.cos(np.pi * t),
        1.0 + np.cos(np.pi * t),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.sin(np.pi * t),
        np.exp(-2.0 * t),
        np.exp(2.0 * t - 2.0),
        1.0 + 0.2 * np.cos(2 * np.pi * 3 * t),
        1.0 + 0.3 * np.cos(2 * np.pi * 5 * t),
    ]
    
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.1, 1.2, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    # Add some random smooth initializations
    rng_init = np.random.default_rng(123)
    for _ in range(30):
        freqs = rng_init.uniform(0.1, 4.0, size=6)
        phases = rng_init.uniform(0, 2*np.pi, size=6)
        amps = rng_init.uniform(0.05, 0.6, size=6)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
        initializations.append(sig)
    
    # Add block-structured initializations
    for block_size in [10, 20, 50, 100]:
        n_blocks = N // block_size
        block_vals = rng_init.uniform(0.5, 1.5, size=n_blocks)
        sig = np.repeat(block_vals, block_size)
        if len(sig) < N:
            sig = np.concatenate([sig, np.ones(N - len(sig))])
        initializations.append(sig)
    
    # Add alternating patterns
    for period in [2, 3, 4, 5]:
        pattern = np.array([1.0, 0.8, 1.2, 0.9, 1.1][:period])
        sig = np.tile(pattern, N // period + 1)[:N]
        initializations.append(sig)
    
    # Add spectral designs
    rng_spec = np.random.default_rng(777)
    for nh in [4, 6, 8, 12]:
        for _ in range(5):
            initializations.append(_spectral_design(rng_spec, n_harmonics=nh))
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: L-BFGS-B with smooth surrogate
    def _smooth_obj(x):
        val, grad = _objective_grad_smooth(x, beta=30.0)
        return val, grad
    
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Try multiple beta values for soft-max (adaptive: start low, increase)
    for beta in [5.0, 10.0, 20.0, 40.0, 80.0, 150.0]:
        if _check_deadline():
            break
        
        def _obj_beta(x, b=beta):
            val, grad = _objective_grad_smooth(x, beta=b)
            return val, grad
        
        try:
            result = minimize(_obj_beta, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 250, 'gtol': 1e-7, 'ftol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Epigraph formulation optimization
    try:
        t_init = best_val * 1.1
        params0 = np.concatenate([best_x, [t_init]])
        
        result = minimize(_epigraph_objective, params0, jac=_epigraph_grad,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-12})
        
        if result.fun < best_val:
            x_new = result.x[:N]
            val_new = _objective(x_new)
            if val_new < best_val:
                best_val = val_new
                best_x = x_new.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, max_iters=300, step_scale=0.1)
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Coordinate descent polish
    best_x, best_val = _coordinate_descent(best_x, max_iters=3000, step_scale=0.05)
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Final L-BFGS-B polish with higher beta (sharper approximation to max)
    def _final_obj(x):
        val, grad = _objective_grad_smooth(x, beta=100.0)
        return val, grad
    
    try:
        result = minimize(_final_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Stochastic coordinate search refinement
    rng = np.random.default_rng(42)
    step_size = 0.05
    stall_count = 0
    max_iters = 200000
    n_trial_coords = 5  # Try modifying this many random coordinates per iteration
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        
        # Pick a random subset of coordinates to perturb
        if n_trial_coords < N:
            indices = rng.choice(N, size=n_trial_coords, replace=False)
        else:
            indices = np.arange(N)
        
        x_trial = best_x.copy()
        for idx in indices:
            x_trial[idx] += rng.normal(0, step_size)
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.9999)
        else:
            stall_count += 1
            if stall_count % 5000 == 0:
                step_size = min(0.2, step_size * 1.05)
    
    # Final peak-targeted polish
    if not _check_deadline():
        best_x, best_val = _peak_targeted_polish(best_x, max_iters=200, step_scale=0.05)
    
    # Final coordinate descent
    if not _check_deadline():
        best_x, best_val = _coordinate_descent(best_x, max_iters=3000, step_scale=0.03)
    
    # Very final polish with higher beta
    if not _check_deadline():
        def _final_high_beta_obj(x):
            val, grad = _objective_grad_smooth(x, beta=200.0)
            return val, grad
        
        try:
            result = minimize(_final_high_beta_obj, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Final peak-targeted polish after high-beta
    if not _check_deadline():
        best_x, best_val = _peak_targeted_polish(best_x, max_iters=150, step_scale=0.03)
    
    # Final coordinate descent after high-beta polish
    if not _check_deadline():
        best_x, best_val = _coordinate_descent(best_x, max_iters=2000, step_scale=0.02)
    
    return best_x
# EVOLVE-BLOCK-END
