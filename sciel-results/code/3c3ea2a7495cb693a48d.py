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

def _phase_objective(theta, beta=30.0):
    """
    Phase-only optimization: x = Re(exp(i*theta)), so x[j] = cos(theta[j]).
    This naturally produces signed step functions in [-1, 1].
    Objective: max(|autoconv|) / (sum x)^2
    """
    x = np.cos(theta)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(theta)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(theta)
    
    # Soft-max via log-sum-exp
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient w.r.t. theta: d(val)/dtheta[j] = d(val)/dx[j] * dx[j]/dtheta[j]
    # dx[j]/dtheta[j] = -sin(theta[j])
    
    # d(val)/dx[j] computation (same as before)
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
    
    d_val_dx = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    # Chain rule: dtheta gradient
    grad_theta = d_val_dx * (-np.sin(theta))
    
    return val, grad_theta

def _phase_optimize(theta0, beta=30.0, maxiter=300):
    """Optimize phase angles using L-BFGS-B."""
    def obj_and_grad(theta):
        val, grad = _phase_objective(theta, beta=beta)
        return val, grad
    
    try:
        result = minimize(obj_and_grad, theta0, method='L-BFGS-B', jac=True,
                        options={'maxiter': maxiter, 'gtol': 1e-7, 'ftol': 1e-12})
        return result.x, result.fun
    except Exception:
        return theta0, _objective(np.cos(theta0))

def _phase_local_search(x, max_iters=2000, step_scale=0.1):
    """
    Local search in phase space: convert x to phase angles, perturb, convert back.
    """
    # Convert x to phase angles (arccos, clipped to [-1,1])
    x_clipped = np.clip(x, -1.0, 1.0)
    theta = np.arccos(x_clipped)
    
    best_val = _objective(x)
    best_x = x.copy()
    best_theta = theta.copy()
    
    rng = np.random.default_rng(999)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Perturb a random subset of phases
        n_perturb = rng.integers(3, 15)
        indices = rng.choice(N, size=n_perturb, replace=False)
        
        theta_trial = best_theta.copy()
        for idx in indices:
            theta_trial[idx] += rng.normal(0, step_scale)
        
        x_trial = np.cos(theta_trial)
        val_trial = _objective(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial
            best_theta = theta_trial
            step_scale = max(0.01, step_scale * 0.99)
        else:
            if it % 1000 == 0:
                step_scale = min(0.3, step_scale * 1.05)
    
    return best_x, best_val

def _sign_pattern_search(x, max_iters=5000):
    """
    Search over sign patterns: for each position, try flipping sign.
    This explores the discrete sign structure.
    """
    best_val = _objective(x)
    best_x = x.copy()
    rng = np.random.default_rng(4242)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Try flipping a random subset of signs
        n_flip = rng.integers(1, 5)
        indices = rng.choice(N, size=n_flip, replace=False)
        
        x_trial = best_x.copy()
        for idx in indices:
            x_trial[idx] = -x_trial[idx]
        
        val_trial = _objective(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial
    
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
    
    # Phase 2: Phase-only optimization (STRUCTURAL CHANGE)
    # Initialize phase angles from current best
    x_clipped = np.clip(best_x, -1.0, 1.0)
    theta0 = np.arccos(x_clipped)
    
    # Try multiple beta values for phase optimization
    for beta in [20.0, 40.0, 80.0, 150.0]:
        if _check_deadline():
            break
        
        theta_opt, val_opt = _phase_optimize(theta0, beta=beta, maxiter=200)
        x_opt = np.cos(theta_opt)
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt.copy()
            theta0 = theta_opt.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Phase local search
    if not _check_deadline():
        best_x, best_val = _phase_local_search(best_x, max_iters=3000, step_scale=0.1)
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Sign pattern search (explores discrete sign structure)
    if not _check_deadline():
        best_x, best_val = _sign_pattern_search(best_x, max_iters=3000)
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Coordinate descent polish
    if not _check_deadline():
        best_x, best_val = _coordinate_descent(best_x, max_iters=2000, step_scale=0.05)
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Another round of phase optimization from improved starting point
    if not _check_deadline():
        x_clipped = np.clip(best_x, -1.0, 1.0)
        theta0 = np.arccos(x_clipped)
        
        for beta in [60.0, 100.0, 200.0]:
            if _check_deadline():
                break
            
            theta_opt, val_opt = _phase_optimize(theta0, beta=beta, maxiter=200)
            x_opt = np.cos(theta_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt.copy()
                theta0 = theta_opt.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Final phase local search
    if not _check_deadline():
        best_x, best_val = _phase_local_search(best_x, max_iters=2000, step_scale=0.05)
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Final coordinate descent
    if not _check_deadline():
        best_x, best_val = _coordinate_descent(best_x, max_iters=1500, step_scale=0.03)
    
    return best_x
# EVOLVE-BLOCK-END
