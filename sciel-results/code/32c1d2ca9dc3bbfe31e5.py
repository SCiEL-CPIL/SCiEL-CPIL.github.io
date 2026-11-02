"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

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

def _objective_grad_smooth(x, beta=50.0):
    """
    Compute soft-max surrogate objective and its gradient using log-sum-exp.
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
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 2 * (2*N-1):
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

def _epigraph_objective(params, beta=50.0):
    """
    Epigraph formulation: minimize t subject to soft_max(autoconv) <= t * (sum x)^2.
    params = [x_0, ..., x_{N-1}, t]
    """
    x = params[:N]
    t = params[N]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(params)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return t, np.zeros_like(params)
    
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    
    # Objective: t + penalty for constraint violation
    constraint_violation = soft_max - t * s * s
    penalty = max(0, constraint_violation) ** 2 + 0.01 * constraint_violation
    
    val = t + penalty
    
    # Gradient
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 2 * (2*N-1):
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    d_softmax_dx = term1 + term2
    
    grad_x = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    # Gradient of penalty
    if constraint_violation > 0:
        d_penalty_dx = 2 * constraint_violation * (d_softmax_dx / (s*s) - 2*soft_max/(s**3)) + 0.01 * (d_softmax_dx / (s*s) - 2*soft_max/(s**3))
        d_penalty_dt = -(2 * constraint_violation + 0.01) * s * s
    else:
        d_penalty_dx = np.zeros(N)
        d_penalty_dt = 0.0
    
    grad = np.zeros_like(params)
    grad[:N] = grad_x + d_penalty_dx
    grad[N] = 1.0 + d_penalty_dt
    
    return val, grad

def _coordinate_descent(x, max_iters=3000, step_scale=0.05):
    """
    Coordinate descent polish: adjust individual coordinates to reduce objective.
    """
    best_val = _objective(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            for delta in [-step_scale, -step_scale*0.5, step_scale*0.5, step_scale]:
                trial = best_x.copy()
                trial[j] += delta
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_x, best_val

def _spectral_design(a, tilt=0.0, mod=0.0):
    """
    Generate a step function from spectral parameters.
    """
    a = np.clip(a, 0.5, 3.0)
    tilt = np.clip(tilt, -1.0, 1.0)
    mod = np.clip(mod, 0.0, 1.0)
    
    k = np.arange(N)
    c = 0.5 + a
    kf = k / N
    
    mag = np.exp(-c * kf**2)
    mag = mag * np.exp(tilt * kf)
    
    X = mag.astype(complex)
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))
    
    if mod > 0.01:
        t = np.linspace(0, 1, N)
        x = x * (1.0 + mod * 0.5 * np.sin(2*np.pi*t))
        x = x / np.max(np.abs(x))
    
    return x

def _prolate_spectral_design(bandwidth, phase_params=None):
    """
    Generate step function using prolate spheroidal sequence amplitude with optimized phase.
    """
    k = np.arange(N)
    kf = k / N
    
    # Prolate-like amplitude: concentrated in low frequencies
    mag = np.exp(-bandwidth * kf**2)
    
    if phase_params is not None:
        # Phase: sum of sinusoidal components
        phase = np.zeros(N)
        for freq, amp, ph in phase_params:
            phase += amp * np.sin(2 * np.pi * freq * kf + ph)
        X = mag * np.exp(1j * phase)
    else:
        X = mag.astype(complex)
    
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))
    
    return x

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
        freqs = rng_init.uniform(0.1, 3.0, size=5)
        phases = rng_init.uniform(0, 2*np.pi, size=5)
        amps = rng_init.uniform(0.1, 0.5, size=5)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
        initializations.append(sig)
    
    # Add prolate spectral designs
    for bw in [0.5, 1.0, 1.5, 2.0, 3.0]:
        initializations.append(_prolate_spectral_design(bw))
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: L-BFGS-B with smooth surrogate - multiple beta values
    for beta in [30.0, 50.0, 100.0]:
        if _check_deadline():
            break
        
        def _smooth_obj(x, b=beta):
            val, grad = _objective_grad_smooth(x, beta=b)
            return val, grad
        
        try:
            result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Epigraph formulation optimization
    for beta in [50.0, 100.0]:
        if _check_deadline():
            break
        
        # Initialize t with current best value
        t0 = best_val * 1.01
        params0 = np.concatenate([best_x, [t0]])
        
        try:
            result = minimize(_epigraph_objective, params0, method='L-BFGS-B', jac=True,
                            options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-14},
                            args=(beta,))
            if result.fun < best_val:
                candidate_x = result.x[:N]
                val = _objective(candidate_x)
                if val < best_val:
                    best_val = val
                    best_x = candidate_x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Spectral parameter optimization
    def _param_objective(params):
        x = _spectral_design(params[0], params[1], params[2])
        return _objective(x)
    
    param_initials = [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.5, 0.5, 0.0],
        [1.5, -0.5, 0.0],
        [1.0, 0.0, 0.5],
        [2.0, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
    ]
    
    for p0 in param_initials:
        if _check_deadline():
            break
        try:
            result = minimize(_param_objective, p0, method='Nelder-Mead', 
                            options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if result.fun < best_val:
                best_val = result.fun
                best_x = _spectral_design(result.x[0], result.x[1], result.x[2])
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Prolate spectral phase optimization
    def _prolate_phase_objective(params):
        bandwidth = params[0]
        phase_params = []
        idx = 1
        for i in range(3):
            freq = params[idx]; idx += 1
            amp = params[idx]; idx += 1
            ph = params[idx]; idx += 1
            phase_params.append((freq, amp, ph))
        x = _prolate_spectral_design(bandwidth, phase_params)
        return _objective(x)
    
    prolate_initials = []
    for bw in [0.5, 1.0, 1.5, 2.0]:
        for _ in range(3):
            p0 = [bw]
            for _ in range(3):
                p0.append(rng_init.uniform(0.5, 3.0))  # freq
                p0.append(rng_init.uniform(0.1, 1.0))  # amp
                p0.append(rng_init.uniform(0, 2*np.pi))  # phase
            prolate_initials.append(p0)
    
    for p0 in prolate_initials:
        if _check_deadline():
            break
        try:
            result = minimize(_prolate_phase_objective, p0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                bandwidth = result.x[0]
                phase_params = []
                idx = 1
                for i in range(3):
                    freq = result.x[idx]; idx += 1
                    amp = result.x[idx]; idx += 1
                    ph = result.x[idx]; idx += 1
                    phase_params.append((freq, amp, ph))
                best_x = _prolate_spectral_design(bandwidth, phase_params)
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Coordinate descent polish
    best_x, best_val = _coordinate_descent(best_x, max_iters=5000, step_scale=0.03)
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Final L-BFGS-B polish with tight tolerance
    def _smooth_obj_final(x):
        val, grad = _objective_grad_smooth(x, beta=100.0)
        return val, grad
    
    try:
        result = minimize(_smooth_obj_final, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-15})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Random perturbation refinement with adaptive step
    rng = np.random.default_rng(42)
    step_size = 0.03
    stall_count = 0
    
    for iteration in range(30000):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.9999)
        else:
            stall_count += 1
            if stall_count % 5000 == 0:
                step_size = min(0.3, step_size * 1.05)
    
    # Final polish: tight BFGS on exact objective
    if not _check_deadline():
        try:
            result = minimize(_objective, best_x, method='BFGS', 
                            options={'maxiter': 200, 'gtol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
