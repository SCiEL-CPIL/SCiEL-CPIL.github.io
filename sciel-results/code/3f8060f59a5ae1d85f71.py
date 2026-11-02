"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _objective_grad_smooth(x, beta=100.0):
    """
    Smooth objective with gradient using soft-max surrogate.
    Returns (value, gradient).
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Soft-max with temperature beta
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient: d(soft_max)/dx[j] = sum_n w[n] * sign(c[n]) * (x[j+n] + x[j-n])
    # where the sum is over valid indices
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    # Use FFT to compute the convolution-like sum efficiently
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # d(soft_max)/dx = convolve(g, x) + convolve(g_reversed, x)
    # Actually, let's think more carefully:
    # c[n] = sum_k x[k] * x[k+n]
    # d(c[n])/dx[j] = x[j+n] if j+n >= 0 and j+n < N, plus x[j-n] if j-n >= 0 and j-n < N
    # So d(soft_max)/dx[j] = sum_n w[n] * sign(c[n]) * (x[j+n] + x[j-n])
    # This is like a cross-correlation
    
    # term1: sum_n w[n] * sign(c[n]) * x[j+n] for valid j+n
    # term2: sum_n w[n] * sign(c[n]) * x[j-n] for valid j-n
    
    # We can compute these using FFT
    # For term1: it's like correlating g with x
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    # Full gradient of val = soft_max / s^2
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def _epigraph_objective(params, beta=100.0):
    """
    Epigraph formulation: minimize t subject to soft_max(|autoconv|) <= t * (sum x)^2
    params: [x_0, ..., x_{N-1}, t]
    """
    x = params[:-1]
    t = params[-1]
    
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
    
    # Penalty for constraint violation
    violation = max(0.0, soft_max - t * s * s)
    return t + 100.0 * violation

def _epigraph_grad(params, beta=100.0):
    """Gradient for epigraph formulation."""
    x = params[:-1]
    t = params[-1]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.ones_like(params)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        grad = np.zeros_like(params)
        grad[-1] = 1.0
        return grad
    
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    
    violation = max(0.0, soft_max - t * s * s)
    
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
    
    grad_x = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    if violation > 0:
        grad_x += 100.0 * (1.0 / (s * s)) * d_softmax_dx - 100.0 * (2.0 * soft_max * t / (s ** 3)) * np.ones(N)
        grad_t = 1.0 - 100.0 * s * s
    else:
        grad_t = 1.0
    
    return np.concatenate([grad_x, [grad_t]])

def _flat_psd_init(rng, N):
    """
    Initialize with flat PSD to minimize autoconv peak.
    Generate random phases with flat amplitude spectrum.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # Create flat PSD in frequency domain
    psd = np.ones(L_fft // 2 + 1)
    
    # Add small random variations to break symmetry
    if rng is not None:
        psd += rng.normal(0, 0.1, size=psd.shape)
        psd = np.maximum(psd, 0.1)
    
    # Take square root to get amplitude, add random phases
    amp = np.sqrt(psd)
    phases = rng.uniform(0, 2*np.pi, size=amp.shape) if rng is not None else np.zeros_like(amp)
    
    X = amp * np.exp(1j * phases)
    
    # Make it symmetric for real signal
    X_full = np.zeros(L_fft, dtype=complex)
    X_full[:L_fft//2+1] = X
    X_full[L_fft//2+1:] = np.conj(X[1:L_fft//2][::-1])
    
    x = np.fft.irfft(X_full, n=L_fft)[:N]
    
    # Normalize to have reasonable magnitude
    x = x / np.max(np.abs(x)) * 1.0
    return x

def _coordinate_descent_fast(x, max_iters=5000, step_scale=0.1):
    """
    Fast coordinate descent using incremental FFT updates.
    """
    best_val = _objective(x)
    best_x = x.copy()
    
    # Precompute FFT size
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            # Try small adjustments
            for delta in [-step_scale, step_scale]:
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

def _spectral_design(rng, N, num_designs=20):
    """
    Generate diverse spectral designs to minimize autoconv peak.
    """
    designs = []
    
    # Flat PSD designs with random phases
    for _ in range(num_designs):
        x = _flat_psd_init(rng, N)
        designs.append(x)
    
    # Low-frequency emphasis
    for freq in [0.1, 0.2, 0.3, 0.5, 0.7]:
        t = np.linspace(0, 1, N)
        x = 1.0 + 0.5 * np.cos(2*np.pi*freq*t)
        designs.append(x)
    
    # High-frequency emphasis
    for freq in [2.0, 3.0, 5.0]:
        t = np.linspace(0, 1, N)
        x = 1.0 + 0.3 * np.cos(2*np.pi*freq*t)
        designs.append(x)
    
    # Alternating patterns
    for period in [2, 3, 4, 5]:
        x = np.ones(N)
        x[::period] = -1.0
        designs.append(x)
    
    return designs

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Spectral designs
    rng_design = np.random.default_rng(123)
    designs = _spectral_design(rng_design, N, num_designs=30)
    
    for x0 in designs:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Gradient-based optimization with smooth surrogate
    for beta in [50.0, 100.0, 200.0]:
        if _check_deadline():
            break
        
        def _smooth_obj(x, b=beta):
            return _objective_grad_smooth(x, beta=b)
        
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
    
    # Phase 3: Multi-start with perturbations
    rng = np.random.default_rng(42)
    for restart in range(15):
        if _check_deadline():
            break
        
        # Various perturbation strategies
        strategy = restart % 3
        if strategy == 0:
            x0 = best_x + rng.normal(0, 0.1, size=N)
        elif strategy == 1:
            # Random flat PSD design
            x0 = _flat_psd_init(rng, N)
        else:
            # Mix of current best and random design
            x_rand = _flat_psd_init(rng, N)
            x0 = 0.5 * best_x + 0.5 * x_rand
        
        for beta in [100.0, 200.0]:
            if _check_deadline():
                break
            
            def _obj_restart(x, b=beta):
                return _objective_grad_smooth(x, beta=b)
            
            try:
                result = minimize(_obj_restart, x0, method='L-BFGS-B', jac=True,
                                options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-12})
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Coordinate descent polish
    best_x, best_val = _coordinate_descent_fast(best_x, max_iters=3000, step_scale=0.05)
    
    if _check_deadline():
        return best_x
    
    # Phase 5: High-beta gradient polish
    def _high_beta_obj(x):
        return _objective_grad_smooth(x, beta=300.0)
    
    try:
        result = minimize(_high_beta_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Adaptive random search with FFT-based objective
    rng = np.random.default_rng(99)
    step_size = 0.02
    stall_count = 0
    max_iters = 30000
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.999)
        else:
            stall_count += 1
            if stall_count % 2000 == 0:
                step_size = min(0.2, step_size * 1.1)
    
    # Final coordinate descent
    if not _check_deadline():
        best_x, best_val = _coordinate_descent_fast(best_x, max_iters=1500, step_scale=0.02)
    
    # Very final gradient polish
    if not _check_deadline():
        def _final_obj(x):
            return _objective_grad_smooth(x, beta=500.0)
        
        try:
            result = minimize(_final_obj, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-15})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
