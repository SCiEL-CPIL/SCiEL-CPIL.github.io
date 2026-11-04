"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400

# EVOLVE-BLOCK-START
def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    if n == 0:
        return np.array([])
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    F = np.fft.rfft(x, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]


def _score_from_signal(x):
    """
    Compute the metric: max(|autoconv|) / (sum(x))^2
    We want to minimize this.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    max_abs = np.max(np.abs(conv))
    return max_abs / (integral * integral)


def _param_to_signal(params):
    """
    Convert spectral coefficients to a time-domain signal.
    params: array of length M, representing coefficients of cos/sin terms.
    x[n] = c0 + sum_{k=1}^{M} c_k * cos(2*pi*k*n/N) + d_k * sin(2*pi*k*n/N)
    """
    M = len(params) // 2
    c0 = params[0]
    x = np.full(N, c0)
    n = np.arange(N)
    for k in range(1, M + 1):
        c_k = params[2 * k - 1]
        d_k = params[2 * k]
        x += c_k * np.cos(2 * np.pi * k * n / N) + d_k * np.sin(2 * np.pi * k * n / N)
    return x


def _objective(params):
    """Objective function for differential evolution."""
    x = _param_to_signal(params)
    return _score_from_signal(x)


def _fft_size_for_n(n):
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    return fft_size


def _autoconv_values(x):
    fft_size = _fft_size_for_n(len(x))
    fx = np.fft.rfft(x, fft_size)
    conv = np.fft.irfft(fx * fx, fft_size)
    return conv[:2 * len(x) - 1]


def _soft_max_abs(conv, beta=1000.0):
    abs_conv = np.abs(conv)
    if len(abs_conv) == 0:
        return 0.0
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta


def _smooth_metric(x, beta=500.0):
    conv = _autoconv_values(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e12
    return _soft_max_abs(conv, beta) / (s * s)


def _phase_objective(phase_vec, mags, target_sum, N_val, fft_size_val, beta):
    """
    Objective for phase optimization.
    phase_vec: array of length len(fx)-1
    Returns value and gradient (approximate).
    """
    fx_new = mags.copy()
    fx_new[0] = target_sum
    
    # Apply phases
    # fx_new[1:] = mags[1:] * np.exp(1j * phase_vec)
    # But we need to handle the real constraint? irfft assumes hermitian symmetry.
    # mags and phases must correspond to a real signal.
    # If we fix mags to be from a real signal, and optimize phases,
    # we must ensure the resulting spectrum is Hermitian.
    # rfft output is [0, 1, ..., N-1] (if N is even, N/2 is Nyquist).
    # For a real signal, F[k] = conj(F[N-k]).
    # rfft only returns the first half.
    # So if we modify phase_vec, we must ensure the reconstructed full spectrum is Hermitian.
    # Since we are using rfft/irfft, irfft automatically assumes the input is the first half of a Hermitian spectrum.
    # So we just need to construct the first half correctly.
    
    fx_new[1:] = mags[1:] * np.exp(1j * phase_vec)
    
    x_new = np.fft.irfft(fx_new, fft_size_val)[:N_val]
    
    conv = _autoconv_values(x_new)
    s = np.sum(x_new)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(phase_vec)
    
    soft_max = _soft_max_abs(conv, beta)
    val = soft_max / (s * s)
    
    # Approximate gradient
    eps = 1e-6
    grad = np.zeros(len(phase_vec))
    for j in range(len(phase_vec)):
        phase_vec[j] += eps
        fx_temp = mags.copy()
        fx_temp[0] = target_sum
        fx_temp[1:] = mags[1:] * np.exp(1j * phase_vec)
        x_temp = np.fft.irfft(fx_temp, fft_size_val)[:N_val]
        conv_temp = _autoconv_values(x_temp)
        s_temp = np.sum(x_temp)
        if abs(s_temp) < 1e-12:
            grad[j] = 0.0
        else:
            val_temp = _soft_max_abs(conv_temp, beta) / (s_temp * s_temp)
        phase_vec[j] -= eps
        grad[j] = (val_temp - val) / eps
        
    return val, grad


def _spectral_phase_optimize(x_init, deadline, max_iters=2000):
    """
    Optimize phases of the FFT spectrum to minimize the metric,
    keeping magnitudes fixed.
    """
    best_x = x_init.copy()
    best_val = _score_from_signal(best_x)
    
    fft_size = _fft_size_for_n(N)
    fx = np.fft.rfft(best_x, fft_size)
    
    mags = np.abs(fx).copy()
    mags[mags < 1e-10] = 1e-10
    target_sum = np.sum(best_x)
    mags[0] = abs(target_sum)
    
    phases_init = np.zeros(len(fx) - 1)
    for k in range(1, len(fx)):
        if mags[k] > 1e-10:
            phases_init[k - 1] = np.angle(fx[k])
    
    # Use L-BFGS-B
    try:
        result = minimize(
            _phase_objective,
            phases_init,
            args=(mags, target_sum, N, fft_size, 200.0),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        
        if result.success or result.fun < best_val:
            fx_final = mags.copy()
            fx_final[0] = target_sum
            fx_final[1:] = mags[1:] * np.exp(1j * result.x)
            
            x_final = np.fft.irfft(fx_final, fft_size)[:N]
            val_final = _score_from_signal(x_final)
            
            if val_final < best_val:
                best_val = val_final
                best_x = x_final.copy()
    except Exception:
        pass
    
    return best_x, best_val


def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Initial best: all ones
    best_x = np.ones(N)
    best_score = _score_from_signal(best_x)
    
    # Differential evolution with spectral parameterization
    M = 15
    n_params = 2 * M + 1
    bounds = [(-10, 10)] * n_params
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True,
            updating='immediate'
        )
        
        if result.success or result.fun < best_score:
            x_opt = _param_to_signal(result.x)
            score_opt = _score_from_signal(x_opt)
            if score_opt < best_score:
                best_x = x_opt
                best_score = score_opt
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 20:
        return best_x
    
    # Try spectral phase optimization
    # This uses the insight from the inspiration to refine the solution
    # by optimizing phases while keeping magnitudes fixed.
    x_refined, score_refined = _spectral_phase_optimize(best_x, deadline, max_iters=1000)
    if score_refined < best_score:
        best_x = x_refined
        best_score = score_refined
    
    # Check time
    if time.time() > deadline - 10:
        return best_x
    
    # Try a few more random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(3):
        if time.time() > deadline - 10:
            break
        
        # Random initial population
        init = rng.uniform(-5, 5, size=n_params)
        
        try:
            result = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=50,
                popsize=10,
                tol=1e-8,
                seed=restart + 100,
                x0=init,
                polish=True,
                updating='immediate'
            )
            
            if result.success or result.fun < best_score:
                x_opt = _param_to_signal(result.x)
                score_opt = _score_from_signal(x_opt)
                if score_opt < best_score:
                    best_x = x_opt
                    best_score = score_opt
                    
                    # Refine with spectral phase optimization
                    x_refined, score_refined = _spectral_phase_optimize(best_x, deadline, max_iters=500)
                    if score_refined < best_score:
                        best_x = x_refined
                        best_score = score_refined
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
