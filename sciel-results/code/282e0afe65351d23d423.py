"""Refined spectral-design approach: flat-spectrum via magnitude-phase decomposition with epigraph formulation."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _integral(x):
    return float(np.sum(x))


def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _soft_max_abs(conv, beta=1000.0):
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta


def _smooth_metric(x, beta=500.0):
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)


def _epigraph_objective(params, mags, target_sum, beta=300.0):
    """
    Epigraph formulation: minimize t subject to soft_max_abs(conv) / s^2 <= t
    Variables: params = [phases..., log_t] where t = exp(log_t)
    This transforms the constraint into a smooth objective.
    """
    n_phases = len(mags) - 1
    phases = params[:n_phases]
    log_t = params[n_phases]
    
    # Reconstruct FFT
    fx = mags.copy()
    fx[0] = target_sum
    for k in range(1, len(fx)):
        fx[k] = mags[k] * np.exp(1j * phases[k - 1])
    
    x = np.fft.irfft(fx, _FFT_SIZE)[:N]
    s = _integral(x)
    
    if abs(s) < 1e-12:
        return 1e10
    
    conv = _autoconv_values(x)
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    soft_max = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    t_val = np.exp(log_t)
    constraint_val = soft_max / (s * s)
    
    # Penalized epigraph: minimize t + penalty * max(0, constraint - t)
    penalty = 1e4
    violation = np.maximum(0, constraint_val - t_val)
    return t_val + penalty * violation


def _epigraph_gradient(params, mags, target_sum, beta=300.0):
    """Gradient of epigraph objective via finite differences."""
    n_phases = len(mags) - 1
    eps = 1e-5
    grad = np.zeros(len(params))
    
    for j in range(len(params)):
        params_plus = params.copy()
        params_plus[j] += eps
        val_plus = _epigraph_objective(params_plus, mags, target_sum, beta)
        
        params_minus = params.copy()
        params_minus[j] -= eps
        val_minus = _epigraph_objective(params_minus, mags, target_sum, beta)
        
        grad[j] = (val_plus - val_minus) / (2 * eps)
    
    return grad


def _spectral_design_optimize(x_init, deadline, max_iters=5000):
    """
    Optimize in the frequency domain using epigraph formulation:
    minimize t subject to max|autoconv|/s^2 <= t
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # Initial FFT
    fx = np.fft.rfft(best_x, _FFT_SIZE)
    
    # Target: flat magnitude spectrum
    mags = np.abs(fx).copy()
    mags[mags < 1e-10] = 1e-10
    
    target_sum = _integral(best_x)
    mags[0] = abs(target_sum)
    
    # Initialize phases
    n_phases = len(fx) - 1
    phases_init = np.zeros(n_phases)
    for k in range(1, len(fx)):
        if mags[k] > 1e-10:
            phases_init[k - 1] = np.angle(fx[k])
    
    # Initial t: current metric value
    t_init = _metric(best_x)
    log_t_init = np.log(max(t_init, 1e-10))
    
    # Parameters: [phases..., log_t]
    params_init = np.concatenate([phases_init, [log_t_init]])
    
    def objective(params):
        return _epigraph_objective(params, mags, target_sum, beta=300.0)
    
    def gradient(params):
        return _epigraph_gradient(params, mags, target_sum, beta=300.0)
    
    # Optimize with deadline guard
    time_limit = max(1.0, deadline - time.time() - 5.0)
    
    try:
        result = minimize(
            objective,
            params_init,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8,
                     'maxfun': 50000, 'maxls': 50}
        )
        
        # Reconstruct from optimized parameters
        phases_opt = result.x[:n_phases]
        fx_final = mags.copy()
        fx_final[0] = target_sum
        for k in range(1, len(fx)):
            fx_final[k] = mags[k] * np.exp(1j * phases_opt[k - 1])
        
        x_final = np.fft.irfft(fx_final, _FFT_SIZE)[:N]
        val_final = _metric(x_final)
        
        if val_final < best_val:
            best_val = val_final
            best_x = x_final.copy()
    except Exception:
        pass
    
    return best_x, best_val


def _try_gradient_descent_smooth(x_init, deadline, rng, max_iters=20000):
    """Gradient descent using smooth log-sum-exp surrogate."""
    best_x = x_init.copy()
    best_val = _metric(best_x)
    current_x = x_init.copy()
    
    lr = 0.005
    patience = 0
    best_smooth_val = _smooth_metric(current_x, beta=300.0)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        conv = _autoconv_values(current_x)
        abs_conv = np.abs(conv)
        s = _integral(current_x)
        
        if abs(s) < 1e-12:
            break
        
        beta = 300.0
        shifted = abs_conv - np.max(abs_conv)
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        
        grad = np.zeros(N)
        for k in range(2 * N - 1):
            if weights[k] < 1e-10:
                continue
            sign_k = np.sign(conv[k]) if conv[k] != 0 else 1.0
            for j in range(N):
                contrib = 0.0
                if j + k < N:
                    contrib += current_x[j + k]
                if j - k >= 0:
                    contrib += current_x[j - k]
                grad[j] += 2.0 * sign_k * weights[k] * contrib
        
        soft_max_val = _soft_max_abs(conv, beta)
        grad *= -2.0 * soft_max_val / (s ** 3)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad /= grad_norm
        
        x_new = current_x - lr * grad
        v_new = _metric(x_new)
        if v_new < best_val:
            best_val = v_new
            best_x = x_new.copy()
            current_x = x_new
            patience = 0
            lr = min(lr * 1.05, 0.05)
        else:
            patience += 1
            if patience > 30:
                lr *= 0.7
                patience = 0
                if lr < 1e-7:
                    break
    
    return best_x, best_val


def _try_local_search_fast(x, deadline, rng, max_rounds=100000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.2
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 150:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        k = int(rng.integers(1, 5))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(2):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(30):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.05, 0.05
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(10):
                if time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x1 = best_x.copy()
                x1[idx] += m1
                v1 = _metric(x1)
                
                x2 = best_x.copy()
                x2[idx] += m2
                v2 = _metric(x2)
                
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
                
                if v1 < best_v:
                    best_v = v1
                    best_delta = m1
                if v2 < best_v:
                    best_v = v2
                    best_delta = m2
            
            if best_delta != 0.0 and best_v < best_val:
                best_x[idx] += best_delta
                best_val = best_v
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(2024)
    
    # Phase 1: Spectral design optimization with epigraph formulation
    if time.time() < deadline - 80:
        x_spec, val_spec = _spectral_design_optimize(best_x, deadline - 80, max_iters=3000)
        if val_spec < best_val:
            best_val = val_spec
            best_x = x_spec
    
    # Phase 2: Gradient descent with smooth surrogate
    if time.time() < deadline - 40:
        x_grad, val_grad = _try_gradient_descent_smooth(best_x, deadline - 40, rng, max_iters=10000)
        if val_grad < best_val:
            best_val = val_grad
            best_x = x_grad
    
    # Phase 3: Local search
    if time.time() < deadline - 15:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 15, rng, max_rounds=50000)
    
    # Phase 4: Final polish
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
# EVOLVE-BLOCK-END
