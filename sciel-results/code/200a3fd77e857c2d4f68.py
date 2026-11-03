"""Frequency-domain phase optimization for signed autocorrelation minimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

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


def _soft_max_abs(conv, beta=500.0):
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


def _phase_objective(phases, mags, target_sum, beta=200.0):
    """
    Objective function for phase optimization.
    phases: array of phases for k=1..K-1 (length K-1)
    mags: fixed magnitudes
    target_sum: fixed DC component
    """
    # Reconstruct FFT
    fx = np.empty_like(mags, dtype=complex)
    fx[0] = target_sum
    fx[1:] = mags[1:] * np.exp(1j * phases)
    
    # Inverse FFT
    x = np.fft.irfft(fx, _FFT_SIZE)[:N]
    
    # Compute metric
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return 1e10
    
    # Smooth surrogate
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    soft_max = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    return soft_max / (s * s)


def _try_phase_optimization(x_init, deadline, rng):
    """
    Optimize phases in frequency domain with fixed flat-ish magnitudes.
    Uses Nelder-Mead for derivative-free optimization.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # Initial FFT
    fx = np.fft.rfft(best_x, _FFT_SIZE)
    
    # Set magnitudes to be flat (equal magnitude for all non-DC bins)
    # This maximizes the "bandwidth" and tends to spread out the autocorrelation
    mags = np.abs(fx).copy()
    
    # Normalize magnitudes to be flat
    # We want |F[k]| to be constant for k=1..K-1
    # But we must preserve the total energy or integral? 
    # The metric is max|conv| / sum^2. 
    # Scaling x by c scales conv by c^2 and sum by c, so metric scales by 1/c.
    # Wait: max|c^2 conv| / (c sum)^2 = c^2 max|conv| / c^2 sum^2 = max|conv|/sum^2.
    # So the metric is scale-invariant!
    # We can fix the sum (DC component) to be 1.0 for convenience.
    
    target_sum = 1.0
    mags[0] = target_sum
    
    # For non-DC bins, set to a constant magnitude.
    # What value? 
    # If we choose too high, the time-domain signal might have large amplitudes, but metric is scale invariant.
    # Let's just use the average magnitude of the initial signal's non-DC bins, or just 1.0.
    # Actually, the shape of the magnitude spectrum matters. Flat is a good guess for "white noise" like properties.
    # Let's try setting all non-DC magnitudes to 1.0.
    mags[1:] = 1.0
    
    # Initial phases from the current best_x
    phases_init = np.zeros(len(fx) - 1)
    for k in range(1, len(fx)):
        if mags[k] > 1e-10:
            phases_init[k - 1] = np.angle(fx[k])
    
    # Nelder-Mead optimization
    # This can be slow for high dimensions (400 dims).
    # We will run it for a limited time or iterations.
    
    # Split into multiple runs with different initial phases to avoid local minima
    num_runs = 5
    time_per_run = (deadline - time.time()) / num_runs
    
    for run_idx in range(num_runs):
        if time.time() > deadline - 10:
            break
        
        # Randomize initial phases for diversity
        if run_idx == 0:
            phases_current = phases_init.copy()
        else:
            phases_current = rng.uniform(-np.pi, np.pi, size=len(fx) - 1)
        
        # Run Nelder-Mead
        # Since N=400, dimension is ~400. Nelder-Mead is O(N^2) per iteration, might be slow.
        # Let's try a smaller subset of phases? No, all phases matter.
        # Maybe use Powell's method? Or just run NM with tight bounds.
        
        start_time = time.time()
        try:
            result = minimize(
                _phase_objective,
                phases_current,
                args=(mags, target_sum, 200.0),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if result.fun < best_val:
                # Reconstruct x
                fx_final = np.empty_like(mags, dtype=complex)
                fx_final[0] = target_sum
                fx_final[1:] = mags[1:] * np.exp(1j * result.x)
                
                x_final = np.fft.irfft(fx_final, _FFT_SIZE)[:N]
                val_final = _metric(x_final)
                
                if val_final < best_val:
                    best_val = val_final
                    best_x = x_final.copy()
                    
        except Exception:
            pass
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > time_per_run:
            # If we exceeded time, stop early for subsequent runs
            # (We can't easily interrupt minimize, so we just skip remaining)
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
    
    # Phase 1: Frequency-domain phase optimization (New Structural Strategy)
    # This explores a different landscape: fixing magnitudes and optimizing phases.
    if time.time() < deadline - 60:
        x_phase, val_phase = _try_phase_optimization(best_x, deadline - 60, rng)
        if val_phase < best_val:
            best_val = val_phase
            best_x = x_phase
    
    # Phase 2: Gradient descent with smooth surrogate (Refinement)
    if time.time() < deadline - 30:
        x_grad, val_grad = _try_gradient_descent_smooth(best_x, deadline - 30, rng, max_iters=10000)
        if val_grad < best_val:
            best_val = val_grad
            best_x = x_grad
    
    # Phase 3: Local search
    if time.time() < deadline - 10:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 10, rng, max_rounds=50000)
    
    # Phase 4: Final polish
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
# EVOLVE-BLOCK-END
