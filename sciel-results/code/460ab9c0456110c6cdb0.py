"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _autoconv_values(x):
    """Full autoconvolution array for gradient computation."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _soft_max_abs(conv, beta=1000.0):
    """Smooth approximation of max |conv| using log-sum-exp."""
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    # log-sum-exp trick for numerical stability
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta


def _smooth_metric(x, beta=1000.0):
    """Smooth surrogate: soft_max |autoconv| / (integral)^2."""
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)


def _try_near_constant_structured(deadline, rng):
    """Try structured perturbations that maintain high integral while reducing autoconv peaks."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Chebyshev polynomial perturbations - optimal for minimax problems
    # T_k(cos(theta)) = cos(k*theta) has minimax property
    t = np.linspace(-1, 1, N)
    for degree in [2, 3, 4, 5, 6, 8, 10]:
        if time.time() > deadline:
            break
        # Generate Chebyshev polynomial of given degree
        # Use recurrence: T_0=1, T_1=x, T_{k+1}=2x*T_k - T_{k-1}
        T_k = np.ones(N)
        if degree == 1:
            T_k = t.copy()
        elif degree >= 2:
            T_prev = np.ones(N)
            T_curr = t.copy()
            for _ in range(2, degree + 1):
                T_next = 2 * t * T_curr - T_prev
                T_prev = T_curr
                T_curr = T_next
            T_k = T_curr.copy()
        
        # Center to maintain zero mean
        T_k_centered = T_k - np.mean(T_k)
        T_k_centered /= np.max(np.abs(T_k_centered)) if np.max(np.abs(T_k_centered)) > 0 else 1
        
        for amp in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
            if time.time() > deadline:
                break
            x = np.ones(N) + amp * T_k_centered
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    
    # Multi-frequency sinusoidal perturbations with small amplitudes
    freqs = [1, 2, 3, 4, 5, 7, 8, 11, 13]
    for f1 in freqs[:5]:
        if time.time() > deadline:
            break
        for f2 in freqs[:5]:
            if time.time() > deadline:
                break
            if f1 == f2:
                continue
            for amp1 in [0.01, 0.02, 0.05]:
                if time.time() > deadline:
                    break
                for amp2 in [0.01, 0.02, 0.05]:
                    if time.time() > deadline:
                        break
                    for phase1 in range(3):
                        if time.time() > deadline:
                            break
                        for phase2 in range(3):
                            if time.time() > deadline:
                                break
                            x = np.ones(N)
                            x += amp1 * np.sin(2 * np.pi * f1 * np.arange(N) / N + 2 * np.pi * phase1 / 3)
                            x += amp2 * np.sin(2 * np.pi * f2 * np.arange(N) / N + 2 * np.pi * phase2 / 3)
                            v = _metric(x)
                            if v < best_val:
                                best_val = v
                                best_x = x.copy()
    
    # Cosine windowed perturbations - taper at edges to reduce boundary effects
    for width in [50, 100, 150, 200, 300]:
        if time.time() > deadline:
            break
        for amp in [0.01, 0.02, 0.05, 0.1]:
            if time.time() > deadline:
                break
            for freq in [1, 2, 3, 5]:
                if time.time() > deadline:
                    break
                center = N // 2
                t = np.arange(N) - center
                window = np.cos(np.pi * t / width) ** 2
                window[window < 0] = 0
                window /= np.max(window) if np.max(window) > 0 else 1
                x = np.ones(N) + amp * window * np.sin(2 * np.pi * freq * np.arange(N) / N)
                v = _metric(x)
                if v < best_val:
                    best_val = v
                    best_x = x.copy()
    
    return best_x, best_val


def _try_fft_guided_optimization(x_init, deadline, rng):
    """Use FFT analysis to guide perturbations that reduce peak autoconvolution."""
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # Analyze current autoconvolution to find problematic lags
    fx = np.fft.rfft(best_x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
    
    # Find lags with large |autoconv|
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    
    # Try to reduce peaks by adding small anti-correlated perturbations
    # at positions that contribute to the peak lags
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Recompute if needed
        fx = np.fft.rfft(best_x, _FFT_SIZE)
        conv = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
        abs_conv = np.abs(conv)
        peak_lag = np.argmax(abs_conv)
        peak_val = abs_conv[peak_lag]
        
        if peak_val < max_val * 0.9:
            # Peaks have been reduced, try more iterations with smaller steps
            max_val = peak_val
        
        # Create perturbation that anti-correlates with the peak lag
        # x_i * x_{i+lag} contributes to autoconv at lag
        # We want to reduce sum x_i * x_{i+lag}
        # Perturb x_i by -eta * x_{i+lag} (clipped)
        eta = 0.01 * (1 - iteration / 50.0)
        
        x_new = best_x.copy()
        lag = peak_lag if peak_lag < N else 2 * N - 1 - peak_lag
        if lag > 0 and lag < N:
            # For lag > 0: perturb x_i based on x_{i+lag}
            for i in range(N - lag):
                x_new[i] -= eta * best_x[i + lag]
            # For the wrap-around part, use x_{i-lag}
            for i in range(N - lag, N):
                x_new[i] -= eta * best_x[i - lag]
        
        # Also try the conjugate perturbation
        x_new2 = best_x.copy()
        if lag > 0 and lag < N:
            for i in range(N - lag):
                x_new2[i] += eta * best_x[i + lag]
            for i in range(N - lag, N):
                x_new2[i] += eta * best_x[i - lag]
        
        v1 = _metric(x_new)
        v2 = _metric(x_new2)
        
        if v1 < best_val:
            best_val = v1
            best_x = x_new
        elif v2 < best_val:
            best_val = v2
            best_x = x_new2
    
    return best_x, best_val


def _try_gradient_descent_smooth(x_init, deadline, rng, max_iters=20000):
    """Gradient descent using smooth log-sum-exp surrogate for the max term."""
    best_x = x_init.copy()
    best_val = _metric(best_x)
    current_x = x_init.copy()
    
    # Adaptive learning rate
    lr = 0.01
    patience = 0
    best_smooth_val = _smooth_metric(current_x, beta=500.0)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute gradient of smooth metric
        conv = _autoconv_values(current_x)
        abs_conv = np.abs(conv)
        s = _integral(current_x)
        
        if abs(s) < 1e-12:
            break
        
        # Gradient of soft_max via log-sum-exp
        beta = 500.0
        shifted = abs_conv - np.max(abs_conv)
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        
        # Gradient of |conv_k| w.r.t. x is 2 * sign(conv_k) * (convolution with shifted x)
        # For autoconv at lag k: sum_i x_i * x_{i+k}
        # d/dx_j = x_{j+k} + x_{j-k} (with appropriate bounds)
        
        grad = np.zeros(N)
        for k in range(2 * N - 1):
            if weights[k] < 1e-10:
                continue
            sign_k = np.sign(conv[k]) if conv[k] != 0 else 1.0
            # Contribution to gradient from lag k
            for j in range(N):
                # x_j contributes to conv[k] if j+k < N or j-k >= 0
                contrib = 0.0
                if j + k < N:
                    contrib += current_x[j + k]
                if j - k >= 0:
                    contrib += current_x[j - k]
                grad[j] += 2.0 * sign_k * weights[k] * contrib
        
        # Gradient of 1/s^2 term: -2 * soft_max / s^3
        soft_max_val = _soft_max_abs(conv, beta)
        grad *= -2.0 * soft_max_val / (s ** 3)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad /= grad_norm
        
        # Take step
        x_new = current_x - lr * grad
        
        # Evaluate
        v_new = _metric(x_new)
        if v_new < best_val:
            best_val = v_new
            best_x = x_new.copy()
            current_x = x_new
            patience = 0
            lr = min(lr * 1.1, 0.1)
        else:
            patience += 1
            if patience > 50:
                lr *= 0.5
                patience = 0
                if lr < 1e-6:
                    break
    
    return best_x, best_val


def _try_local_search_fast(x, deadline, rng, max_rounds=200000):
    """Fast local search with in-place modifications and early termination."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.3
    total_steps = 0
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 200:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.0005:
                break
        
        k = int(rng.integers(1, 4))
        indices = rng.choice(N, size=k, replace=False)
        
        # Try multiple perturbations per round
        best_delta = 0
        best_idx = None
        for trial in range(3):
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
            total_steps += 1
        else:
            no_improve += 1
    
    return best_x, best_val


def _try_simulated_annealing(x, deadline, rng, max_rounds=40000):
    """Simulated annealing with adaptive step size."""
    best_x = x.copy()
    best_val = _metric(best_x)
    current_x = x.copy()
    current_val = _metric(current_x)
    
    T_start = 0.03
    T_end = 0.0001
    cooling = (T_end / T_start) ** (1.0 / max_rounds)
    T = T_start
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        k = int(rng.integers(1, 5))
        indices = rng.choice(N, size=k, replace=False)
        x_new = current_x.copy()
        for idx in indices:
            x_new[idx] += rng.normal(0, 0.2)
        
        v_new = _metric(x_new)
        delta = v_new - current_val
        
        if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
            current_x = x_new
            current_val = v_new
            if v_new < best_val:
                best_val = v_new
                best_x = x_new.copy()
        
        T *= cooling
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """High-precision final polish with adaptive coordinate descent and line search."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    # Adaptive coordinate descent with binary search line search
    for sweep in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            # Try to find best step in [-0.1, 0.1] using ternary search
            lo, hi = -0.1, 0.1
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(15):
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

    # Phase 1: Structured near-constant perturbations including Chebyshev (most promising)
    if time.time() < deadline - 70:
        x_nc, val_nc = _try_near_constant_structured(deadline - 70, rng)
        if val_nc < best_val:
            best_val = val_nc
            best_x = x_nc

    # Phase 2: FFT-guided optimization
    if time.time() < deadline - 50:
        x_fft, val_fft = _try_fft_guided_optimization(best_x, deadline - 50, rng)
        if val_fft < best_val:
            best_val = val_fft
            best_x = x_fft

    # Phase 3: Gradient descent with smooth surrogate
    if time.time() < deadline - 30:
        x_grad, val_grad = _try_gradient_descent_smooth(best_x, deadline - 30, rng, max_iters=15000)
        if val_grad < best_val:
            best_val = val_grad
            best_x = x_grad

    # Phase 4: Simulated annealing
    if time.time() < deadline - 15:
        best_x, best_val = _try_simulated_annealing(best_x, deadline - 15, rng, max_rounds=20000)

    # Phase 5: Fast local search
    if time.time() < deadline - 5:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 5, rng, max_rounds=50000)

    # Phase 6: Final adaptive coordinate descent polish
    if time.time() < deadline - 1:
        best_x, best_val = _final_polish(best_x, deadline - 1, rng)

    return best_x
# EVOLVE-BLOCK-END
