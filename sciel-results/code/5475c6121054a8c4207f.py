"""
Signed autocorrelation max-abs minimization via spectral flatness gradient descent.

The key insight: max_k |c_k| for k != 0 is bounded by the variance of the
autoconvolution, which relates to the fourth power of the DFT magnitude.
Minimizing the peak of the autoconvolution (excluding lag 0) is equivalent to
making the spectral magnitude as flat as possible (Barker-like sequences).

We use a soft-max surrogate on the off-diagonal autoconvolution coefficients,
with gradient descent in the continuous space, then threshold to signs.
"""
import numpy as np
import time

N = 400
DEADLINE = 150.0
SAFETY_MARGIN = 5.0


def _autoconv(x):
    """Compute full autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N
    X = np.fft.fft(x, n=n)
    C = np.fft.ifft(X * X).real[:2 * N - 1]
    return C


def _score(x):
    """Compute the raw objective: max|c_k| / (sum x_i)^2."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(c)) / (s ** 2)


def _soft_max_abs(vals, temperature):
    """
    Soft approximation of max(|vals|) with gradient.
    Returns (value, gradient w.r.t. vals).
    """
    abs_vals = np.abs(vals)
    m = np.max(abs_vals)
    if m < 1e-15:
        return m, np.zeros_like(vals)
    scaled = (abs_vals - m) / temperature
    # Clip to avoid overflow
    scaled = np.clip(scaled, -50, 50)
    exp_vals = np.exp(scaled)
    sum_exp = np.sum(exp_vals)
    if sum_exp < 1e-30:
        return m, np.zeros_like(vals)
    soft_max_val = m + temperature * np.log(sum_exp)
    grad = np.sign(vals) * exp_vals / sum_exp
    return soft_max_val, grad


def _objective_and_grad(x, temperature):
    """
    Compute the soft-max objective on off-diagonal autoconvolution
    and its gradient w.r.t. x.
    
    Objective: sum_k w_k * |c_k| for k != N-1 (off-diagonal),
    where w_k are softmax weights.
    
    Gradient: dc_k/dx_i = x_{i+k} + x_{i-k} (for valid indices)
    So d(obj)/dx_i = sum_{k != 0} w_k * sign(c_k) * (x_{i+k} + x_{i-k})
    """
    c = _autoconv(x)
    # c has length 2N-1, center at index N-1
    # Off-diagonal: indices 0..N-2 and N..2N-2
    off_diag = np.concatenate([c[:N - 1], c[N:]])
    
    sm_val, sm_grad = _soft_max_abs(off_diag, temperature)
    
    # sm_grad has length 2*(N-1)
    # First N-1 entries correspond to negative lags (c[0]..c[N-2])
    # Last N-1 entries correspond to positive lags (c[N]..c[2N-2])
    w_neg = sm_grad[:N - 1]   # weights for c[0]..c[N-2]
    w_pos = sm_grad[N - 1:]   # weights for c[N]..c[2N-2]
    
    # By symmetry of autoconvolution, c[k] = c[2N-2-k], so we can average
    # But let's just compute the gradient directly
    # 
    # d(c[k])/dx[i] = x[i+k-(N-1)] + x[i-k+(N-1)] for valid indices
    # Actually, c[k] = sum_j x[j] * x[j + k - (N-1)]
    # So dc[k]/dx[i] = x[i + k - (N-1)] + x[i - k + (N-1)] (when both in range)
    
    # Let's use the fact that c = x * x (linear convolution)
    # c[k] = sum_{j} x[j] * x[j - k + (N-1)]  ... this is getting complex
    # 
    # Simpler: use the derivative of convolution
    # If c = x * x, then dc/dx[i] = x convolved with delta at position i
    # which gives: (dc/dx[i])[k] = x[k-i] + x[i-k] (with appropriate shifts)
    
    # Let's just do it with the weight vectors
    # For positive lag l (1..N-1): c[N-1+l] = sum_j x[j]*x[j+l]
    # dc[N-1+l]/dx[i] = x[i+l] + x[i-l] (when valid)
    
    # For negative lag -l: c[N-1-l] = sum_j x[j]*x[j-l] = c[N-1+l] by symmetry
    
    # So we can combine: for lag l=1..N-1, the effective weight is
    # w_pos[l-1] + w_neg[N-1-l] (since c[N-1-l] = c[N-1+l])
    
    # Actually w_neg[N-1-l] corresponds to c[N-1-l], and by symmetry c[N-1-l]=c[N-1+l]
    # So the combined weight for |c[N-1+l]| is w_pos[l-1] + w_neg[N-1-l]
    
    # But wait, the gradient of |c| w.r.t. c is sign(c), and the soft-max
    # already incorporates this. Let me just compute directly.
    
    grad = np.zeros(N)
    
    # For each lag l = 1..N-1:
    # Contribution from c[N-1+l]: weight w_pos[l-1]
    # dc[N-1+l]/dx[i] = x[i+l] + x[i-l] (for valid i)
    for l in range(1, N):
        w = w_pos[l - 1]
        if abs(w) < 1e-15:
            continue
        # x[i+l] for i=0..N-l-1
        if N - l > 0:
            grad[:N - l] += w * x[l:]
        # x[i-l] for i=l..N-1
        if N - l > 0:
            grad[l:] += w * x[:N - l]
    
    # Contribution from c[N-1-l] for l=1..N-1: weight w_neg[N-1-l]
    # But c[N-1-l] = c[N-1+l], so this is the same lag
    # Actually no: w_neg[j] is the gradient w.r.t. c[j] for j=0..N-2
    # c[N-1-l] = c[j] where j = N-1-l, so l = N-1-j
    # This is the same as positive lag l, so we should add w_neg[N-1-l]
    
    for l in range(1, N):
        w = w_neg[N - 1 - l]
        if abs(w) < 1e-15:
            continue
        if N - l > 0:
            grad[:N - l] += w * x[l:]
            grad[l:] += w * x[:N - l]
    
    # Project gradient to maintain sum(x) = 1
    grad -= np.mean(grad)
    
    return sm_val, grad


def _spectral_objective(x):
    """
    Alternative: minimize the variance of |X|^2 (spectral flatness).
    This is a proxy for minimizing the peak of the autoconvolution.
    """
    X = np.fft.fft(x)
    power = np.abs(X) ** 2
    mean_power = np.mean(power)
    if mean_power < 1e-15:
        return 0.0, np.zeros(N)
    # Variance of power spectrum
    var_power = np.mean((power - mean_power) ** 2)
    # Gradient of var_power w.r.t. x
    # This is more complex; use the autoconv approach instead
    return var_power, np.zeros(N)


def _gradient_descent(x0, max_iters=2000, lr_init=0.02, lr_min=1e-8,
                      temp_init=1.0, temp_min=0.01, deadline=None):
    """
    Gradient descent with adaptive learning rate and temperature annealing.
    """
    x = x0.copy().astype(np.float64)
    s = np.sum(x)
    if abs(s) < 1e-12:
        x = np.ones(N) / N
    else:
        x = x / s
    
    best_obj = np.inf
    best_x = x.copy()
    
    temperature = temp_init
    lr = lr_init
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        obj, grad = _objective_and_grad(x, temperature)
        
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
        
        # Adaptive step
        x_new = x - lr * grad
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x
        else:
            x_new = x_new / s_new
        
        # Check for divergence
        obj_new, _ = _objective_and_grad(x_new, temperature)
        
        if obj_new > obj * 2.0 + 1e-10:
            lr = max(lr_min, lr * 0.5)
        else:
            x = x_new
            if obj_new < obj * 0.999:
                lr = min(lr_init, lr * 1.02)
        
        # Anneal temperature
        if it % 30 == 0 and it > 0:
            temperature = max(temp_min, temperature * 0.95)
            lr = max(lr_min, lr * 0.99)
        
        # Early stopping
        if it > 100 and obj > best_obj * 1.001:
            if it % 50 == 0:
                # Check if stuck
                pass
    
    return best_x, best_obj


def _smooth_init(seed=None):
    """Generate a smooth random initialization."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    x = rng.randn(N)
    X = np.fft.fft(x)
    mask = np.zeros(N)
    low = max(1, N // 16)
    high = N // 2
    mask[low:high] = 1
    mask[N - high:N - low] = 1
    X *= mask
    return np.fft.ifft(X).real


def _banded_init(seed=None):
    """Generate a banded initialization."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    x = np.zeros(N)
    num_bands = rng.randint(3, 8)
    band_widths = rng.randint(10, 40, size=num_bands)
    positions = rng.choice(N, size=num_bands, replace=False)
    for pos, width in zip(positions, band_widths):
        x[pos:pos + width] = rng.choice([-1, 1])
    return x


def _block_init(seed=None):
    """Generate a block-pattern initialization."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    x = np.zeros(N)
    num_blocks = rng.randint(4, 12)
    block_size = N // num_blocks
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size if i < num_blocks - 1 else N
        x[start:end] = rng.choice([-1, 1])
    return x


def _fourier_init(seed=None):
    """Generate initialization from random Fourier coefficients."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    X = np.zeros(N, dtype=complex)
    num_freqs = rng.randint(5, 20)
    freqs = rng.choice(N, size=num_freqs, replace=False)
    for f in freqs:
        X[f] = rng.randn() + 1j * rng.randn()
        X[N - f] = np.conj(X[f])
    return np.fft.ifft(X).real


def construct():
    """
    Return signed step heights that minimize max|c_k| / (sum x_i)^2.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    # Baseline: constant sequence
    baseline = np.ones(N)
    best_result = baseline.copy()
    best_val = _score(baseline)

    np.random.seed(42)

    # Multiple restarts with diverse initializations
    restart_configs = [
        ("perturbed_ones", lambda s: np.ones(N) + 0.1 * np.random.RandomState(s).randn(N)),
        ("random_binary", lambda s: np.random.RandomState(s).choice([-1, 1], size=N).astype(np.float64)),
        ("smooth_random", lambda s: _smooth_init(seed=s)),
        ("banded", lambda s: _banded_init(seed=s)),
        ("alternating", lambda s: np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])),
        ("block_pattern", lambda s: _block_init(seed=s)),
        ("fourier_random", lambda s: _fourier_init(seed=s)),
    ]

    max_iters = 1500

    for r, (name, init_fn) in enumerate(restart_configs):
        if time.time() > deadline:
            break

        x0 = init_fn(42 + r * 1000)
        s = np.sum(x0)
        if abs(s) < 1e-12:
            x0 = np.ones(N) / N
        else:
            x0 = x0 / s

        x_best, obj_val = _gradient_descent(
            x0,
            max_iters=max_iters,
            lr_init=0.02,
            lr_min=1e-8,
            temp_init=1.0,
            temp_min=0.01,
            deadline=deadline
        )

        raw_val = _score(x_best)

        if raw_val < best_val:
            best_result = x_best.copy()
            best_val = raw_val

        # Also try sign-thresholded version
        sign_result = np.sign(x_best)
        sign_result[sign_result == 0] = 1
        raw_val_sign = _score(sign_result)
        if raw_val_sign < best_val:
            best_result = sign_result
            best_val = raw_val_sign

    # Final: try more iterations on the best result found
    if time.time() < deadline - 10:
        x_final, _ = _gradient_descent(
            best_result.copy(),
            max_iters=3000,
            lr_init=0.01,
            lr_min=1e-9,
            temp_init=0.5,
            temp_min=0.005,
            deadline=deadline
        )
        raw_val_final = _score(x_final)
        if raw_val_final < best_val:
            best_result = x_final
            best_val = raw_val_final

        sign_final = np.sign(x_final)
        sign_final[sign_final == 0] = 1
        raw_val_sign_final = _score(sign_final)
        if raw_val_sign_final < best_val:
            best_result = sign_final
            best_val = raw_val_sign_final

    # Ensure valid output
    s_final = np.sum(best_result)
    if abs(s_final) < 1e-12:
        return np.ones(N)

    return best_result
