"""
Optimized spectral-flatness minimization for signed step functions.

Minimize max_k |c_k(x)| / (sum x_i)^2 where c = x * x (autoconvolution).
Uses FFT-based gradient computation for speed, with adaptive temperature
annealing and diverse restarts.
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
    C = np.fft.ifft(X * X).real[:2*N-1]
    return C

def _soft_max_abs(vals, temperature):
    """Soft approximation of max of |vals| with gradient."""
    abs_vals = np.abs(vals)
    m = np.max(abs_vals)
    if m < 1e-15:
        return m, np.zeros_like(vals)
    exp_vals = np.exp((abs_vals - m) / temperature)
    sum_exp = np.sum(exp_vals)
    soft_max_val = m + temperature * np.log(sum_exp)
    grad = np.sign(vals) * exp_vals / sum_exp
    return soft_max_val, grad

def _compute_gradient_fft(x, sm_grad, temperature):
    """
    Compute gradient of (c0 + soft_max(off_diag)) w.r.t. x using FFT.
    
    c0 = sum x_i^2, so d(c0)/dx_i = 2*x_i
    
    For off-diagonal terms, by symmetry of autoconvolution:
    grad_off_i = 2 * sum_{k=1}^{N-1} w_k * (x_{i+k} + x_{i-k})
    where w_k are the soft-max weights for lag k.
    
    This can be computed as a convolution: grad_off = 2 * (w * x) with appropriate
    zero-padding and shifting.
    """
    # Build weight vector for lags 1..N-1
    # sm_grad has length 2*(N-1): first N-1 are negative lags, last N-1 are positive
    w_pos = sm_grad[N-1:]   # lags 1..N-1
    w_neg = sm_grad[:N-1]   # lags -(N-1)..-1
    
    # By symmetry, average them
    w = 0.5 * (w_pos + w_neg[::-1])  # w[k-1] for lag k
    
    # Compute grad_off_i = 2 * sum_{k=1}^{N-1} w[k-1] * (x[i+k] + x[i-k])
    # = 2 * [sum_{k=1}^{N-1} w[k-1] * x[i+k]] + 2 * [sum_{k=1}^{N-1} w[k-1] * x[i-k]]
    
    # Use FFT for efficient convolution
    n_fft = 2 * N
    W = np.fft.fft(w, n=n_fft)
    X = np.fft.fft(x, n=n_fft)
    
    # First term: sum_k w[k-1] * x[i+k]
    # This is like (w_reversed conv x)[i] with zero padding
    # Let's compute: y1[i] = sum_{k=1}^{N-1} w[k-1] * x[i+k]
    # = sum_{j} w[j] * x[i+1+j] for j=0..N-2
    # This is cross-correlation style.
    
    # Alternative: just use direct computation but vectorized
    # For N=400, O(N^2) = 160k ops is actually fine
    grad_off = np.zeros(N)
    
    # Vectorized computation using broadcasting
    for k in range(1, N):
        wk = 2.0 * w[k-1]
        if wk == 0.0:
            continue
        # x[i+k] for i=0..N-k-1
        if N-k > 0:
            grad_off[:N-k] += wk * x[k:]
        # x[i-k] for i=k..N-1
        if N-k > 0:
            grad_off[k:] += wk * x[:N-k]
    
    # Total gradient: 2*x + grad_off
    grad = 2.0 * x + grad_off
    
    # Project to maintain sum(x)=1: remove mean
    grad -= np.mean(grad)
    
    return grad

def _gradient_descent(x0, max_iters=5000, lr_init=0.05, lr_min=1e-7,
                      temp_init=2.0, temp_min=0.05, deadline=None):
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
    best_obj_history = []
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        c = _autoconv(x)
        c0 = c[N-1]
        off_diag = np.concatenate([c[:N-1], c[N:]])
        
        sm_val, sm_grad = _soft_max_abs(off_diag, temperature)
        obj = c0 + sm_val
        
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
        
        # Compute gradient using FFT-based approach
        grad = _compute_gradient_fft(x, sm_grad, temperature)
        
        # Adaptive learning rate: reduce if loss increasing
        x_new = x - lr * grad
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x
        else:
            x_new = x_new / s_new
        
        # Check for divergence
        c_new = _autoconv(x_new)
        c0_new = c_new[N-1]
        off_diag_new = np.concatenate([c_new[:N-1], c_new[N:]])
        sm_val_new, _ = _soft_max_abs(off_diag_new, temperature)
        obj_new = c0_new + sm_val_new
        
        if obj_new > obj * 1.5:
            # Too large step, reduce lr
            lr = max(lr_min, lr * 0.5)
            x = x  # revert
        else:
            x = x_new
            # Slightly increase lr if improving
            if obj_new < obj * 0.99:
                lr = min(lr_init, lr * 1.05)
        
        # Anneal temperature
        if it % 50 == 0 and it > 0:
            temperature = max(temp_min, temperature * 0.92)
            lr = max(lr_min, lr * 0.98)
        
        # Early stopping if converged
        best_obj_history.append(obj)
        if len(best_obj_history) > 50:
            if best_obj_history[-1] > min(best_obj_history[-50:]) * 1.01:
                if it > 200:
                    break
    
    return best_x, best_obj

def _evaluate(x):
    """Compute the raw objective value."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(c)) / (s**2)

def construct():
    """
    Return signed step heights that minimize max|c_k| / (sum x_i)^2.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Baseline: constant sequence
    baseline = np.ones(N)
    best_result = baseline.copy()
    best_val = _evaluate(baseline)
    
    np.random.seed(42)
    
    # Multiple restarts with diverse initializations
    restart_configs = [
        # (name, initializer)
        ("perturbed_ones", lambda: np.ones(N) + 0.1 * np.random.randn(N)),
        ("random_binary", lambda: np.random.choice([-1, 1], size=N).astype(np.float64)),
        ("random_binary_1", lambda: np.concatenate(([1.0], np.random.choice([-1, 1], size=N-1).astype(np.float64)))),
        ("smooth_random", lambda: _smooth_init()),
        ("banded", lambda: _banded_init()),
        ("alternating", lambda: np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])),
        ("block_pattern", lambda: _block_init()),
        ("fourier_random", lambda: _fourier_init()),
    ]
    
    max_iters = 3000
    
    for r, (name, init_fn) in enumerate(restart_configs):
        if time.time() > deadline:
            break
        
        x0 = init_fn()
        s = np.sum(x0)
        if abs(s) < 1e-12:
            x0 = np.ones(N) / N
        else:
            x0 = x0 / s
        
        x_best, obj_val = _gradient_descent(
            x0,
            max_iters=max_iters,
            lr_init=0.05,
            lr_min=1e-7,
            temp_init=2.0,
            temp_min=0.05,
            deadline=deadline
        )
        
        raw_val = _evaluate(x_best)
        
        if raw_val < best_val:
            best_result = x_best
            best_val = raw_val
        
        # Also try sign-thresholded version
        sign_result = np.sign(x_best)
        sign_result[sign_result == 0] = 1
        raw_val_sign = _evaluate(sign_result)
        if raw_val_sign < best_val:
            best_result = sign_result
            best_val = raw_val_sign
    
    # Final: try more iterations on the best result found
    if time.time() < deadline - 10:
        x_final, _ = _gradient_descent(
            best_result.copy(),
            max_iters=5000,
            lr_init=0.02,
            lr_min=1e-8,
            temp_init=1.0,
            temp_min=0.02,
            deadline=deadline
        )
        raw_val_final = _evaluate(x_final)
        if raw_val_final < best_val:
            best_result = x_final
            best_val = raw_val_final
        
        sign_final = np.sign(x_final)
        sign_final[sign_final == 0] = 1
        raw_val_sign_final = _evaluate(sign_final)
        if raw_val_sign_final < best_val:
            best_result = sign_final
            best_val = raw_val_sign_final
    
    # Ensure valid output
    s_final = np.sum(best_result)
    if abs(s_final) < 1e-12:
        return np.ones(N)
    
    return best_result

def _smooth_init():
    """Generate a smooth random initialization."""
    x = np.random.randn(N)
    X = np.fft.fft(x)
    # Band-pass filter
    mask = np.zeros(N)
    low = N // 8
    high = N // 2
    mask[low:high] = 1
    mask[N-high:N-low] = 1
    X *= mask
    return np.fft.ifft(X).real

def _banded_init():
    """Generate a banded initialization."""
    x = np.zeros(N)
    num_bands = np.random.randint(3, 8)
    band_widths = np.random.randint(10, 40, size=num_bands)
    positions = np.random.choice(N, size=num_bands, replace=False)
    for pos, width in zip(positions, band_widths):
        x[pos:pos+width] = np.random.choice([-1, 1])
    return x

def _block_init():
    """Generate a block-pattern initialization."""
    x = np.zeros(N)
    num_blocks = np.random.randint(4, 12)
    block_size = N // num_blocks
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size if i < num_blocks - 1 else N
        x[start:end] = np.random.choice([-1, 1])
    return x

def _fourier_init():
    """Generate initialization from random Fourier coefficients."""
    X = np.zeros(N, dtype=complex)
    num_freqs = np.random.randint(5, 20)
    freqs = np.random.choice(N, size=num_freqs, replace=False)
    for f in freqs:
        X[f] = np.random.randn() + 1j * np.random.randn()
        X[N-f] = np.conj(X[f])  # Ensure real signal
    return np.fft.ifft(X).real

# EVOLVE-BLOCK-END
