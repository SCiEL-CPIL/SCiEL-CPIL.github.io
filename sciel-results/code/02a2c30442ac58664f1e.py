"""
Rewritten program: Spectral-flatness optimization via continuous relaxation.

The task: minimize max_k |c_k(x)| / (sum x_i)^2 for signed step functions of length N.
The constant sequence gives ratio = 1/N (since c_0 = N, sum^2 = N^2).

Key insight from inspiration: use a soft-max surrogate for the off-diagonal max
to get smoother gradients, then do gradient-based descent on the continuous
relaxation. The c_0 term is a hard floor (c_0 = sum x_i^2 >= (sum x_i)^2/N),
so we focus on minimizing the off-diagonal autoconvolution peaks.

Strategy:
1. Normalize so sum x_i = 1 (since the ratio is scale-invariant).
2. Then c_0 = sum x_i^2, and we want max_k |c_k| small for k != 0.
3. Use soft-max to approximate max over |c_k| for k != 0.
4. Gradient descent on the continuous values, then threshold to signs at the end.
5. Use multiple restarts with different initial conditions.
6. Time budget: 150s wall clock, return best found.
"""
import numpy as np
import time

N = 400
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

def _autoconv(x):
    """Compute full autoconvolution of x (length N) -> length 2N-1."""
    # Use FFT for speed
    n = 2 * N
    X = np.fft.fft(x, n=n)
    C = np.fft.ifft(X * X).real[:2*N-1]
    return C

def _soft_max_abs(vals, temperature):
    """Soft approximation of max of |vals| using log-sum-exp."""
    abs_vals = np.abs(vals)
    # Shift for numerical stability
    m = np.max(abs_vals)
    if m < 1e-15:
        return m, abs_vals / max(len(abs_vals), 1)
    log_sum = m + np.log(np.sum(np.exp((abs_vals - m) / temperature)))
    soft_max = temperature * (np.log(np.sum(np.exp((abs_vals - m) / temperature))))
    # Actually let's use: soft_max = T * log(sum(exp(|v_i|/T)))
    # For gradient: d(soft_max)/d(v_i) = sign(v_i) * exp(|v_i|/T) / sum(exp(|v_j|/T))
    exp_vals = np.exp((abs_vals - m) / temperature)
    sum_exp = np.sum(exp_vals)
    soft_max_val = m + temperature * np.log(sum_exp)
    # Gradient w.r.t. v_i: sign(v_i) * exp_vals[i] / sum_exp
    grad = np.sign(vals) * exp_vals / sum_exp
    return soft_max_val, grad

def _objective(x, temperature):
    """
    Objective: c_0 + soft_max(|c_k| for k != 0)
    where c = autoconv(x), and we normalize sum(x)=1.
    Actually, let's work with the ratio directly.
    
    Since ratio = max|c_k| / (sum x_i)^2, and we fix sum x_i = 1,
    ratio = max|c_k|.
    c_0 = sum x_i^2 is always >= 1/N by Cauchy-Schwarz.
    So we minimize max(c_0, max_{k!=0} |c_k|).
    But c_0 is a lower bound, so effectively minimize max_{k!=0} |c_k|
    subject to c_0 being as small as possible (which pushes toward uniform).
    
    Let's use: objective = c_0 + soft_max(|c_k|, k!=0, temperature)
    This encourages both small c_0 and small off-diagonal peaks.
    """
    c = _autoconv(x)
    c0 = c[N-1]  # center of autoconvolution is at index N-1
    off_diag = np.concatenate([c[:N-1], c[N:]])  # exclude center
    sm_val, sm_grad = _soft_max_abs(off_diag, temperature)
    obj = c0 + sm_val
    
    # Gradient: d(c0)/dx_i = 2*x_i (since c0 = sum x_j^2)
    # d(off_diag_k)/dx_i: off_diag_k = sum_j x_j * x_{j+k-N+1} for k in 1..N-1
    # This is complex. Let's use numerical gradient or FFT-based gradient.
    
    return obj, c0, sm_val, off_diag, sm_grad

def _gradient_descent(x0, max_iters=5000, lr_init=0.01, lr_min=1e-6, 
                      temp_init=1.0, temp_min=0.01, patience=2000,
                      deadline=None):
    """
    Gradient descent on continuous relaxation.
    We project to sum(x)=1 at each step.
    """
    x = x0.copy().astype(np.float64)
    # Normalize to sum=1
    s = np.sum(x)
    if abs(s) < 1e-12:
        x = np.ones(N) / N
    else:
        x = x / s
    
    best_obj = np.inf
    best_x = x.copy()
    best_temp = temp_init
    
    temperature = temp_init
    lr = lr_init
    
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
            best_temp = temperature
        
        # Compute gradient
        # d(c0)/dx = 2*x
        grad = 2.0 * x
        
        # d(sm_val)/dx: need to propagate through off_diag
        # off_diag_k for k=1..N-1: c_{k+N-1} and c_{N-1-k}
        # c_m = sum_j x_j * x_{j+m} (with appropriate indexing)
        # For autoconv: c_m = sum_{j} x_j * x_{j+m} where valid indices
        # Actually c = ifft(fft(x)^2), so c_m = sum_j x_j * x_{(j-m) mod ...}
        # Let's use: c_m = sum_j x_j * x_{j+m} for m >= 0 (with x out of range = 0)
        # For m < 0: c_m = sum_j x_j * x_{j+m} = sum_j x_{j-m} * x_j
        
        # Gradient of c_m w.r.t. x_i:
        # c_m = sum_j x_j * x_{j+m}
        # d(c_m)/d(x_i) = x_{i+m} + x_{i-m} (where valid)
        
        # This is O(N^2) per step which is 160000 ops, acceptable.
        # But we can use FFT-based approach:
        # grad_off = 0
        # For each k in 1..N-1:
        #   grad_off += sm_grad[k-1] * d(c_{N-1+k})/dx + sm_grad[N-1+k-1] * d(c_{N-1-k})/dx
        
        # Let's compute this efficiently.
        # c_m for m = -(N-1)..(N-1)
        # off_diag[k] corresponds to c_{N-1+k} for k=0..N-2 (positive lags)
        # and c_{N-1-k} for k=N-1..2N-3 (negative lags)
        
        # Actually off_diag has length 2(N-1) = 2N-2
        # off_diag[0..N-2] = c[0..N-2] (negative lags: m = -(N-1)..-1)
        # off_diag[N-1..2N-3] = c[N..2N-2] (positive lags: m = 1..N-1)
        
        # c_m = sum_j x_j * x_{j+m} for m >= 0
        # c_{-m} = c_m (by symmetry for real signals)
        
        # d(c_m)/dx_i = x_{i+m} + x_{i-m} for m > 0 (with boundary handling)
        
        # Let's build the gradient using convolution.
        # grad_from_off = sum_k w_k * (x_{i+k} + x_{i-k}) for k=1..N-1
        # where w_k = sm_grad[k-1] for positive lag k, and sm_grad[N-1+k-1] for negative
        # By symmetry of autoconv, w_{-k} = w_k, so:
        # grad_from_off_i = 2 * sum_{k=1}^{N-1} w_k * x_{i+k} (with appropriate w)
        
        # Let's define weight vector w of length N-1:
        # w[k-1] for lag k (1..N-1)
        # From off_diag: positive lags are off_diag[N-1..2N-3], corresponding to c[N..2N-2]
        # c[N-1+k] for k=1..N-1 -> off_diag[N-1+k-1] = off_diag[N+k-2]
        # Negative lags: c[N-1-k] for k=1..N-1 -> off_diag[k-1]
        # By symmetry c[N-1-k] = c[N-1+k], so weights should be symmetric.
        
        w_pos = sm_grad[N-1:]  # length N-1, for lags 1..N-1
        w_neg = sm_grad[:N-1]  # length N-1, for lags -(N-1)..-1
        
        # Use the average or just one (they should be similar by symmetry)
        w = 0.5 * (w_pos + w_neg[::-1])  # w[k-1] for lag k
        
        # grad_from_off_i = 2 * sum_{k=1}^{N-1} w[k-1] * x_{i+k} + 2 * sum_{k=1}^{N-1} w[k-1] * x_{i-k}
        # = 2 * (w * x)_circular-ish (with zero padding)
        
        # Use np.convolve or FFT
        # Let's use direct convolution with zero padding
        # Actually, this is a linear operation: grad_from_off = 2 * (w convolved with x, appropriately)
        
        # For efficiency, use FFT:
        n_fft = 2 * N
        W = np.fft.fft(w, n=n_fft)
        X = np.fft.fft(x, n=n_fft)
        # conv: y_i = sum_k w[k] * x[i-k] (cross-correlation style)
        # grad_from_off_i = 2 * sum_{k=1}^{N-1} w[k-1] * (x[i+k] + x[i-k])
        # = 2 * sum_k w[k-1] * x[i+k] + 2 * sum_k w[k-1] * x[i-k]
        
        # sum_k w[k-1] * x[i+k] = (w reversed) convolved with x, shifted
        # Let's just do it directly for N=400, it's 400*400 = 160k ops, fine.
        
        grad_off = np.zeros(N)
        # For each lag k (1 to N-1):
        #   grad_off[i] += 2 * w[k-1] * (x[i+k] if i+k < N else 0 + x[i-k] if i-k >= 0 else 0)
        for k in range(1, N):
            wk = 2.0 * w[k-1]
            # x[i+k] for i=0..N-k-1
            grad_off[:N-k] += wk * x[k:]
            # x[i-k] for i=k..N-1
            grad_off[k:] += wk * x[:N-k]
        
        grad += grad_off
        
        # Project gradient to maintain sum(x)=1: remove mean component
        grad -= np.mean(grad)
        
        # Update
        x_new = x - lr * grad
        # Project to sum=1
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x  # revert
        else:
            x_new = x_new / s_new
        
        # Check if we improved
        x = x_new
        
        # Anneal
        if it > 0 and it % 100 == 0:
            temperature = max(temp_min, temperature * 0.95)
            lr = max(lr_min, lr * 0.99)
        
        # Patience check
        if obj > best_obj * 1.001 and it > 500:
            # No improvement for a while
            pass  # Continue anyway, annealing helps
    
    return best_x, best_obj

def construct():
    """
    Return signed step heights that minimize max|c_k| / (sum x_i)^2.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_result = np.ones(N)  # Default: constant sequence, score = 1.0 baseline
    best_score = 1.0  # This is the ratio for ones(N): max|c_k|/sum^2 = N/N^2 = 1/N... 
    # Wait, let me recalculate. For x = ones(N):
    # sum x_i = N
    # c_k: autoconvolution. c_{N-1} = N (center). c_{N-1+k} = N-k for k=0..N-1.
    # max|c_k| = N
    # ratio = N / N^2 = 1/N
    # So the score (raw value) is 1/N = 0.0025
    # The "score" in the search is seed/value, so if seed program gives value V_seed,
    # and we give value V, our search score = V_seed / V.
    # The current program scores 1.0000, meaning it achieves the seed value.
    # So the raw value for ones(N) is what the seed gives.
    
    # Let's compute the actual raw value for ones(N):
    c = _autoconv(np.ones(N))
    raw_val_ones = np.max(np.abs(c)) / (np.sum(np.ones(N))**2)
    # = N / N^2 = 1/N
    
    # We want to minimize raw_val. Let's try to do better.
    
    # Strategy: Use gradient descent with multiple restarts
    np.random.seed(42)
    
    num_restarts = 10
    max_iters_per_restart = 3000
    
    # Pre-compute some things
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        if r == 0:
            x0 = np.ones(N) + 0.1 * np.random.randn(N)
        elif r == 1:
            # Random signs
            x0 = np.random.choice([-1, 1], size=N).astype(np.float64)
        elif r == 2:
            # M-sequence like (pseudo-random binary)
            x0 = np.random.choice([-1, 1], size=N).astype(np.float64)
            x0[0] = 1.0
        else:
            # Smooth random
            x0 = np.random.randn(N)
            # Apply a low-pass filter
            from numpy.fft import fft, ifft
            X = fft(x0)
            mask = np.ones(N)
            mask[N//4:] = 0
            mask[:N//4] = 0
            mask[N//8:N//4] = 1
            X *= mask
            x0 = ifft(X).real
        
        x0 = x0 / np.sum(x0) if abs(np.sum(x0)) > 1e-12 else np.ones(N)/N
        
        # Run gradient descent
        x_best, obj_val = _gradient_descent(
            x0, 
            max_iters=max_iters_per_restart,
            lr_init=0.01,
            lr_min=1e-6,
            temp_init=1.0,
            temp_min=0.01,
            patience=2000,
            deadline=deadline
        )
        
        # Evaluate the actual ratio for this solution
        c = _autoconv(x_best)
        s = np.sum(x_best)
        if abs(s) < 1e-12:
            continue
        raw_val = np.max(np.abs(c)) / (s**2)
        
        if raw_val < raw_val_ones:
            best_result = x_best
            raw_val_ones = raw_val
    
    # Now, try to threshold to signs and see if that helps
    # The task says "signed step functions" which might mean values in {-1, 1} or just real numbers
    # Let's check both continuous and sign-thresholded versions
    
    # Threshold to signs
    if np.any(best_result != 0):
        sign_result = np.sign(best_result)
        # Handle zeros
        sign_result[sign_result == 0] = 1
        c_sign = _autoconv(sign_result)
        s_sign = np.sum(sign_result)
        if abs(s_sign) > 1e-12:
            raw_val_sign = np.max(np.abs(c_sign)) / (s_sign**2)
            if raw_val_sign < raw_val_ones:
                best_result = sign_result
                raw_val_ones = raw_val_sign
    
    # Final check: make sure we return something valid
    s_final = np.sum(best_result)
    if abs(s_final) < 1e-12:
        return np.ones(N)
    
    return best_result

# EVOLVE-BLOCK-END
