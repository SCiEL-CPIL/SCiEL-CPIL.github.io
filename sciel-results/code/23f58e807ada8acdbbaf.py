"""Rewritten program: projected gradient descent with log-sum-exp surrogate for C3 bound."""
import numpy as np
import time

N = 400
TIME_LIMIT = 110.0  # safety margin before 120s kill

# Precompute autocorrelation lag indices (non-zero lags for sidelobe)
# For a step function on 400 intervals, the autoconvolution at lag k
# involves sum of x[i]*x[i+k] for valid i.
# C3 bound is related to the peak of the autoconvolution sidelobes.

def _compute_c3_bound(x):
    """
    Compute the C3 upper bound from a signed step function x of length N.
    
    The autoconvolution of x with itself: (x * x)[k] = sum_i x[i] * x[i+k]
    The C3 bound involves the maximum sidelobe of the autoconvolution.
    
    For a ±1 sequence, the autoconvolution at lag 0 is N (all ones product).
    Sidelobes are at lags k != 0.
    
    The C3 inequality bound is typically:
    C3 <= max_{k != 0} |sum_i x[i] x[i+k]|
    
    This is the peak sidelobe of the autocorrelation.
    """
    # Use FFT-based autocorrelation for efficiency
    n = len(x)
    # Zero-pad to avoid circular convolution issues
    size = 1
    while size < 2 * n:
        size *= 2
    X = np.fft.rfft(x, size)
    acf = np.fft.irfft(X * np.conj(X), size)[:n]
    # acf[k] for k=0..n-1 gives sum_i x[i]*x[i+k] (with appropriate indexing)
    # Actually, irfft of |X|^2 gives circular autocorrelation.
    # For linear autocorrelation with zero-padding, acf[k] = sum_{i=0}^{n-1-k} x[i]*x[i+k]
    # The peak sidelobe is max of |acf[k]| for k in 1..n-1
    sidelobes = np.abs(acf[1:])
    return np.max(sidelobes) if len(sidelobes) > 0 else 0.0


def _compute_c3_bound_fast(x):
    """Faster computation using direct correlation for small lags, or full FFT."""
    n = len(x)
    size = 1
    while size < 2 * n:
        size *= 2
    X = np.fft.rfft(x, size)
    acf = np.fft.irfft(X * np.conj(X), size)[:n]
    sidelobes = np.abs(acf[1:])
    return np.max(sidelobes) if len(sidelobes) > 0 else 0.0


def _log_sum_exp_surrogate(x, beta=50.0):
    """
    Smooth surrogate for the peak sidelobe using log-sum-exp.
    f(x) = (1/beta) * log(sum_k exp(beta * |acf[k]|))
    This approximates max_k |acf[k]| as beta -> infinity.
    """
    n = len(x)
    size = 1
    while size < 2 * n:
        size *= 2
    X = np.fft.rfft(x, size)
    acf = np.fft.irfft(X * np.conj(X), size)[:n]
    sidelobes = np.abs(acf[1:])
    if len(sidelobes) == 0:
        return 0.0
    # log-sum-exp for numerical stability
    m = np.max(sidelobes)
    return m + np.log(np.sum(np.exp(beta * (sidelobes - m)))) / beta


def _gradient_c3(x, beta=50.0):
    """
    Compute gradient of the log-sum-exp surrogate with respect to x.
    
    The autoconvolution acf[k] = sum_i x[i] * x[i+k]
    d(acf[k])/dx[j] = x[j+k] (if j+k < n) + x[j-k] (if j-k >= 0)
    
    For the surrogate L = (1/beta) * log(sum_k exp(beta * acf[k]))  [using acf not |acf| for simplicity]
    Actually we want to minimize the peak of |acf[k]|. Let's use a different approach.
    
    Let's define f(x) = (1/beta) * log(sum_{k=1}^{n-1} exp(beta * acf[k]^2 / (2 * max_val)))
    or more simply, use the squared version for differentiability.
    
    Actually, let's use: f(x) = (1/beta) * log(sum_k exp(beta * acf[k]))
    But acf[k] can be negative. We want to minimize the peak absolute value.
    
    Better: use f(x) = (1/beta) * log(sum_k exp(beta * |acf[k]|))
    The gradient of |acf[k]| w.r.t. x[j] is sign(acf[k]) * d(acf[k])/dx[j]
    
    d(acf[k])/dx[j] = x[(j+k) % n] ... no, for linear correlation:
    acf[k] = sum_{i=0}^{n-1-k} x[i] * x[i+k]
    d(acf[k])/dx[j] = x[j+k] (if j+k < n, i.e., j < n-k) + x[j-k] (if j-k >= 0, i.e., j >= k)
    
    This is getting complex. Let me use a simpler approach:
    Just compute the gradient numerically or use a simpler surrogate.
    """
    n = len(x)
    size = 1
    while size < 2 * n:
        size *= 2
    X = np.fft.rfft(x, size)
    acf = np.fft.irfft(X * np.conj(X), size)[:n]
    
    # Use a smoothed version: minimize sum of exp(beta * acf[k]^2) for k=1..n-1
    # This is differentiable and approximates the peak when beta is large.
    sidelobes_sq = acf[1:] ** 2
    m = np.max(sidelobes_sq) if len(sidelobes_sq) > 0 else 0
    if m == 0:
        return np.zeros_like(x)
    
    # weights = exp(beta * sidelobes_sq / m) / sum(exp(...))
    # To avoid overflow, use: w_k = exp(beta * (sidelobes_sq[k] - m)) / sum(exp(beta * (sidelobes_sq[j] - m)))
    beta_eff = beta
    shifted = beta_eff * (sidelobes_sq - m)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    # Gradient of sum_k w_k * acf[k]^2 w.r.t. x[j]
    # d(acf[k]^2)/dx[j] = 2 * acf[k] * d(acf[k])/dx[j]
    # d(acf[k])/dx[j] = x[j+k] (if j < n-k) + x[j-k] (if j >= k)
    
    grad = np.zeros(n)
    for k in range(1, n):
        if weights[k-1] < 1e-10:
            continue
        # d(acf[k])/dx[j] for j=0..n-1
        # For j < n-k: d(acf[k])/dx[j] includes x[j+k]
        # For j >= k: d(acf[k])/dx[j] includes x[j-k]
        # So d(acf[k])/dx[j] = (x[j+k] if j < n-k else 0) + (x[j-k] if j >= k else 0)
        
        # Vectorized:
        # Part 1: j from 0 to n-1-k, contribution x[j+k]
        if n - k > 0:
            grad[0:n-k] += 2 * weights[k-1] * acf[k] * x[k:n]
        # Part 2: j from k to n-1, contribution x[j-k]
        if k < n:
            grad[k:n] += 2 * weights[k-1] * acf[k] * x[0:n-k]
    
    # Scale by the derivative of the log-sum-exp
    # f = (1/beta) * log(sum exp(beta * acf[k]^2))
    # df/dx = (1/beta) * sum_k exp(beta*acf[k]^2) * 2*beta*acf[k] * d(acf[k])/dx / sum exp(...)
    #       = sum_k w_k * 2 * acf[k] * d(acf[k])/dx
    # So grad is already computed correctly above (with the 2*acf[k] factor)
    
    return grad


def _project_to_sign(x):
    """Project onto the sign constraint: map to ±1."""
    return np.sign(x)
    # For values exactly 0, np.sign returns 0. Let's handle that.
    # Actually, let's use a soft projection during optimization and hard at the end.


def _projected_gradient_descent(x_init, max_iters=5000, lr=0.01, beta=20.0, time_limit=110.0):
    """
    Projected gradient descent on the continuous relaxation, then project to ±1.
    """
    start_time = time.time()
    x = x_init.astype(np.float64).copy()
    best_x = x.copy()
    best_val = _compute_c3_bound(_project_to_sign(x))
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Compute gradient of the smooth surrogate
        grad = _gradient_c3(x, beta=beta)
        
        # Gradient step
        x = x - lr * grad
        
        # Project to ±1
        x_sign = _project_to_sign(x)
        
        # Evaluate
        val = _compute_c3_bound(x_sign)
        if val < best_val:
            best_val = val
            best_x = x_sign.copy()
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            # Reduce learning rate slightly
            lr *= 0.95
    
    return best_x, best_val


def _local_search(x, max_iters=10000, time_limit=110.0):
    """
    Local search: try flipping individual bits and pairs to reduce C3 bound.
    """
    start_time = time.time()
    x = x.copy()
    best_val = _compute_c3_bound(x)
    
    n = len(x)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Try flipping bit i
            x[i] *= -1
            val = _compute_c3_bound(x)
            if val < best_val:
                best_val = val
                improved = True
            else:
                x[i] *= -1  # revert
        
        if time.time() - start_time > time_limit - 5.0:
            break
    
    return x, best_val


def _simulated_annealing(x_init, time_limit=110.0, T_start=10.0, T_end=0.1, max_iters=50000):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    x = x_init.copy()
    best_x = x.copy()
    best_val = _compute_c3_bound(x)
    current_val = best_val
    
    n = len(x)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Temperature schedule
        progress = it / max_iters
        T = T_start * (T_end / T_start) ** progress
        
        # Random bit flip
        i = np.random.randint(n)
        x[i] *= -1
        new_val = _compute_c3_bound(x)
        
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[i] *= -1  # revert
    
    return best_x, best_val


def construct():
    """
    Construct a signed step function (±1 sequence of length N) that minimizes
    the C3 autoconvolution upper bound.
    """
    start_time = time.time()
    
    # Initialize with random ±1 sequence
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # Phase 1: Projected gradient descent on continuous relaxation
    x_pgd, val_pgd = _projected_gradient_descent(x, max_iters=3000, lr=0.05, beta=15.0, time_limit=TIME_LIMIT)
    
    # Phase 2: Local search polish
    x_ls, val_ls = _local_search(x_pgd, max_iters=20000, time_limit=TIME_LIMIT)
    
    # Phase 3: Simulated annealing to escape local minima
    x_sa, val_sa = _simulated_annealing(x_ls, time_limit=TIME_LIMIT, T_start=5.0, T_end=0.05, max_iters=30000)
    
    # Phase 4: Final local search
    x_final, val_final = _local_search(x_sa, max_iters=20000, time_limit=TIME_LIMIT)
    
    # Ensure we return a valid ±1 sequence
    x_final = np.sign(x_final)
    # Handle any zeros (shouldn't happen but just in case)
    x_final[x_final == 0] = 1.0
    
    # Check time - if we're close to the limit, return immediately
    if time.time() - start_time > TIME_LIMIT - 2.0:
        return x_final
    
    return x_final
