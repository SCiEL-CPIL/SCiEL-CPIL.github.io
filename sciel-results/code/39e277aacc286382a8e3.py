"""Improved program: use continuous optimization with gradient descent on autocorrelation metric."""
N = 400
import numpy as np
import time

def compute_autocorrelation_bound(f):
    """Compute the upper bound on C3: L2 norm of autocorrelation sidelobes."""
    nfft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N]
    # Use L2 norm of sidelobes as the metric to minimize
    sidelobes = acf[1:]
    bound = np.sqrt(np.sum(sidelobes ** 2))
    return bound

def compute_gradient(f):
    """Compute gradient of the L2 sidelobe norm w.r.t. f."""
    nfft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N]
    # Gradient of sum(acf[k]^2) for k>=1 w.r.t. f
    # acf = ifft(|F|^2), so d(acf[k])/df involves F
    # Simpler: use finite difference or analytic
    # For L2 norm: grad = 2 * acf_sidelobes convolved appropriately
    # Use FFT-based gradient: grad_f = 2 * ifft(F * conj(d_acf/df))
    # d(acf[k])/df[j] = delta(k-j) + delta(k+j) for circular
    # So grad = 2 * (acf[1:] shifted) 
    grad = np.zeros(N)
    # The gradient of sum_{k=1}^{N-1} acf[k]^2 w.r.t. f[j]
    # = 2 * sum_{k=1}^{N-1} acf[k] * d(acf[k])/df[j]
    # d(acf[k])/df[j] = 1 if k==j or k==N-j (for linear autocorrelation)
    # For linear autocorrelation: acf[k] = sum_i f[i]*f[i+k]
    # d(acf[k])/df[j] = f[j+k] if j+k<N else 0, plus f[j-k] if j-k>=0 else 0
    # So grad[j] = 2 * sum_{k=1}^{N-1} acf[k] * (f[j+k] if j+k<N else 0 + f[j-k] if j-k>=0 else 0)
    # This is expensive. Use FFT approach instead.
    # Actually, let's just use the fact that acf = ifft(|F|^2)
    # d(|F|^2)/df is complex, but we can use:
    # grad_f = 2 * ifft(F * conj(d_acf/df)) where d_acf/df is the correlation kernel
    # For simplicity, use numerical gradient estimation with small step
    return None  # We'll use finite differences

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = np.random.default_rng(42)
    
    # Initialize with smooth low-frequency signal
    # Use a chirp or sine wave to start with low autocorrelation sidelobes
    t = np.linspace(0, 1, N, endpoint=False)
    f = np.sin(2 * np.pi * 2 * t)  # Start with a simple sinusoid
    
    current_bound = compute_autocorrelation_bound(f)
    best_f = f.copy()
    best_bound = current_bound
    
    # Gradient descent with adaptive step size
    lr = 0.01
    momentum = np.zeros(N)
    beta = 0.9
    
    while time.time() < deadline:
        # Compute gradient via finite differences (batch)
        # Use a small batch of indices for efficiency
        batch_size = min(50, N)
        indices = rng.choice(N, size=batch_size, replace=False)
        
        grad = np.zeros(N)
        eps = 1e-4
        for idx in indices:
            f_plus = f.copy()
            f_minus = f.copy()
            f_plus[idx] += eps
            f_minus[idx] -= eps
            grad[idx] = (compute_autocorrelation_bound(f_plus) - compute_autocorrelation_bound(f_minus)) / (2 * eps)
        
        # Update with momentum
        momentum = beta * momentum + (1 - beta) * grad
        f -= lr * momentum
        
        # Project to reasonable range
        f = np.clip(f, -10, 10)
        
        new_bound = compute_autocorrelation_bound(f)
        
        if new_bound < best_bound:
            best_bound = new_bound
            best_f = f.copy()
        
        # Adapt learning rate
        if new_bound > current_bound * 1.1:
            lr *= 0.5
        elif new_bound < current_bound * 0.9:
            lr *= 1.1
        lr = min(lr, 0.1)
        
        current_bound = new_bound
    
    return best_f
