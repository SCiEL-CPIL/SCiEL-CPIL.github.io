import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Continuous relaxation: minimize max_k |c[k]|^2 / c[0]^2
    # where c is the linear autoconvolution of x with itself
    
    # Initialize with random signs
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    def autoconvolve(x):
        """Compute linear autoconvolution using FFT."""
        n = 2 * N
        X = np.fft.rfft(x, n=n)
        C = np.fft.irfft(X * X, n=n)
        # Linear autoconvolution: c[k] for k=0..N-1 is C[k], for k=-(N-1)..-1 is C[n-k]
        # Actually for autoconvolution (x * x), the result has length 2N-1
        # C[0] = sum x_i^2, C[1] = sum x_i x_{i+1}, etc.
        # The full linear convolution of x with x has length 2N-1
        # Using FFT with size >= 2N-1
        n_fft = 2 * N
        X = np.fft.rfft(x, n=n_fft)
        C = np.fft.irfft(X * X, n=n_fft)
        # C has length n_fft, but we only need the first 2N-1 values
        # For autoconvolution, c[k] for k=0,...,N-1 and c[-k] = c[k]
        # The linear autoconvolution: c[k] = sum_{i} x[i] x[i+k] for k >= 0
        # c[k] = C[k] for k = 0, ..., N-1
        return C[:N]  # c[0], c[1], ..., c[N-1]
    
    def objective(x):
        c = autoconvolve(x)
        c0 = c[0]
        if c0 <= 1e-10:
            return 1e10
        # Max of |c[k]| for k=1..N-1, normalized by c[0]
        max_abs = np.max(np.abs(c[1:]))
        return max_abs / c0
    
    # Projected gradient descent on continuous relaxation
    # Objective: f(x) = max_{k>=1} |c[k]| / c[0]
    # This is not differentiable, but we can use subgradient methods
    
    best_x = x.copy()
    best_val = objective(x)
    
    # Use ADMM-like approach or projected gradient
    # Let's try: minimize t subject to |c[k]| <= t * c[0] for all k >= 1
    # This is a convex optimization in (x, t) but the constraint involves products
    
    # Alternative: use the fact that for ±1 vectors, we can do local search
    # but first try continuous relaxation with gradient steps
    
    # Gradient of c[k] with respect to x[i]:
    # c[k] = sum_i x[i] x[i+k]
    # dc[k]/dx[i] = x[i+k] + x[i-k] (with boundary conditions)
    
    def gradient(x, c):
        """Compute gradient of max_k |c[k]| with respect to x."""
        # Find the k that achieves the max
        abs_c = np.abs(c[1:])
        k_max = np.argmax(abs_c) + 1  # k in 1..N-1
        
        # Gradient of c[k_max] with respect to x
        grad = np.zeros(N)
        km = k_max
        for i in range(N - km):
            grad[i] += x[i + km]
            grad[i + km] += x[i]
        
        # Sign of c[k_max]
        sign = np.sign(c[k_max])
        return sign * grad
    
    # Projected gradient descent
    lr = 0.1
    for iteration in range(5000):
        if time.time() > deadline - 10.0:
            break
        
        c = autoconvolve(x)
        c0 = c[0]
        if c0 <= 1e-10:
            break
        
        val = np.max(np.abs(c[1:])) / c0
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Subgradient step
        abs_c = np.abs(c[1:])
        k_max = np.argmax(abs_c) + 1
        
        grad = np.zeros(N)
        km = k_max
        for i in range(N - km):
            grad[i] += x[i + km]
            grad[i + km] += x[i]
        
        sign = np.sign(c[k_max])
        grad = sign * grad
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        x = x - lr * grad
        
        # Project to unit sphere or keep as is (we'll threshold later)
        # Actually, let's project to have similar magnitude
        x_norm = np.linalg.norm(x)
        if x_norm > 1e-10:
            x = x * (np.sqrt(N) / x_norm)  # Keep average magnitude ~1
    
    # Threshold to ±1
    x_bin = np.sign(x)
    x_bin[x_bin == 0] = 1.0
    
    # Local search: flip individual bits or blocks to improve
    def score_binary(s):
        c = autoconvolve(s)
        c0 = c[0]
        if c0 <= 1e-10:
            return 1e10
        return np.max(np.abs(c[1:])) / c0
    
    best_bin = x_bin.copy()
    best_bin_val = score_binary(x_bin)
    
    # Greedy local search: try flipping each bit
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            best_bin[i] = -best_bin[i]
            val = score_binary(best_bin)
            if val < best_bin_val:
                best_bin_val = val
                improved = True
            else:
                best_bin[i] = -best_bin[i]
    
    # Block flips
    for block_size in [4, 8, 16, 32]:
        if time.time() > deadline - 5.0:
            break
        for start in range(0, N - block_size + 1, block_size):
            if time.time() > deadline - 5.0:
                break
            best_bin[start:start+block_size] *= -1
            val = score_binary(best_bin)
            if val < best_bin_val:
                best_bin_val = val
            else:
                best_bin[start:start+block_size] *= -1
    
    return best_bin
