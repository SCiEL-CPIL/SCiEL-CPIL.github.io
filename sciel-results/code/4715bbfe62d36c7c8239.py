import numpy as np
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 145  # Safety margin before 150s kill

def _autoconv_max(h):
    """Compute max of full auto-convolution using FFT."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    S = H * np.conj(H)
    conv = np.fft.irfft(S, n=n_fft)
    return np.max(conv[:2 * N - 1])

def _evaluate(h):
    """Evaluate the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    max_conv = _autoconv_max(h)
    return abs(2.0 * N * max_conv / (s * s))

def construct():
    global START_TIME, TIME_LIMIT
    
    rng = np.random.default_rng(42)
    
    # Start with the seed
    best_h = np.ones(N).astype(float)
    best_val = _evaluate(best_h)  # Should be 2.0
    
    # Precompute DCT-II basis for fast evaluation
    # h[n] = c0 + sum_{k=1}^{K-1} c_k * cos(pi * (2n+1) * k / (2N))
    n = np.arange(N)
    K = 60  # Number of DCT coefficients to use
    
    # Precompute basis functions
    basis = np.zeros((K, N))
    basis[0, :] = 1.0 / np.sqrt(N)  # DC
    for k in range(1, K):
        basis[k, :] = np.sqrt(2.0 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    
    def coeffs_to_h(coeffs):
        """Convert DCT coefficients to time domain."""
        return basis @ coeffs
    
    def h_to_coeffs(h):
        """Convert time domain to DCT coefficients (orthogonal basis)."""
        return basis @ h
    
    # Initial coefficients
    c_init = h_to_coeffs(best_h)
    c_best = c_init.copy()
    
    # Simulated annealing in coefficient space
    # Fix DC to keep sum non-zero and controlled
    dc_fixed = c_init[0]  # Keep DC fixed initially
    
    T = 1.0  # Temperature
    T_min = 1e-6
    cooling = 0.9995
    max_iter = 20000
    
    c_curr = c_best.copy()
    c_curr[0] = dc_fixed  # Fix DC
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Propose a move: perturb a random subset of AC coefficients
        num_perturb = rng.integers(1, 5)
        indices = rng.integers(1, K, size=num_perturb)
        
        c_prop = c_curr.copy()
        for idx in indices:
            c_prop[idx] += rng.standard_normal() * T
        
        c_prop[0] = dc_fixed  # Keep DC fixed
        
        h_prop = coeffs_to_h(c_prop)
        val_prop = _evaluate(h_prop)
        
        if val_prop < best_val:
            best_val = val_prop
            best_h = h_prop.copy()
            c_best = c_prop.copy()
            c_curr = c_prop.copy()
        else:
            # Accept with probability exp(-(val_prop - val_curr)/T)
            val_curr = _evaluate(coeffs_to_h(c_curr))
            delta = val_prop - val_curr
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                c_curr = c_prop.copy()
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Local refinement: gradient-free coordinate descent on coefficients
    c_curr = c_best.copy()
    step_size = 0.01
    
    for coord_iter in range(2000):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        improved = False
        for k in range(1, K):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            old_val = c_curr[k]
            for delta_c in [-step_size, -step_size/2, step_size/2, step_size]:
                c_curr[k] = old_val + delta_c
                h_test = coeffs_to_h(c_curr)
                val_test = _evaluate(h_test)
                if val_test < best_val:
                    best_val = val_test
                    best_h = h_test.copy()
                    c_best = c_curr.copy()
                    improved = True
                    break
                else:
                    c_curr[k] = old_val
            
            if not improved:
                step_size *= 0.95
                if step_size < 1e-10:
                    break
    
    # Final validation
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    s = np.sum(h)
    conv = np.convolve(h, h)
    val = abs(2 * N * np.max(conv) / s**2)
    print(f"Score: {val}")
