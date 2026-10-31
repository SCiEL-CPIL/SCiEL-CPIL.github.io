"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import numpy as np
import time

def construct():
    """
    Return a signed step function with N=400 equal intervals on [-1/4, 1/4].
    We use a two-level block structure with alternating signs to reduce the
    autoconvolution upper bound.
    
    The bound C3 is minimized by making the function have near-zero mean and
    structured sign patterns that reduce the L4 norm of the autoconvolution.
    """
    deadline = time.time() + 110.0  # safety margin before 150s kill
    
    # Start with a simple alternating sign pattern
    # Two-level structure: blocks of 25 with alternating sign
    x = np.zeros(N)
    block_size = 25
    for i in range(0, N, block_size):
        block_idx = i // block_size
        sign = 1.0 if block_idx % 2 == 0 else -1.0
        x[i:i+block_size] = sign
    
    # Try to improve with a simple local search / coordinate descent
    # We'll try flipping individual signs to reduce the bound
    
    def compute_bound(f):
        """Compute the autoconvolution upper bound C3."""
        # Autoconvolution via FFT
        nfft = 2 * N
        F = np.fft.rfft(f, n=nfft)
        conv = np.fft.irfft(F * F, n=nfft)[:N]
        # The bound involves L4 norm of autoconvolution and L2 norm of f
        # C3 ~ ||f * f||_4 / ||f||_2^2 or similar
        # Let's use a standard form: (sum |conv|^4)^(1/4) / (sum f^2)
        l2_sq = np.sum(f ** 2)
        if l2_sq == 0:
            return np.inf
        l4_conv = (np.sum(np.abs(conv) ** 4)) ** 0.25
        return l4_conv / l2_sq
    
    best_x = x.copy()
    best_bound = compute_bound(best_x)
    
    # Local search: try flipping individual elements
    n_iterations = 0
    max_iterations = 50000
    while n_iterations < max_iterations:
        if time.time() > deadline - 15.0:
            break
        improved = False
        # Try a random subset of positions to flip
        for pos in range(N):
            if time.time() > deadline - 15.0:
                break
            old_val = best_x[pos]
            best_x[pos] = -old_val
            new_bound = compute_bound(best_x)
            if new_bound < best_bound - 1e-12:
                best_bound = new_bound
                improved = True
            else:
                best_x[pos] = old_val
        n_iterations += 1
        if not improved:
            # Try block flips
            block_flipped = False
            for bs in [2, 4, 5, 10, 25, 50]:
                if time.time() > deadline - 15.0:
                    break
                for start in range(0, N - bs, bs):
                    if time.time() > deadline - 15.0:
                        break
                    old_vals = best_x[start:start+bs].copy()
                    best_x[start:start+bs] = -old_vals
                    new_bound = compute_bound(best_x)
                    if new_bound < best_bound - 1e-12:
                        best_bound = new_bound
                        block_flipped = True
                    else:
                        best_x[start:start+bs] = old_vals
            if not block_flipped:
                break
    
    return best_x
