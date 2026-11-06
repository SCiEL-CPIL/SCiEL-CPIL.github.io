import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    N = 24
    target = 1.2583758 * np.sqrt(N)
    
    # Fast evaluation of max |P(z)| on unit circle
    def eval_max_norm(coeffs):
        # coeffs is array of 24 values ±1
        # Use FFT to evaluate on dense grid
        M = 4096
        z = np.exp(2j * np.pi * np.arange(M) / M)
        # P(z) = sum c_k z^k, evaluate via convolution
        # Pad coeffs to length M
        c = np.zeros(M, dtype=complex)
        c[:N] = coeffs.astype(complex)
        vals = np.fft.fft(c) * np.fft.fft(np.ones(M, dtype=complex))
        # Actually simpler: just compute directly for small N
        # P(z_j) = sum_k c_k * z_j^k
        # = sum_k c_k * exp(2j*pi*j*k/M)
        # This is like an inverse FFT of c padded
        c_pad = np.zeros(M, dtype=complex)
        c_pad[:N] = coeffs.astype(complex)
        # vals[j] = sum_k c_k * exp(2j*pi*j*k/M) = IFFT(c_pad) * M
        vals = np.fft.ifft(c_pad) * M
        return np.max(np.abs(vals))
    
    def eval_max_norm_refined(coeffs):
        """Refine the max by golden section around the peak."""
        M = 4096
        c_pad = np.zeros(M, dtype=complex)
        c_pad[:N] = coeffs.astype(complex)
        vals = np.fft.ifft(c_pad) * M
        idx = np.argmax(np.abs(vals))
        # Refine around idx using golden section
        theta0 = 2 * np.pi * idx / M
        # Search in [theta0 - pi/M, theta0 + pi/M]
        a = theta0 - np.pi / M
        b = theta0 + np.pi / M
        gr = (np.sqrt(5) - 1) / 2
        for _ in range(40):
            c1 = b - gr * (b - a)
            c2 = a + gr * (b - a)
            v1 = np.abs(np.sum(coeffs.astype(complex) * np.exp(1j * np.arange(N) * c1)))
            v2 = np.abs(np.sum(coeffs.astype(complex) * np.exp(1j * np.arange(N) * c2)))
            if v1 > v2:
                b = c2
            else:
                a = c1
        theta = (a + b) / 2
        return np.abs(np.sum(coeffs.astype(complex) * np.exp(1j * np.arange(N) * theta)))
    
    # Rudin-Shapiro initial
    def rudin_shapiro(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n], dtype=float)
    
    best_coeffs = rudin_shapiro(N)
    best_score = eval_max_norm_refined(best_coeffs) / np.sqrt(N)
    
    # Hill climbing
    def hill_climb(coeffs, max_iters=5000):
        coeffs = coeffs.copy()
        current_score = eval_max_norm_refined(coeffs) / np.sqrt(N)
        for i in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for k in range(N):
                new_coeffs = coeffs.copy()
                new_coeffs[k] *= -1
                new_score = eval_max_norm_refined(new_coeffs) / np.sqrt(N)
                if new_score < current_score - 1e-12:
                    coeffs = new_coeffs
                    current_score = new_score
                    improved = True
                    break
            if not improved:
                break
        return coeffs, current_score
    
    # Multiple restarts
    rng = np.random.default_rng(42)
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        if restart == 0:
            coeffs = rudin_shapiro(N)
        else:
            coeffs = np.where(rng.random(N) < 0.5, 1.0, -1.0)
        
        coeffs, score = hill_climb(coeffs)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs.copy()
    
    # More aggressive local search with random restarts
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        # Start from best and perturb
        coeffs = best_coeffs.copy()
        # Flip 2-3 random coefficients
        n_flips = rng.integers(2, 4)
        flip_idx = rng.choice(N, size=n_flips, replace=False)
        coeffs[flip_idx] *= -1
        
        coeffs, score = hill_climb(coeffs, max_iters=2000)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs.copy()
    
    return [int(c) for c in best_coeffs]
