import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 110.0

def _gr_sequence(n):
    """Generate Golay-Rudin-Shapiro sequence of length n."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    # Recursively build
    # Standard GR: P_{k+1} = P_k, Q_k; Q_{k+1} = P_k, -Q_k
    # Start P_0 = [1], Q_0 = [1]
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    return p[:n]

def _supnorm_grid(coeffs, n_grid=512):
    """Fast sup-norm estimate using a grid."""
    # coeffs: list or array of +1/-1
    if isinstance(coeffs, list):
        coeffs = np.array(coeffs)
    theta = 2 * np.pi * np.arange(n_grid) / n_grid
    # Compute P(e^{i theta}) = sum c_k e^{i k theta}
    # Use matrix multiplication: Z is n_grid x 24, Z[j, k] = e^{i j k 2 pi / n_grid}
    # This is a Vandermonde matrix.
    # Can use FFT? P(z) is a polynomial.
    # If we pad coeffs to length n_grid, FFT gives values at roots of unity.
    # But we want values at n_grid equally spaced points.
    # np.fft.fft(coeffs_padded) gives sum c_k (e^{i 2 pi / n_grid})^k.
    # This is exactly P(e^{i 2 pi j / n_grid}) for j=0..n_grid-1.
    # So we can just use FFT!
    coeffs_padded = np.zeros(n_grid, dtype=complex)
    coeffs_padded[:len(coeffs)] = coeffs
    vals = np.fft.fft(coeffs_padded)
    return float(np.max(np.abs(vals)))

def _supnorm_precise(coeffs, n_grid=512, refine_top_k=5, refine_iters=30):
    """Precise sup-norm with golden-section refinement."""
    if isinstance(coeffs, list):
        coeffs = np.array(coeffs)
    
    # 1. Grid search to find candidate peaks
    theta = 2 * np.pi * np.arange(n_grid) / n_grid
    coeffs_padded = np.zeros(n_grid, dtype=complex)
    coeffs_padded[:len(coeffs)] = coeffs
    vals = np.fft.fft(coeffs_padded)
    mags = np.abs(vals)
    
    # Get indices of top K peaks
    top_k_indices = np.argsort(mags)[-refine_top_k:][::-1]
    
    max_val = 0.0
    for idx in top_k_indices:
        ang = theta[idx]
        # Golden section search in [ang - delta, ang + delta]
        delta = 2 * np.pi / n_grid
        a = ang - delta
        b = ang + delta
        
        def f(t):
            # P(e^{it})
            # coeffs * exp(i t * k)
            k = np.arange(len(coeffs))
            w = np.exp(1j * t)
            # Efficient evaluation:
            # P(t) = sum c_k (e^{it})^k
            # Can use Horner's method or dot product.
            # Dot product with powers of w.
            # w^k for k=0..23
            powers = np.power(w, k)
            return float(np.abs(np.dot(coeffs, powers)))
        
        # Golden section
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f(c)
        fd = f(d)
        
        for _ in range(refine_iters):
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
        
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val / np.sqrt(len(coeffs))

def construct():
    # Initial candidate: GR sequence
    coeffs = _gr_sequence(24)
    current_val = _supnorm_precise(coeffs)
    best_coeffs = coeffs[:]
    best_val = current_val
    
    # Local search
    # We will try to improve by flipping bits.
    # To save time, we use _supnorm_grid for quick checks.
    # If a flip improves the grid value significantly, we verify with _supnorm_precise.
    
    n = 24
    grid_val = _supnorm_grid(coeffs)
    
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        for i in range(n):
            # Flip bit i
            new_coeffs = coeffs[:]
            new_coeffs[i] = -new_coeffs[i]
            
            # Quick check
            new_grid_val = _supnorm_grid(new_coeffs)
            
            # If grid value is lower, it's a candidate
            if new_grid_val < grid_val - 1e-9:
                # Precise check
                new_precise_val = _supnorm_precise(new_coeffs)
                
                if new_precise_val < best_val - 1e-9:
                    # Accept
                    coeffs = new_coeffs
                    grid_val = new_grid_val
                    best_coeffs = coeffs[:]
                    best_val = new_precise_val
                    improved = True
                    # If we hit the known optimum, we can stop early
                    if abs(best_val - 1.2583758) < 1e-6:
                        return best_coeffs
    
    # Final return
    return best_coeffs
