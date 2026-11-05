import numpy as np

def _eval_norm(coeffs):
    """
    Computes the sup norm of the polynomial P(z) = sum c_k z^k for |z|=1.
    Uses dense sampling and golden section refinement for accuracy.
    """
    n = len(coeffs)
    # Dense sampling
    N = 2048
    z = np.exp(2j * np.pi * np.arange(N) / N)
    # Evaluate polynomial using FFT or direct matrix multiplication
    # coeffs are c_0, ..., c_{n-1}
    # P(z) = c_0 + c_1 z + ... + c_{n-1} z^{n-1}
    # We can use np.polynomial.polynomial.polyval or just dot product with powers
    # For speed, use FFT? No, n is small (24). Direct evaluation is fine.
    # z^k for k in 0..n-1
    powers = np.vander(z, n, increasing=True).T # Shape (n, N)
    # coeffs shape (n,). P(z) shape (N,)
    P = coeffs @ powers
    
    # Find peaks
    mags = np.abs(P)
    # Take top 10 peaks
    idx = np.argsort(mags)[-10:]
    max_val = 0.0
    for i in idx:
        # Golden section search around z_i
        # The peak is near angle theta_i
        theta = 2 * np.pi * i / N
        # Search interval [theta - pi/N, theta + pi/N]
        a = theta - np.pi / N
        b = theta + np.pi / N
        
        def f(theta_val):
            z_val = np.exp(1j * theta_val)
            # Compute P(z_val)
            # Use Horner's method or sum
            # coeffs is array of floats
            val = 0.0
            z_pow = 1.0
            for c in coeffs:
                val += c * z_pow
                z_pow *= z_val
            return abs(val)
        
        # Golden section
        gr = (np.sqrt(5) - 1) / 2
        c_val = b - gr * (b - a)
        d_val = a + gr * (b - a)
        fc = f(c_val)
        fd = f(d_val)
        
        for _ in range(50):
            if fc < fd:
                b = d_val
                d_val = c_val
                fd = fc
                c_val = b - gr * (b - a)
                fc = f(c_val)
            else:
                a = c_val
                c_val = d_val
                fc = fd
                d_val = a + gr * (b - a)
                fd = f(d_val)
        
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def construct():
    # Initial guess: Truncated Golay-Rudin-Shapiro sequence of length 32
    p, q = [1], [1]
    while len(p) < 32:
        p, q = p + q, p + [-c for c in q]
    # p is length 32
    coeffs = np.array(p[:24], dtype=float)
    
    current_max = _eval_norm(coeffs)
    
    # Hill Climbing
    improved = True
    while improved:
        improved = False
        best_flip_idx = -1
        best_new_max = current_max
        best_new_coeffs = coeffs.copy()
        
        for k in range(24):
            # Flip bit k
            new_coeffs = coeffs.copy()
            new_coeffs[k] = -new_coeffs[k]
            
            new_max = _eval_norm(new_coeffs)
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_flip_idx = k
                best_new_coeffs = new_coeffs
        
        if best_flip_idx != -1:
            coeffs = best_new_coeffs
            current_max = best_new_max
            improved = True
            
    # Convert back to list of ints
    return [int(c) for c in coeffs]
