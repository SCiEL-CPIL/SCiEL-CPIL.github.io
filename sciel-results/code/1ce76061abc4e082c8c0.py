import numpy as np

def construct():
    # Start with Golay-Rudin-Shapiro polynomial truncated to 24
    p, q = [1], [1]
    while len(p) < 32: # Generate up to 32 to ensure we have enough, then truncate
        p, q = p + q, p + [-c for c in q]
    coeffs = np.array(p[:24], dtype=np.float64)
    
    # Pre-sample points on the unit circle
    # Dense sampling to approximate max modulus
    num_samples = 1024
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    # Precompute powers of z for each sample? 
    # z^k for k=0..23
    # This is a matrix of shape (24, num_samples)
    # Z[k, j] = exp(1j * angles[j] * k)
    # P(z_j) = sum c_k Z[k, j]
    # We can precompute Z
    k_vals = np.arange(24)
    Z = np.exp(1j * np.outer(k_vals, angles)) # Shape (24, 1024)
    
    def get_max_modulus(c):
        # c is shape (24,)
        # P_vals = Z.T @ c
        p_vals = Z.T @ c
        return np.max(np.abs(p_vals))
    
    current_max = get_max_modulus(coeffs)
    
    # Local Search: Hill Climbing
    # Try flipping each bit
    improved = True
    while improved:
        improved = False
        for i in range(24):
            # Flip bit i
            old_val = coeffs[i]
            coeffs[i] = -old_val
            new_max = get_max_modulus(coeffs)
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                # Revert
                coeffs[i] = old_val
                
    # Convert back to list of ints
    return [int(x) for x in coeffs]
