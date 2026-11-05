import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 110.0  # Safety margin before 120s kill

def _supnorm(coeffs, n_grid=4096):
    """Compute sup norm of P(z) on unit circle using dense sampling and peak refinement."""
    # Fast evaluation using FFT
    # P(z) = sum c_k z^k. We can evaluate at z_j = exp(2pi i j / N)
    # This is equivalent to DFT if we pad coeffs.
    # coeffs are c_0 ... c_23.
    # P(z_j) = sum_{k=0}^{23} c_k exp(2pi i j k / N)
    # This is the DFT of the sequence [c_0, ..., c_23, 0, ..., 0] at indices 0..N-1?
    # Let x = [c_0, ..., c_23, 0, ..., 0] of length N.
    # DFT(x)_j = sum_{k=0}^{N-1} x_k exp(-2pi i j k / N)
    # Our sum is sum c_k exp(+2pi i j k / N).
    # So we can take np.fft.ifft(x) * N? Or just use np.fft.fft on reversed?
    # Actually, simpler: just compute directly for N=4096, it's fast enough.
    # But let's use FFT for speed.
    # P(z_j) = sum c_k (exp(2pi i / N))^k j
    # Let w = exp(2pi i / N).
    # This is the inverse DFT of c padded with zeros?
    # np.fft.ifft(c_padded) * N gives sum c_k exp(2pi i j k / N).
    
    c_padded = np.zeros(n_grid, dtype=complex)
    c_padded[:len(coeffs)] = coeffs
    vals = np.fft.ifft(c_padded) * n_grid
    max_val = np.max(np.abs(vals))
    
    # Refine peaks
    # Find indices where |vals| is close to max
    # Simple approach: check the top few peaks
    idx_max = np.argmax(np.abs(vals))
    # Golden section around idx_max
    # Angle = 2*pi*idx_max/n_grid
    # We can do a few iterations of refinement on the highest peak
    # For simplicity and speed, maybe just return max_val if it's close to optimal?
    # The task says "accuracy about 1e-9".
    # Let's do a quick refinement on the global max.
    
    def f(theta):
        # Evaluate P at angle theta
        # P(e^{i theta}) = sum c_k e^{i k theta}
        # Use dot product with precomputed powers?
        # For speed in refinement, we can just compute directly.
        # coeffs is small (24).
        k = np.arange(len(coeffs))
        return np.abs(np.sum(coeffs * np.exp(1j * theta * k)))
    
    # Golden section search
    a = 2 * np.pi * (idx_max - 1) / n_grid
    b = 2 * np.pi * (idx_max + 1) / n_grid
    # Ensure a < b
    if a > b:
        a, b = b, a
    
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    
    for _ in range(30):
        if fc < fd:
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
    
    max_val = max(max_val, fc, fd)
    return max_val

def construct():
    # Start with random or Golay-Rudin-Shapiro
    # Golay-Rudin-Shapiro is a good starting point.
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=float)
    best_score = _supnorm(best_coeffs) / np.sqrt(24)
    
    # Simulated Annealing
    temp = 1.0
    cooling = 0.999
    min_temp = 1e-6
    
    n_coeffs = 24
    k = np.arange(n_coeffs)
    
    while time.time() < _DEADLINE and temp > min_temp:
        # Random flip
        idx = np.random.randint(n_coeffs)
        new_coeffs = best_coeffs.copy()
        new_coeffs[idx] *= -1
        
        new_score = _supnorm(new_coeffs) / np.sqrt(24)
        
        if new_score < best_score or np.random.random() < np.exp((best_score - new_score) / temp):
            best_coeffs = new_coeffs
            best_score = new_score
        
        temp *= cooling
        
        # Optional: restart if stuck?
        # For now, keep it simple.
        
        # Check time more frequently
        if time.time() > _DEADLINE - 5:
            break
            
    return best_coeffs.tolist()
