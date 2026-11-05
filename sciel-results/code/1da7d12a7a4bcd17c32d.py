import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 105.0

def _supnorm_grid(coeffs, n_grid=4096):
    """Fast sup-norm estimate using FFT on a grid."""
    if isinstance(coeffs, list):
        coeffs = np.array(coeffs)
    coeffs_padded = np.zeros(n_grid, dtype=complex)
    coeffs_padded[:len(coeffs)] = coeffs
    vals = np.fft.fft(coeffs_padded)
    return float(np.max(np.abs(vals)))

def _supnorm_precise(coeffs, n_grid=4096, refine_top_k=8, refine_iters=40):
    """Precise sup-norm with golden-section refinement of top peaks."""
    if isinstance(coeffs, list):
        coeffs = np.array(coeffs)
    n = len(coeffs)
    
    theta = 2 * np.pi * np.arange(n_grid) / n_grid
    coeffs_padded = np.zeros(n_grid, dtype=complex)
    coeffs_padded[:n] = coeffs
    vals = np.fft.fft(coeffs_padded)
    mags = np.abs(vals)
    
    # Get indices of top K peaks
    top_k_indices = np.argsort(mags)[-refine_top_k:][::-1]
    
    max_val = 0.0
    k_arr = np.arange(n)
    
    for idx in top_k_indices:
        ang = theta[idx]
        delta = 2 * np.pi / n_grid
        a = ang - delta
        b = ang + delta
        
        def f(t):
            w = np.exp(1j * t)
            powers = np.power(w, k_arr)
            return float(np.abs(np.dot(coeffs, powers)))
        
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
            
    return max_val / np.sqrt(n)

def _gr_sequence(n):
    """Generate Golay-Rudin-Shapiro sequence of length n."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    return p[:n]

def construct():
    n = 24
    # Start with GR sequence
    coeffs = _gr_sequence(n)
    best_coeffs = coeffs[:]
    best_val = _supnorm_precise(best_coeffs)
    current_val = best_val
    
    # Simulated annealing
    T_start = 0.5
    T_end = 0.001
    n_steps = 50000
    
    step = 0
    while time.time() < _DEADLINE:
        # Temperature schedule
        if step < n_steps:
            T = T_start * (T_end / T_start) ** (step / n_steps)
        else:
            T = T_end
        
        # Pick a random position to flip
        i = np.random.randint(n)
        new_coeffs = coeffs[:]
        new_coeffs[i] = -new_coeffs[i]
        
        # Quick grid check
        new_grid = _supnorm_grid(new_coeffs)
        old_grid = _supnorm_grid(coeffs)
        
        # If grid suggests improvement or is close, do precise check
        if new_grid < old_grid + 0.5:
            new_precise = _supnorm_precise(new_coeffs)
            
            if new_precise < current_val:
                # Accept
                coeffs = new_coeffs
                current_val = new_precise
                if new_precise < best_val:
                    best_val = new_precise
                    best_coeffs = coeffs[:]
                    # Early exit if we hit the known optimum
                    if best_val < 1.2583758 + 1e-9:
                        break
            elif np.random.rand() < np.exp(-(new_precise - current_val) / T):
                # Metropolis acceptance
                coeffs = new_coeffs
                current_val = new_precise
        
        step += 1
        
        # Periodic restarts if stuck
        if step % 10000 == 0 and time.time() < _DEADLINE - 10:
            # Try a few random restarts
            for _ in range(3):
                if time.time() >= _DEADLINE - 5:
                    break
                rand_coeffs = np.random.choice([-1, 1], size=n)
                rand_val = _supnorm_precise(rand_coeffs)
                if rand_val < best_val:
                    best_val = rand_val
                    best_coeffs = rand_coeffs.tolist()
                    coeffs = rand_coeffs
                    current_val = rand_val
                else:
                    # Keep current state
                    pass
    
    return best_coeffs
