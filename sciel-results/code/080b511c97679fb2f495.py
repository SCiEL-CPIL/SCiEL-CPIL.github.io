"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 115.0  # Safety margin for 120s limit

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly():
    """Rudin-Shapiro coefficients truncated to _N."""
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    return p[:_N]

def _fast_score(coeffs):
    """Fast estimate of max |P(z)| using FFT."""
    n = 4096
    c = np.zeros(n, dtype=complex)
    c[:len(coeffs)] = coeffs
    # P(z_j) = sum c_k exp(2*pi*i*j*k/n) = n * IDFT(c)
    vals = np.fft.ifft(c) * n
    return np.max(np.abs(vals))

def _precise_score(coeffs):
    """Precise estimate of max |P(z)| using dense sampling and refinement."""
    n = 16384
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c = np.zeros(n, dtype=complex)
    c[:len(coeffs)] = coeffs
    vals = np.fft.ifft(c) * n
    mag = np.abs(vals)
    
    max_val = np.max(mag)
    
    # Refine top 3 peaks
    peak_indices = np.argsort(mag)[-3:]
    golden = (1 + math.sqrt(5)) / 2
    
    for idx in peak_indices:
        a = angles[(idx - 1) % n]
        b = angles[(idx + 1) % n]
        if a > b:
            a, b = b, a
        # Golden section search
        for _ in range(20):
            m1 = b - (b - a) / golden
            m2 = a + (b - a) / golden
            # Evaluate P at m1 and m2
            # P(theta) = sum c_k exp(i * k * theta)
            k = np.arange(_N)
            f1 = np.abs(np.sum(coeffs * np.exp(1j * m1 * k)))
            f2 = np.abs(np.sum(coeffs * np.exp(1j * m2 * k)))
            if f1 < f2:
                b = m2
            else:
                a = m1
        f_mid = np.abs(np.sum(coeffs * np.exp(1j * ((a + b) / 2) * k)))
        max_val = max(max_val, f_mid)
        
    return max_val / _SQRTN

def _local_search_precise(coeffs, max_iter=500):
    """Hill climbing using precise score."""
    coeffs = list(coeffs)
    best_score = _precise_score(coeffs)
    
    for _ in range(max_iter):
        if _time_left() < 5.0:
            break
        improved = False
        for k in range(_N):
            old = coeffs[k]
            coeffs[k] = -old
            score = _precise_score(coeffs)
            if score < best_score - 1e-12:
                best_score = score
                improved = True
            else:
                coeffs[k] = old
        if not improved:
            break
    return coeffs, best_score

def construct():
    # Start with Rudin-Shapiro
    coeffs = _rs_poly()
    
    # Simulated Annealing
    current = list(coeffs)
    current_score = _fast_score(current)
    best = list(current)
    best_score = current_score
    
    T = 1.0
    cooling = 0.999
    
    while _time_left() > 60.0:
        # Flip a random coefficient
        k = np.random.randint(0, _N)
        old = current[k]
        current[k] = -old
        
        new_score = _fast_score(current)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best = list(current)
                best_score = new_score
        else:
            current[k] = old
            
        T *= cooling
        
    # Polish the best found with precise score
    best_coeffs, best_precise_score = _local_search_precise(best, max_iter=1000)
    
    # If time permits, try a few restarts from the best solution with perturbations
    while _time_left() > 10.0:
        # Perturb best solution
        perturbed = list(best_coeffs)
        for _ in range(3):
            k = np.random.randint(0, _N)
            perturbed[k] = -perturbed[k]
        
        # Local search from perturbed
        new_coeffs, new_score = _local_search_precise(perturbed, max_iter=100)
        if new_score < best_precise_score:
            best_coeffs = new_coeffs
            best_precise_score = new_score
        else:
            # If no improvement, we might be stuck, but keep trying if time is left
            pass
            
    return best_coeffs
# EVOLVE-BLOCK-END
