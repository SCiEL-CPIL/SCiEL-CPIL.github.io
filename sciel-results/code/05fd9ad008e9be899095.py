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

# Precompute FFT infrastructure for fast incremental updates
_NFFT = 16384
_ANGLES = np.linspace(0, 2 * np.pi, _NFFT, endpoint=False)
# Precompute exp(i * k * angle) for each coefficient index k
_EXP_TABLE = np.exp(1j * np.outer(_ANGLES, np.arange(_N)))  # shape (_NFFT, _N)

def _init_fft_state(coeffs):
    """Initialize FFT evaluation state for incremental updates."""
    c = np.zeros(_NFFT, dtype=complex)
    c[:len(coeffs)] = coeffs
    vals = np.fft.ifft(c) * _NFFT
    return vals

def _flip_coeff(vals, coeffs, k):
    """Update FFT values after flipping coefficient k. Returns new vals."""
    # Flipping c_k changes P(z_j) by -2*c_k*exp(i*k*z_j)
    delta = -2.0 * coeffs[k] * _EXP_TABLE[:, k]
    return vals + delta

def _get_max(vals):
    """Get max |P(z)| from FFT values."""
    return np.max(np.abs(vals))

def _precise_score(coeffs):
    """Precise estimate of max |P(z)| using dense sampling and refinement."""
    vals = _init_fft_state(coeffs)
    mag = np.abs(vals)
    max_val = np.max(mag)
    
    # Find all local maxima in the magnitude profile
    # A local max is where mag[i] >= mag[i-1] and mag[i] >= mag[i+1]
    local_maxima = []
    for i in range(_NFFT):
        if mag[i] >= mag[(i-1) % _NFFT] and mag[i] >= mag[(i+1) % _NFFT]:
            local_maxima.append(i)
    
    # Refine top 5 local maxima
    local_maxima.sort(key=lambda i: -mag[i])
    golden = (1 + math.sqrt(5)) / 2
    k_arr = np.arange(_N)
    
    for idx in local_maxima[:5]:
        a = _ANGLES[(idx - 1) % _NFFT]
        b = _ANGLES[(idx + 1) % _NFFT]
        if a > b:
            a, b = b + 2*np.pi, b  # Handle wrap-around
            # Actually, let's handle this differently
            # If idx is near 0, a wraps around
        if a > b:
            a = a - 2*np.pi
            b = b
        # Golden section search
        for _ in range(25):
            m1 = b - (b - a) / golden
            m2 = a + (b - a) / golden
            f1 = np.abs(np.sum(coeffs * np.exp(1j * m1 * k_arr)))
            f2 = np.abs(np.sum(coeffs * np.exp(1j * m2 * k_arr)))
            if f1 < f2:
                b = m2
            else:
                a = m1
        f_mid = np.abs(np.sum(coeffs * np.exp(1j * ((a + b) / 2) * k_arr)))
        max_val = max(max_val, f_mid)
        
    return max_val / _SQRTN

def _fast_local_search(coeffs, max_iter=200):
    """Fast hill climbing using incremental FFT updates."""
    coeffs = list(coeffs)
    vals = _init_fft_state(coeffs)
    best_score = _get_max(vals) / _SQRTN
    
    for _ in range(max_iter):
        if _time_left() < 5.0:
            break
        improved = False
        best_k = -1
        best_new_score = best_score
        for k in range(_N):
            old_val = coeffs[k]
            new_vals = _flip_coeff(vals, coeffs, k)
            new_score = _get_max(new_vals) / _SQRTN
            if new_score < best_new_score - 1e-12:
                best_new_score = new_score
                best_k = k
        if best_k >= 0:
            vals = _flip_coeff(vals, coeffs, best_k)
            coeffs[best_k] = -coeffs[best_k]
            best_score = best_new_score
            improved = True
        if not improved:
            break
    return coeffs, best_score

def _precise_local_search(coeffs, max_iter=50):
    """Hill climbing using precise score for final polish."""
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
    
    # Phase 1: Fast simulated annealing with incremental FFT
    current = list(coeffs)
    vals = _init_fft_state(current)
    current_score = _get_max(vals) / _SQRTN
    best = list(current)
    best_score = current_score
    
    T = 1.0
    cooling = 0.9995
    
    while _time_left() > 40.0:
        k = np.random.randint(0, _N)
        old_val = current[k]
        new_vals = _flip_coeff(vals, current, k)
        new_score = _get_max(new_vals) / _SQRTN
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
            vals = new_vals
            current[k] = -old_val
            current_score = new_score
            if new_score < best_score:
                best = list(current)
                best_score = new_score
        else:
            current[k] = old_val
            
        T *= cooling
        
    # Phase 2: Fast local search from best
    best_coeffs, best_fast_score = _fast_local_search(best, max_iter=500)
    
    # Phase 3: Precise local search to polish
    best_coeffs, best_precise_score = _precise_local_search(best_coeffs, max_iter=100)
    
    # Phase 4: Try perturbations and restarts
    while _time_left() > 15.0:
        perturbed = list(best_coeffs)
        n_flips = np.random.randint(2, 6)
        for _ in range(n_flips):
            k = np.random.randint(0, _N)
            perturbed[k] = -perturbed[k]
        
        # Fast local search
        new_coeffs, new_fast_score = _fast_local_search(perturbed, max_iter=100)
        if new_fast_score < best_precise_score + 0.01:  # Only pursue if close
            new_coeffs, new_score = _precise_local_search(new_coeffs, max_iter=50)
            if new_score < best_precise_score:
                best_coeffs = new_coeffs
                best_precise_score = new_score
                
    return best_coeffs
# EVOLVE-BLOCK-END
