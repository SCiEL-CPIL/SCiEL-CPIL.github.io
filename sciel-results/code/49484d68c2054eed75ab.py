"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_peak(h):
    """Compute max absolute value of linear autoconvolution of step function h."""
    n = len(h)
    size = 2 * n
    f = np.zeros(size)
    f[:n] = h
    F = np.fft.fft(f)
    conv = np.fft.ifft(F * F).real[:n]
    return np.max(np.abs(conv))

def _thue_morse(n):
    """Generate Thue-Morse sequence mapped to +1/-1."""
    t = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Count set bits in i
        count = bin(i).count('1')
        t[i] = 1.0 if count % 2 == 0 else -1.0
    return t

def _minkowski_question_mark(n):
    """Generate a Minkowski question mark-like sequence mapped to +1/-1."""
    # Use the Thue-Morse sequence as a base, but with a different mapping
    # Actually, let's use a simpler low-complexity sequence: the Rudin-Shapiro sequence
    t = np.zeros(n, dtype=np.float64)
    a, b = 1.0, 1.0
    t[0] = 1.0
    t[1] = 1.0
    for i in range(2, n):
        # Rudin-Shapiro: a_{2n} = a_n, a_{2n+1} = (-1)^n * b_n
        # This is getting complex, let's stick with Thue-Morse variants
        # Instead, use a random-like but structured sequence
        pass
    # Fallback to Thue-Morse if implementation is too complex
    return _thue_morse(n)

def _local_search(signs, time_limit=10.0):
    """Greedy local search to minimize autoconvolution peak."""
    best_signs = signs.copy()
    best_val = _autoconv_peak(best_signs)
    n = len(best_signs)
    start_time = time.time()
    
    # Greedy: try flipping each position, keep if it improves
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            old = best_signs[i]
            best_signs[i] = -old
            val = _autoconv_peak(best_signs)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_signs[i] = old
    
    return best_signs, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Start with Thue-Morse sequence
    tm = _thue_morse(N)
    
    # Try both polarities
    signs_a = tm.copy()
    signs_b = -tm
    val_a = _autoconv_peak(signs_a)
    val_b = _autoconv_peak(signs_b)
    
    if val_a <= val_b:
        best_signs = signs_a
        best_val = val_a
    else:
        best_signs = signs_b
        best_val = val_b
    
    # Local search to improve
    if time.time() < deadline - 20.0:
        best_signs, best_val = _local_search(best_signs, time_limit=110.0)
    
    # Final check: ensure we're within deadline
    return best_signs
# EVOLVE-BLOCK-END
