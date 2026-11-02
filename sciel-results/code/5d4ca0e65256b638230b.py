import numpy as np
import time
import math

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _halton_sequence(n, base):
    """Generate first n values of Halton sequence in given base."""
    result = np.zeros(n)
    for i in range(1, n + 1):
        f = 1.0 / base
        val = 0.0
        j = i
        while j > 0:
            f /= base
            val += f * (j % base)
            j //= base
        result[i - 1] = val
    return result

def _update_acf_flip(A, s, i, old_si):
    """
    Update ACF after flipping s[i] in-place.
    old_si is the value before the flip.
    Returns new A array (copy).
    """
    A_new = A.copy()
    for k in range(1, N):
        change = 0.0
        if i + k < N:
            change -= 2.0 * old_si * s[i + k]
        if i - k >= 0:
            change -= 2.0 * s[i - k] * old_si
        A_new[k] += change
    return A_new

def _local_search(s, A, current_val, deadline, max_passes=100):
    """Perform local search with incremental ACF updates."""
    improved = True
    passes = 0
    while improved and passes < max_passes:
        if time.time() > deadline:
            break
        improved = False
        passes += 1
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > deadline:
                break
            old_si = s[i]
            s[i] = -old_si
            A_new = _update_acf_flip(A, s, i, old_si)
            val_new = _max_abs_acf_from_array(A_new)
            if val_new < current_val:
                current_val = val_new
                A = A_new
                improved = True
            else:
                s[i] = old_si
            # Periodic recomputation to avoid floating point drift
            if passes % 20 == 0 and i % 50 == 0:
                A = _calc_acf(s)
                current_val = _max_abs_acf_from_array(A)
    return s, A, current_val

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Generate diverse initializations
    initializations = []
    bases = [2, 3, 5, 7, 11, 13, 17, 19]
    for base in bases:
        halton_vals = _halton_sequence(N, base)
        s = np.where(halton_vals < 0.5, 1.0, -1.0)
        initializations.append(s)
    for _ in range(8):
        s = np.random.randint(0, 2, size=N) * 2 - 1
        initializations.append(s)
    initializations.append(np.ones(N))
    initializations.append(-np.ones(N))
    s_alt = np.ones(N)
    s_alt[1::2] = -1.0
    initializations.append(s_alt)
    for freq in [0.1, 0.2, 0.3, 0.5]:
        x = np.linspace(0, 2 * np.pi * freq, N)
        s_sine = np.where(np.sin(x) >= 0, 1.0, -1.0)
        initializations.append(s_sine)
    
    for s in initializations:
        if time.time() > deadline - 5:
            break
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        s, A, current_val = _local_search(s, A, current_val, deadline - 5, max_passes=100)
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    while time.time() < deadline - 5:
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        s, A, current_val = _local_search(s, A, current_val, deadline - 5, max_passes=50)
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()

    return best_seq
