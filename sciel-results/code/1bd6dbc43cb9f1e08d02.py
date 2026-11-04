"""
Strategy: Thue-Morse Initialization + Vectorized Greedy Refinement.

The Thue-Morse sequence has known low auto-correlation properties, providing a
much better starting point than Legendre sequences or random noise.
Combined with a vectorized greedy refinement that evaluates all single-bit flips
simultaneously via NumPy broadcasting, this approach efficiently minimizes the
maximum absolute auto-correlation peak.

Key insights:
1. Thue-Morse sequences have O(sqrt(N)) auto-correlation bounds, better than random.
2. Vectorized greedy: For each position i, compute the exact change in all
   autocorrelation lags k if f[i] is flipped. This allows evaluating all N
   possible single-bit flips in O(N^2) time per pass using NumPy, which is
   extremely fast for N=400 (~160k operations).
3. The greedy step picks the flip that most reduces the current peak.
4. Multiple passes with random restarts help escape local minima.
"""

import numpy as np
import time

N = 400

def _thue_morse(n):
    """
    Generate Thue-Morse sequence of length n, mapped to ±1.
    t(i) = parity of number of 1s in binary representation of i.
    Map even parity -> +1, odd parity -> -1.
    """
    seq = np.ones(n, dtype=np.float64)
    for i in range(n):
        # Count set bits
        count = bin(i).count('1')
        if count % 2 == 1:
            seq[i] = -1.0
    return seq

def _compute_autocorr(f):
    """
    Compute linear autocorrelation R[k] for k=0..N-1.
    R[k] = sum_{i=0}^{N-1-k} f[i] * f[i+k]
    Using FFT for speed.
    """
    n = len(f)
    size = 1
    while size < 2 * n:
        size *= 2
    F = np.fft.rfft(f, size)
    R = np.fft.irfft(F * np.conj(F), size)
    return R[:n]

def _peak_value(R):
    """Return max absolute autocorrelation for non-zero lags."""
    if len(R) <= 1:
        return 0.0
    return float(np.max(np.abs(R[1:])))

def _vectorized_greedy_pass(f, R, current_peak, deadline):
    """
    Perform one pass of vectorized greedy refinement.
    Returns (f, R, current_peak, improved).
    """
    n = len(f)
    
    # Precompute index arrays
    i_idx = np.arange(n)
    k_idx = np.arange(n)
    
    # M_add[i,k] = i + k, M_sub[i,k] = i - k
    M_add = i_idx[:, None] + k_idx[None, :]
    M_sub = i_idx[:, None] - k_idx[None, :]
    
    # f[i+k] if 0 <= i+k < n, else 0
    M_add_safe = np.clip(M_add, 0, n - 1)
    f_add = np.where((M_add >= 0) & (M_add < n), f[M_add_safe], 0.0)
    
    # f[i-k] if 0 <= i-k < n, else 0
    M_sub_safe = np.clip(M_sub, 0, n - 1)
    f_sub = np.where((M_sub >= 0) & (M_sub < n), f[M_sub_safe], 0.0)
    
    # Delta[i,k] = -2 * f[i] * (f[i+k] + f[i-k])
    f_i = f[:, None]
    Delta = -2.0 * f_i * (f_add + f_sub)
    
    # NewR[i,k] = R[k] + Delta[i,k]
    NewR = R[None, :] + Delta
    
    # Peak for each candidate flip i: max over k=1..n-1 of |NewR[i,k]|
    NewAbsR = np.abs(NewR[:, 1:])
    NewPeaks = np.max(NewAbsR, axis=1)
    
    # Find best flip
    best_i = int(np.argmin(NewPeaks))
    best_peak = float(NewPeaks[best_i])
    
    if best_peak < current_peak - 1e-12:
        f[best_i] = -f[best_i]
        R = R + Delta[best_i, :]
        current_peak = best_peak
        return f, R, current_peak, True
    else:
        return f, R, current_peak, False

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_f = None
    best_peak = float('inf')
    
    # Try multiple initializations
    inits = []
    
    # 1. Thue-Morse
    tm = _thue_morse(N)
    inits.append(tm)
    inits.append(-tm)  # Flip polarity
    
    # 2. Random sequences
    rng = np.random.RandomState(42)
    for _ in range(4):
        r = rng.choice([-1.0, 1.0], size=N)
        inits.append(r)
    
    # 3. Legendre-like (using quadratic residues mod a prime)
    # Use a prime p >= N
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    p = N
    while not is_prime(p):
        p += 1
    
    leg = np.ones(N, dtype=np.float64)
    for i in range(N):
        val = pow((i + 1) % p, (p - 1) // 2, p)
        if val == p - 1:  # Legendre symbol -1
            leg[i] = -1.0
        # else stays +1 (including 0 case)
    inits.append(leg)
    inits.append(-leg)
    
    for init_f in inits:
        if time.time() > deadline - 5.0:
            break
        
        f = init_f.copy()
        R = _compute_autocorr(f)
        current_peak = _peak_value(R)
        
        # Vectorized greedy refinement
        max_passes = 500
        for _pass in range(max_passes):
            if time.time() > deadline - 5.0:
                break
            f, R, current_peak, improved = _vectorized_greedy_pass(
                f, R, current_peak, deadline
            )
            if not improved:
                break
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_f = f.copy()
    
    if best_f is None:
        # Fallback: return Thue-Morse
        best_f = _thue_morse(N)
    
    return best_f
