import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1] via FFT (init/verification only)."""
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

def _flip_delta(s, A, i):
    """Change in ACF if we flip s[i] -> -s[i].  O(N)."""
    delta = np.zeros(N)
    factor = -2.0 * s[i]
    if i < N:
        delta[:N-i] += factor * s[i:N]
    if i >= 0:
        delta[:i+1] += factor * s[i::-1]
    return delta

def _thue_morse():
    n = np.arange(N)
    x = n
    x = x ^ (x >> 1)
    x = x ^ (x >> 2)
    x = x ^ (x >> 4)
    x = x ^ (x >> 8)
    x = x ^ (x >> 16)
    x = x ^ (x >> 32)
    parity = (x & 1).astype(np.float64)
    return 1.0 - 2.0 * parity

def _legendre():
    p = N
    def is_prime(n):
        if n < 2: return False
        if n % 2 == 0: return n == 2
        i = 3
        while i * i <= n:
            if n % i == 0: return False
            i += 2
        return True
    while not is_prime(p):
        p += 1
    n = np.arange(N)
    ls = np.array([pow(int(x), (p-1)//2, p) for x in n], dtype=np.float64)
    return np.where(ls == 0, 1.0, np.where(ls == 1, 1.0, -1.0))

def _greedy_polish(s, A, current_val, deadline):
    """Single-bit best-improvement greedy. O(N) per candidate via analytic delta."""
    passes = 0
    max_passes = 500
    while passes < max_passes and time.time() < deadline:
        passes += 1
        best_i = -1
        best_val = current_val
        best_delta = None
        for i in range(N):
            if time.time() > deadline:
                break
            delta = _flip_delta(s, A, i)
            val_new = float(np.max(np.abs((A + delta)[1:])))
            if val_new < best_val:
                best_val = val_new
                best_i = i
                best_delta = delta
        if best_i < 0:
            break
        # apply: A_new = A + delta, then flip s
        A = A + best_delta
        s[best_i] = -s[best_i]
        current_val = best_val
    return s, A, current_val

def _sa_polish(s, A, current_val, deadline):
    T = 0.05
    T_min = 0.0005
    cooling = 0.997
    while T > T_min and time.time() < deadline:
        i = np.random.randint(N)
        delta = _flip_delta(s, A, i)
        A_new = A + delta
        val_new = float(np.max(np.abs(A_new[1:])))
        if val_new < current_val or np.random.random() < np.exp(-(val_new - current_val) / T):
            s[i] = -s[i]
            A = A_new
            current_val = val_new
        T *= cooling
    return s, A, current_val

def _softmax_polish(s, A, current_val, deadline):
    beta = 50.0
    def surrogate(A):
        x = beta * np.abs(A[1:])
        x = x - x.max()
        return float((x.max() + np.log(np.sum(np.exp(x)))) / beta)
    cur_f = surrogate(A)
    passes = 0
    while passes < 200 and time.time() < deadline:
        passes += 1
        best_i = -1
        best_f = cur_f
        best_delta = None
        for i in range(N):
            if time.time() > deadline:
                break
            delta = _flip_delta(s, A, i)
            f_new = surrogate(A + delta)
            if f_new < best_f:
                best_f = f_new
                best_i = i
                best_delta = delta
        if best_i < 0:
            break
        A = A + best_delta
        s[best_i] = -s[best_i]
        cur_f = best_f
    return s, A, float(np.max(np.abs(A[1:])))

def construct():
    start_time = time.time()
    deadline = 138.0  # safety margin under 150s kill

    best_val = float('inf')
    best_seq = _thue_morse()

    inits = [
        _thue_morse(),
        _legendre(),
        np.random.randint(0, 2, size=N) * 2 - 1,
    ]

    for s0 in inits:
        if time.time() > deadline - 20:
            break
        s = s0.copy()
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)

        s, A, current_val = _greedy_polish(s, A, current_val, deadline - 15)
        s, A, current_val = _sa_polish(s, A, current_val, deadline - 8)
        s, A, current_val = _softmax_polish(s, A, current_val, deadline - 2)
        s, A, current_val = _greedy_polish(s, A, current_val, deadline - 1)

        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()

    return best_seq
