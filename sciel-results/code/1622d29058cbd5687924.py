import numpy as np
import time
from scipy import signal

N = 400
_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _autoconv_max_abs(x):
    """Compute max absolute value of autoconvolution using FFT."""
    n = len(x)
    # Zero-pad to at least 2n for linear convolution
    size = 1
    while size < 2 * n:
        size <<= 1
    f = np.fft.rfft(x, size)
    ac = np.fft.irfft(f * np.conj(f), size)[:n]
    return np.max(np.abs(ac))


def _squared_integral(x):
    return np.sum(x) ** 2


def _score(x):
    """Lower is better: max|autoconv| / integral^2"""
    si = _squared_integral(x)
    if abs(si) < 1e-10:
        return np.inf
    return _autoconv_max_abs(x) / si


def _welch_costas(p, g):
    """Generate Welch Costas permutation of length p using primitive root g mod p."""
    perm = []
    val = 1
    for k in range(p):
        perm.append(val % p)
        val = (val * g) % p
    return np.array(perm)


def _find_primitive_root(p):
    """Find a primitive root modulo prime p."""
    # Factor p-1
    factors = []
    n = p - 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    
    for g in range(2, p):
        is_prim = True
        for f in factors:
            if pow(g, (p - 1) // f, p) == 1:
                is_prim = False
                break
        if is_prim:
            return g
    return None


def _costas_to_signed(p, g, N):
    """Convert Costas permutation to signed sequence of length N."""
    perm = _welch_costas(p, g)
    # Create signed sequence: +1 where perm[i] > median, -1 otherwise
    median = np.median(perm)
    x = np.sign(perm - median)
    # Ensure non-zero (replace 0 with +1)
    x[x == 0] = 1
    # Pad or truncate to N
    if len(x) < N:
        x = np.concatenate([x, np.ones(N - len(x))])
    else:
        x = x[:N]
    return x.astype(np.float64)


def _local_sign_flip_optimize(x, max_rounds=50):
    """Local optimization by flipping signs to reduce max|autoconv|."""
    best_x = x.copy()
    best_score = _score(best_x)
    
    n = len(best_x)
    for round_num in range(max_rounds):
        if _time_remaining() < 5.0:
            break
        
        improved = False
        for i in range(n):
            if _time_remaining() < 5.0:
                break
            
            # Try flipping sign at position i
            new_x = best_x.copy()
            new_x[i] = -new_x[i]
            new_score = _score(new_x)
            
            if new_score < best_score:
                best_x = new_x
                best_score = new_score
                improved = True
        
        if not improved:
            break
    
    return best_x, best_score


def _try_multiple_costas(N):
    """Try multiple Costas-based initializations and return the best."""
    # Find primes near N
    primes = [397, 389, 383, 379, 373, 367, 359, 353, 349, 347]
    best_x = None
    best_score = np.inf
    
    for p in primes:
        if _time_remaining() < 10.0:
            break
        
        g = _find_primitive_root(p)
        if g is None:
            continue
        
        x = _costas_to_signed(p, g, N)
        x, score = _local_sign_flip_optimize(x, max_rounds=20)
        
        if score < best_score:
            best_score = score
            best_x = x
    
    return best_x, best_score


def _thue_morse_sequence(n):
    """Generate Thue-Morse sequence of length n as +1/-1."""
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Count bits
        bits = 0
        m = i
        while m:
            bits += m & 1
            m >>= 1
        seq[i] = 1.0 if bits % 2 == 0 else -1.0
    return seq


def _try_thue_morse(N):
    """Try Thue-Morse based initialization."""
    x = _thue_morse_sequence(N)
    x, score = _local_sign_flip_optimize(x, max_rounds=30)
    return x, score


def _try_chirp(N):
    """Try a chirp-like sequence."""
    n = np.arange(N)
    # Quadratic chirp
    x = np.sign(np.cos(2 * np.pi * 0.1 * n**2 / N))
    x[x == 0] = 1
    x, score = _local_sign_flip_optimize(x, max_rounds=30)
    return x, score


def construct():
    """Construct signed step function minimizing max|autoconv|/integral^2."""
    if _time_remaining() < 10.0:
        # Fallback: return a simple balanced sequence
        x = np.ones(N)
        x[N//2:] = -1
        return x
    
    candidates = []
    
    # Try Costas-based approaches
    x_costas, score_costas = _try_multiple_costas(N)
    if x_costas is not None:
        candidates.append((x_costas, score_costas))
    
    # Try Thue-Morse
    x_tm, score_tm = _try_thue_morse(N)
    candidates.append((x_tm, score_tm))
    
    # Try chirp
    x_chirp, score_chirp = _try_chirp(N)
    candidates.append((x_chirp, score_chirp))
    
    # Select best candidate
    best_x = min(candidates, key=lambda t: t[1])[0]
    
    # Final polish with more local optimization
    best_x, _ = _local_sign_flip_optimize(best_x, max_rounds=30)
    
    return best_x
