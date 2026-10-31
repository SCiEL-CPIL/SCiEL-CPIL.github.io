import numpy as np
import time

N = 400

def _generate_legendre(p, length):
    # Vectorized Legendre symbol calculation
    # i from 1 to length
    i = np.arange(1, length + 1)
    # Euler's criterion: (i/p) = i^((p-1)/2) mod p
    # Result is 1 or p-1 (which is -1)
    # If i is multiple of p, result is 0.
    # We assume length < p so no multiples of p.
    exp = (p - 1) // 2
    res = np.array([pow(int(x), exp, p) for x in i], dtype=np.int32)
    # Map p-1 to -1
    res[res == p - 1] = -1
    # Any 0s should be mapped to 1 (just in case, though length < p)
    res[res == 0] = 1
    return res.astype(np.int8)

def _max_offpeak_autoconv(x):
    n = len(x)
    size = 2 * n
    X = np.fft.fft(x, size=size)
    auto = np.fft.ifft(X * np.conj(X)).real
    offpeak = auto[1:n]
    return np.max(np.abs(offpeak))

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    candidates = []
    # Try a few primes
    for p in [397, 401, 409, 389]:
        if p <= N:
            continue
        seq = _generate_legendre(p, N)
        score = _max_offpeak_autoconv(seq)
        candidates.append((score, seq))
        
    if not candidates:
        return np.ones(N, dtype=np.int8)
        
    # Sort by score (lower is better)
    candidates.sort(key=lambda x: x[0])
    
    best_score, best_x = candidates[0]
    x = best_x.copy()
    
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            x[i] *= -1
            new_score = _max_offpeak_autoconv(x)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                x[i] *= -1
                
    return best_x
