import numpy as np
import time

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

# Precompute FFT size
FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    F = np.fft.rfft(f, n=FFT_SIZE)
    C = np.fft.irfft(F * F, n=FFT_SIZE)
    return C[:2 * N - 1]

def _score(f):
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _local_search_binary_fast(f, max_iter=1000, time_budget=5.0):
    """
    Local search on binary {-1, 1} sequences using incremental convolution updates.
    """
    f = np.sign(f).astype(np.float64)
    S = np.sum(f)
    if abs(S) < 1e-6:
        return f, 1e10
        
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
            
        best_improvement = 0
        best_flip_idx = -1
        best_new_C = None
        best_new_S = S
        
        for i in range(N):
            v = f[i]
            delta = -2.0 * v
            
            C_new = C.copy()
            
            k_start = i
            k_end = i + N
            
            if k_start < len(C_new):
                end_idx = min(k_end, len(C_new))
                start_idx = k_start
                f_segment = f[:end_idx - start_idx]
                C_new[start_idx:end_idx] += 2.0 * delta * f_segment
                
            if 2 * i < len(C_new):
                C_new[2 * i] += delta * delta
                
            new_S = S - 2.0 * v
            if abs(new_S) < 1e-6:
                continue
                
            new_score = np.max(np.abs(C_new)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_flip_idx = i
                best_new_C = C_new
                best_new_S = new_S
        
        if best_flip_idx >= 0:
            v = f[best_flip_idx]
            f[best_flip_idx] = -v
            S = best_new_S
            C = best_new_C
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 10:
                break
                
    return f, score

def _legendre_symbol(a, p):
    """
    Compute the Legendre symbol (a/p).
    Returns 1 if a is a quadratic residue mod p, -1 if non-residue, 0 if a % p == 0.
    """
    if a % p == 0:
        return 0
    res = pow(a, (p - 1) // 2, p)
    if res == 1:
        return 1
    elif res == p - 1:
        return -1
    else:
        return 0

def _generate_legendre_sequence(p, N):
    """
    Generate a sequence of length N based on the Legendre symbol of a prime p.
    h[i] = Legendre(i+1, p) for i in range(N).
    If Legendre is 0, map to 1.
    """
    h = np.zeros(N)
    for i in range(N):
        sym = _legendre_symbol(i + 1, p)
        h[i] = 1.0 if sym >= 0 else -1.0
    return h

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Algebraic initializations (Legendre sequences)
    # These have good autocorrelation properties theoretically.
    primes_to_try = [397, 401, 389, 409, 383, 419, 379, 421, 373, 431, 433, 439, 443, 449, 457]
    
    for p in primes_to_try:
        if time.time() > DEADLINE:
            break
        h = _generate_legendre_sequence(p, N)
        s = np.sum(h)
        if abs(s) < N * 0.1:
             # Ensure sum is not too small
             if s < 0:
                 h = -h
             if abs(np.sum(h)) < 1e-6:
                 h[0] = 1.0
        
        h_opt, val_opt = _local_search_binary_fast(h, max_iter=2000, time_budget=10.0)
        if val_opt < best_score:
            best_score = val_opt
            best_f = h_opt.copy()

    # Strategy 2: Structured initializations (Cosines, Chirps, Blocks)
    x = np.arange(N)
    initializations = []
    
    # Cosines
    for k in range(1, 11):
        h = np.sign(np.cos(2 * np.pi * k * x / N))
        h[h == 0] = 1.0
        initializations.append(h)
        
    # Chirps
    for freq_mult in [1.0, 2.0, 5.0, 10.0]:
        h = np.sign(np.cos(2 * np.pi * freq_mult * x * x / (2 * N)))
        h[h == 0] = 1.0
        initializations.append(h)
        
    # Blocks
    for _ in range(10):
        h = np.ones(N)
        pos = 0
        while pos < N:
            block_len = rng.integers(2, 20)
            sign = rng.choice([-1.0, 1.0])
            h[pos:pos+block_len] = sign
            pos += block_len
        initializations.append(h)
        
    # Mostly +1 with sparse -1
    for n_neg in [5, 10, 20, 50, 100]:
        h = np.ones(N)
        neg_idx = rng.choice(N, n_neg, replace=False)
        h[neg_idx] = -1.0
        initializations.append(h)

    for h in initializations:
        if time.time() > DEADLINE:
            break
        s = np.sum(h)
        if abs(s) < N * 0.1:
             if s < 0:
                 h = -h
             if abs(np.sum(h)) < 1e-6:
                 h[0] = 1.0
                 
        h_opt, val_opt = _local_search_binary_fast(h, max_iter=2000, time_budget=10.0)
        if val_opt < best_score:
            best_score = val_opt
            best_f = h_opt.copy()

    # Strategy 3: Random Restarts
    # Use different densities of 1s to ensure S is not too small
    for _ in range(50):
        if time.time() > DEADLINE:
            break
            
        f_rand = rng.choice([-1.0, 1.0], size=N)
        S_rand = np.sum(f_rand)
        if abs(S_rand) < N * 0.2:
            continue
            
        f_rand_local, score_rand_local = _local_search_binary_fast(f_rand, max_iter=1000, time_budget=5.0)
        
        if score_rand_local < best_score:
            best_score = score_rand_local
            best_f = f_rand_local.copy()

    # Final Polish: Run local search on the best found so far with more time
    if time.time() < DEADLINE - 5.0:
        f_final, score_final = _local_search_binary_fast(best_f, max_iter=5000, time_budget=DEADLINE - time.time() - 2.0)
        if score_final < best_score:
            best_score = score_final
            best_f = f_final

    return best_f
