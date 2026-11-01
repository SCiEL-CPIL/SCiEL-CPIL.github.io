# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

DEADLINE = 150.0
TIME_MARGIN = 15.0

def _compute_c3(x):
    """Compute C3 = max_{k!=0} |R[k]| / R[0] using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if R0 == 0:
        return 1.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    return max_sidelobe / R0

def _compute_R(x):
    """Compute full autocorrelation R[0..2N-2] via FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    return np.fft.irfft(power, n=n_fft)[:2 * N - 1]

def _c3_from_R(R):
    """Compute C3 from precomputed R."""
    R0 = R[0]
    if R0 == 0:
        return 1.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    return np.max(np.abs(sidelobes)) / R0

def _local_search_fft(x, max_iter=50000, time_limit=None):
    """Local search using FFT for C3 evaluation."""
    start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve = 0
    max_no_improve = 500
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        
        new_c3 = _compute_c3(x_new)
        
        if new_c3 < current_c3 - 1e-12:
            x = x_new
            current_c3 = new_c3
            no_improve = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_x, best_c3

def _simulated_annealing_fft(x, max_iter=50000, T_start=0.1, T_end=0.001, time_limit=None):
    """Simulated annealing using FFT for C3 evaluation."""
    start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    T = T_start
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        
        new_c3 = _compute_c3(x_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or (T > 1e-10 and random.random() < np.exp(-delta / T)):
            x = x_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
    
    return best_x, best_c3

def _thue_morse(n):
    """Generate Thue-Morse sequence."""
    seq = np.zeros(n)
    for i in range(n):
        c = bin(i).count('1')
        seq[i] = 1.0 if c % 2 == 0 else -1.0
    return seq

def _legendre_symbol(a, p):
    """Compute Legendre symbol (a/p) for prime p."""
    if a % p == 0:
        return 0
    res = pow(a, (p - 1) // 2, p)
    return 1 if res == 1 else -1

def _generate_legendre_seq(n, prime):
    seq = np.zeros(n)
    for i in range(n):
        val = _legendre_symbol(i + 1, prime)
        if val == 0:
            seq[i] = 1.0
        else:
            seq[i] = float(val)
    return seq

def _generate_mobius_seq(n):
    mu = np.ones(n + 1, dtype=int)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            for p in primes:
                if i * p > n:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    break
    mu[0] = 0
    for i in range(2, n + 1):
        if is_prime[i]:
            for j in range(i, n + 1, i):
                mu[j] *= -1
            for j in range(i*i, n + 1, i*i):
                mu[j] = 0
    seq = np.zeros(n)
    for i in range(n):
        val = mu[i + 1]
        if val == 0:
            seq[i] = 1.0
        else:
            seq[i] = float(val)
    return seq

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_c3 = _compute_c3(best_x)
    
    candidates = []
    
    # 1. All ones
    candidates.append(best_x.copy())
    
    # 2. Thue-Morse
    candidates.append(_thue_morse(N))
    
    # 3. Legendre (prime 401)
    candidates.append(_generate_legendre_seq(N, 401))
    
    # 4. Mobius
    candidates.append(_generate_mobius_seq(N))
    
    # 5. Random
    candidates.append(np.random.choice([-1, 1], size=N))
    
    # 6. Alternating
    alt = np.ones(N)
    alt[1::2] = -1
    candidates.append(alt)
    
    # Evaluate all candidates first
    for x_cand in candidates:
        c3_cand = _compute_c3(x_cand)
        if c3_cand < best_c3:
            best_c3 = c3_cand
            best_x = x_cand.copy()
            
    # Run local search on the best candidate
    if time.time() < DEADLINE - TIME_MARGIN - 10.0:
        time_limit = min(40.0, DEADLINE - TIME_MARGIN - time.time() - 5.0)
        if time_limit > 0:
            x_ls, c3_ls = _local_search_fft(best_x.copy(), max_iter=100000, time_limit=time_limit)
            if c3_ls < best_c3:
                best_x = x_ls
                best_c3 = c3_ls
                
    # Run simulated annealing on the best candidate
    if time.time() < DEADLINE - TIME_MARGIN - 5.0:
        time_limit = min(30.0, DEADLINE - TIME_MARGIN - time.time() - 2.0)
        if time_limit > 0:
            x_sa, c3_sa = _simulated_annealing_fft(best_x.copy(), max_iter=50000, time_limit=time_limit)
            if c3_sa < best_c3:
                best_x = x_sa
                best_c3 = c3_sa
                
    # Final polish
    if time.time() < DEADLINE - TIME_MARGIN - 2.0:
        time_limit = max(1.0, DEADLINE - TIME_MARGIN - time.time() - 1.0)
        x_final, c3_final = _local_search_fft(best_x.copy(), max_iter=50000, time_limit=time_limit)
        if c3_final < best_c3:
            best_x = x_final
            best_c3 = c3_final
            
    return best_x
# EVOLVE-BLOCK-END
