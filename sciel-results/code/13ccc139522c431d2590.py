"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _fft_size(n):
    """Next power of 2 >= n."""
    s = 1
    while s < n:
        s *= 2
    return s

def _full_autoconv(x, fft_size=None):
    """Compute full linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    if fft_size is None:
        fft_size = _fft_size(size)
    X = np.fft.rfft(x, fft_size)
    conv = np.fft.irfft(X * X, fft_size)[:size]
    return conv

def _score_from_conv(conv, integral):
    """Compute score from precomputed convolution and integral."""
    if abs(integral) < 1e-12:
        return float('inf')
    max_conv = np.max(np.abs(conv))
    return max_conv / (integral * integral)

def _score(x):
    """Compute the score: max_abs_autoconv / (sum(x))^2"""
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    conv = _full_autoconv(x)
    return _score_from_conv(conv, integral)

def _update_conv_on_flip(conv, x, i, n):
    """
    Incrementally update the autoconvolution when flipping sign at index i.
    When x[i] flips to -x[i], the new convolution is:
      conv_new[k] = conv_old[k] - 2*x[i]*(x * delta_i)[k]
    where (x * delta_i)[k] = sum_j x[j] * delta_i(k-j) = x[k-i] if 0<=k-i<n else 0
    Actually: (x * delta_i)[k] = x[k-i] for 0 <= k-i < n, else 0.
    So conv_new = conv_old - 2*x[i]*x_shifted where x_shifted[k] = x[k-i] if valid.
    """
    old_val = x[i]
    # The contribution of position i to the autoconvolution:
    # conv[k] includes x[i]*x[k-i] for all valid k-i
    # When we flip x[i] to -x[i], the change in conv[k] is:
    #   delta_conv[k] = (-x[i])*x[k-i] - x[i]*x[k-i] = -2*x[i]*x[k-i]
    # where x[k-i] is only valid when 0 <= k-i < n
    # So k ranges from i to i+n-1
    lo = i
    hi = i + n  # exclusive
    # Clip to valid range [0, 2n-2]
    lo_clipped = max(lo, 0)
    hi_clipped = min(hi, 2 * n - 1)
    
    if lo_clipped < hi_clipped:
        # x[k-i] for k in [lo_clipped, hi_clipped)
        idx_start = lo_clipped - i
        idx_end = hi_clipped - i
        x_segment = x[idx_start:idx_end]
        conv[lo_clipped:hi_clipped] -= 2.0 * old_val * x_segment
    
    return conv

def _legendre_sequence(n):
    """Generate a Legendre symbol sequence truncated/padded to length n."""
    # Use a prime p > n, generate Legendre symbols, take first n
    p = 2
    while p < n + 10:
        # Check if p is prime
        is_prime = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime and p % 4 == 3:  # For better properties
            break
        p += 1
    
    x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a = i + 1
        if a % p == 0:
            x[i] = 1.0
        else:
            # Legendre symbol (a/p)
            ls = pow(a, (p - 1) // 2, p)
            if ls == p - 1:
                x[i] = -1.0
            else:
                x[i] = 1.0
    return x

def _m_sequence(n, seed=1):
    """Generate an M-sequence (maximal length LFSR sequence) of length n."""
    # Use a known primitive polynomial for LFSR
    # For simplicity, use a fixed primitive polynomial
    # x^8 + x^4 + x^3 + x^2 + 1 (degree 8, period 255)
    # We'll generate a long sequence and take first n bits
    degree = 8
    poly = 0b100111001  # x^8 + x^4 + x^3 + x^2 + 1
    # State: 8-bit register
    state = seed if seed != 0 else 1
    state &= (1 << degree) - 1
    
    seq = []
    period = (1 << degree) - 1
    for _ in range(n):
        seq.append(1.0 if (state & 1) == 0 else -1.0)
        # LFSR step
        feedback = (state & 1) ^ ((state >> 4) & 1) ^ ((state >> 5) & 1) ^ ((state >> 6) & 1)
        state = (state >> 1) | (feedback << (degree - 1))
        state &= (1 << degree) - 1
    
    return np.array(seq[:n], dtype=np.float64)

def _chirp_sequence(n):
    """Generate a chirp-like sequence based on quadratic residue."""
    # Use a prime p >= n
    p = 2
    while p < n:
        is_prime = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            break
        p += 1
    
    x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a = i * i % p
        if a == 0:
            x[i] = 1.0
        else:
            ls = pow(a, (p - 1) // 2, p)
            if ls == p - 1:
                x[i] = -1.0
            else:
                x[i] = 1.0
    return x

def _random_init(n, seed):
    """Generate a random ±1 sequence."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=n)

def _local_search_flip(x, max_iters=100, deadline=None, tol=1e-14):
    """
    Fast local search using incremental convolution updates.
    """
    x = x.copy().astype(np.float64)
    n = len(x)
    integral = np.sum(x)
    conv = _full_autoconv(x)
    best_score = _score_from_conv(conv, integral)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            old_val = x[i]
            x[i] = -old_val
            new_integral = integral - 2.0 * old_val
            
            # Update convolution incrementally
            _update_conv_on_flip(conv, x, i, n)
            
            new_score = _score_from_conv(conv, new_integral)
            
            if new_score < best_score - tol:
                best_score = new_score
                improved = True
                integral = new_integral
            else:
                # Revert
                x[i] = old_val
                _update_conv_on_flip(conv, x, i, n)
        if not improved:
            break
    return x, best_score

def _pair_flip_search(x, max_iters=20, deadline=None, tol=1e-14):
    """
    Local search with pair flips (flip two positions simultaneously)
    to escape local minima.
    """
    x = x.copy().astype(np.float64)
    n = len(x)
    integral = np.sum(x)
    conv = _full_autoconv(x)
    best_score = _score_from_conv(conv, integral)
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try a number of random pair flips
        n_tries = n
        for _ in range(n_tries):
            if deadline is not None and time.time() > deadline:
                break
            i, j = rng.choice(n, size=2, replace=False)
            if i == j:
                continue
            
            vi = x[i]
            vj = x[j]
            
            # Flip both
            x[i] = -vi
            x[j] = -vj
            new_integral = integral - 2.0 * vi - 2.0 * vj
            
            # Update conv for flip i
            _update_conv_on_flip(conv, x, i, n)
            # Update conv for flip j
            _update_conv_on_flip(conv, x, j, n)
            
            new_score = _score_from_conv(conv, new_integral)
            
            if new_score < best_score - tol:
                best_score = new_score
                improved = True
                integral = new_integral
            else:
                # Revert both
                x[i] = vi
                x[j] = vj
                _update_conv_on_flip(conv, x, i, n)
                _update_conv_on_flip(conv, x, j, n)
        
        if not improved:
            break
    return x, best_score

def _simulated_annealing(x, start_temp=1.0, end_temp=0.01, max_iters=100, deadline=None, tol=1e-14):
    """Simulated annealing with incremental updates."""
    x = x.copy().astype(np.float64)
    n = len(x)
    integral = np.sum(x)
    conv = _full_autoconv(x)
    best_x = x.copy()
    best_integral = integral
    best_conv = conv.copy()
    best_score = _score_from_conv(conv, integral)
    current_score = best_score
    rng = np.random.RandomState(7)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
        n_flips = max(1, n // 5)
        for _ in range(n_flips):
            if deadline is not None and time.time() > deadline:
                break
            i = rng.randint(n)
            old_val = x[i]
            x[i] = -old_val
            new_integral = integral - 2.0 * old_val
            
            _update_conv_on_flip(conv, x, i, n)
            
            new_score = _score_from_conv(conv, new_integral)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
                current_score = new_score
                integral = new_integral
                if new_score < best_score - tol:
                    best_score = new_score
                    best_x = x.copy()
                    best_integral = integral
                    best_conv = conv.copy()
            else:
                x[i] = old_val
                _update_conv_on_flip(conv, x, i, n)
    
    return best_x, best_score

def construct():
    """Construct a signed step function minimizing max absolute autoconvolution."""
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin (limit is 150s)
    
    best_x = None
    best_score = float('inf')
    
    # Generate diverse initial patterns
    initializers = []
    
    # Legendre sequence
    initializers.append(_legendre_sequence(N))
    
    # M-sequence
    initializers.append(_m_sequence(N, seed=1))
    initializers.append(_m_sequence(N, seed=3))
    
    # Chirp
    initializers.append(_chirp_sequence(N))
    
    # Random inits with different seeds
    for seed in [42, 123, 456, 789, 101, 202]:
        initializers.append(_random_init(N, seed))
    
    for x_init in initializers:
        if time.time() > deadline:
            break
        
        # Phase 1: Fast local search with incremental updates
        x_gs, score_gs = _local_search_flip(x_init, max_iters=50, deadline=deadline)
        
        if score_gs < best_score:
            best_score = score_gs
            best_x = x_gs.copy()
        
        # Phase 2: Pair flip search to escape local minima
        if time.time() < deadline - 10:
            x_pf, score_pf = _pair_flip_search(x_gs, max_iters=15, deadline=deadline)
            if score_pf < best_score:
                best_score = score_pf
                best_x = x_pf.copy()
        
        # Phase 3: Simulated annealing
        if time.time() < deadline - 10:
            x_sa, score_sa = _simulated_annealing(x_pf if 'x_pf' in dir() else x_gs,
                                                  start_temp=0.5, end_temp=0.01,
                                                  max_iters=30, deadline=deadline)
            if score_sa < best_score:
                best_score = score_sa
                best_x = x_sa.copy()
    
    # Final polish on the best solution
    if best_x is not None and time.time() < deadline - 5:
        x_final, score_final = _local_search_flip(best_x, max_iters=80, deadline=deadline)
        if score_final < best_score:
            best_x = x_final
            best_score = score_final
        
        # One more pair flip pass
        if time.time() < deadline - 3:
            x_pf2, score_pf2 = _pair_flip_search(x_final, max_iters=20, deadline=deadline)
            if score_pf2 < best_score:
                best_x = x_pf2
                best_score = score_pf2
    
    # Convert to integer sequence (±1)
    if best_x is not None:
        result = np.where(best_x > 0, 1, -1).astype(np.int8)
    else:
        result = np.ones(N, dtype=np.int8)
    
    return result
# EVOLVE-BLOCK-END
