"""
Combined approach: Quadratic residue (Legendre symbol) initialization +
gradient-based greedy local search with linear autoconvolution scoring.

The structural change: use number-theoretic pseudorandom sequences (Legendre
symbols mod primes near 400) as initial candidates, which inherently have
low additive energy and thus suppressed autoconvolution peaks. Then refine
with an efficient greedy local search that uses the exact linear ACF update
formula to avoid recomputing the full FFT on every trial.
"""
import numpy as np
import time

N = 400
DX = 0.5 / N  # interval width: total length 0.5, N intervals

def _linear_acf(h):
    """
    Compute linear autoconvolution coefficients c_k = sum_i h[i]*h[i+k]
    for k = -(N-1), ..., N-1 using FFT.
    Returns array of length 2N-1 where index (N-1)+k corresponds to lag k.
    """
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    power = np.abs(H) ** 2
    acf_full = np.fft.irfft(power, n=fft_size)
    # Build the linear ACF of length 2n-1
    acf = np.empty(2 * n - 1, dtype=np.float64)
    acf[n - 1] = acf_full[0]
    for k in range(1, n):
        acf[n - 1 + k] = acf_full[k]
        acf[n - 1 - k] = acf_full[fft_size - k]
    return acf

def _psl_value(acf, n):
    """Get PSL value: max |c_k| for k != 0."""
    if n <= 1:
        return 0.0
    center = n - 1
    right = acf[center + 1:]
    left = acf[:center]
    return float(max(np.max(np.abs(right)), np.max(np.abs(left))))

def _score(h):
    """Upper bound on C3: DX * max |c_k| for k != 0."""
    acf = _linear_acf(h)
    return float(DX * _psl_value(acf, len(h)))

def _legendre_symbol(a, p):
    """
    Compute the Legendre symbol (a/p) for prime p.
    Returns 0 if p|a, 1 if a is a QR mod p, -1 if QNR mod p.
    """
    if a % p == 0:
        return 0
    # Euler's criterion
    return pow(a, (p - 1) // 2, p) if pow(a, (p - 1) // 2, p) != p - 1 else -1

def _legendre_sequence(p, offset=0, length=None):
    """
    Generate a binary sequence from the Legendre symbol mod prime p.
    Maps +1 (QR) -> +1, -1 (QNR) -> -1, 0 -> 0 (but we skip 0).
    For a sequence of 'length', we take Legendre symbols of (offset+i) mod p,
    skipping 0 values by shifting.
    """
    if length is None:
        length = N
    seq = np.zeros(length, dtype=np.float64)
    count = 0
    i = offset
    while count < length:
        x = (offset + count) % p
        if x == 0:
            count += 1
            continue
        ls = _legendre_symbol(x, p)
        seq[count] = float(ls)
        count += 1
    return seq

def _find_primes_near(target, count=10):
    """Find primes near target value."""
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = []
    # Search above and below
    for delta in range(0, 200):
        for candidate in [target + delta, target - delta]:
            if candidate > 1 and is_prime(candidate) and candidate not in primes:
                primes.append(candidate)
        if len(primes) >= count:
            break
    return primes

def _greedy_local_search(h, time_budget=10.0, start_time=None):
    """
    Efficient greedy local search using exact ACF delta updates.
    For each candidate flip at position i, compute the delta to the ACF
    analytically and find the best flip without recomputing FFT.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(h)
    h = h.copy().astype(np.float64)
    acf = _linear_acf(h)
    center = n - 1
    current_psl = _psl_value(acf, n)
    best_h = h.copy()
    best_psl = current_psl
    
    # Precompute which lags are "active" (non-zero in ACF) for faster max tracking
    # We'll just recompute PSL after each accepted flip (cheap for n=400)
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try all single flips, pick the best one
        best_i = -1
        best_new_psl = current_psl + 1e-15
        
        for i in range(n):
            # Compute delta to ACF from flipping h[i]
            # For lag k > 0: delta[k] = -2*h[i]*(h[i+k] if i+k<n else 0) - 2*h[i]*(h[i-k] if i-k>=0 else 0)
            # For lag k < 0: symmetric
            # For lag k = 0: delta[0] = -4*h[i]^2 (but we don't care about lag 0 for PSL)
            
            # Efficiently compute the delta vector
            # h[i] is either +1 or -1 (or 0)
            hi = h[i]
            if hi == 0:
                continue  # Flipping 0 does nothing
            
            # Positive lags 1..n-1
            # delta[k] = -2*hi*(h[i+k]*1_{i+k<n} + h[i-k]*1_{i-k>=0})
            
            # We need to check if this flip reduces the max |acf[k]| for k != 0
            # Rather than building the full delta array each time, let's be smart:
            # The flip only affects lags k where i-k >= 0 or i+k < n
            # That's lags k in [max(1, i-(n-1)), min(n-1, n-1-i)] union symmetric
            
            # Actually for n=400, building the delta array is fast enough
            delta = np.zeros(2 * n - 1, dtype=np.float64)
            
            for k in range(1, n):
                term = 0.0
                if i + k < n:
                    term += h[i + k]
                if i - k >= 0:
                    term += h[i - k]
                d = -2.0 * hi * term
                delta[center + k] += d
                delta[center - k] += d
            
            new_acf = acf + delta
            new_psl = _psl_value(new_acf, n)
            
            if new_psl < best_new_psl:
                best_new_psl = new_psl
                best_i = i
        
        if best_i == -1 or best_new_psl >= current_psl - 1e-12:
            break
        
        # Apply the best flip
        h[best_i] = -h[best_i]
        acf = _linear_acf(h)  # Recompute for consistency
        current_psl = _psl_value(acf, n)
        
        if current_psl < best_psl:
            best_psl = current_psl
            best_h = h.copy()
        
        improved = True
        if not improved:
            break
    
    return best_h, float(DX * best_psl)

def _simulated_annealing(h, time_budget=8.0, start_time=None):
    """
    Simulated annealing for escaping local minima.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(h)
    current_h = h.copy().astype(np.float64)
    acf = _linear_acf(current_h)
    center = n - 1
    current_psl = _psl_value(acf, n)
    best_h = current_h.copy()
    best_psl = current_psl
    
    T = 2.0
    cooling = 0.997
    min_T = 0.05
    
    while T > min_T and time.time() - start_time < time_budget:
        # Random move: flip a random position
        i = np.random.randint(n)
        trial = current_h.copy()
        trial[i] = -trial[i] if trial[i] != 0 else (1.0 if np.random.random() < 0.5 else -1.0)
        
        # Compute new PSL
        new_acf = _linear_acf(trial)
        new_psl = _psl_value(new_acf, n)
        
        delta = new_psl - current_psl
        
        if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
            current_h = trial
            acf = new_acf
            current_psl = new_psl
            if new_psl < best_psl:
                best_psl = new_psl
                best_h = current_h.copy()
        
        T *= cooling
    
    return best_h, float(DX * best_psl)

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    
    Strategy:
    1. Generate candidates from Legendre symbol sequences (quadratic residues
       mod primes near 400) -- these have inherent low additive energy.
    2. Also include Thue-Morse and random initializations.
    3. Refine each candidate with greedy local search + simulated annealing.
    4. Return the best solution found within the time budget.
    """
    deadline = time.time() + 130  # Safety margin before 150s kill
    start_time = time.time()
    
    best_h = None
    best_score = float('inf')
    
    def try_candidate(h_init, greedy_time, sa_time):
        nonlocal best_h, best_score
        if time.time() > deadline:
            return
        h_opt, score_g = _greedy_local_search(h_init, time_budget=greedy_time, start_time=start_time)
        if time.time() < deadline - 2:
            h_opt, score_sa = _simulated_annealing(h_opt, time_budget=sa_time, start_time=start_time)
            if score_sa < score_g:
                score_g = score_sa
        if score_g < best_score:
            best_score = score_g
            best_h = h_opt.copy()
    
    # Phase 1: Legendre symbol sequences from primes near 400
    primes = _find_primes_near(N, count=15)
    for p in primes:
        if time.time() > deadline - 15:
            break
        # Try a few offsets for diversity
        for offset in [1, 2, 3]:
            if time.time() > deadline - 15:
                break
            h_leg = _legendre_sequence(p, offset=offset, length=N)
            # Ensure no zeros remain (shouldn't happen with our construction)
            h_leg[h_leg == 0] = 1.0
            try_candidate(h_leg, greedy_time=6.0, sa_time=5.0)
    
    # Phase 2: Thue-Morse sequence
    if time.time() < deadline - 15:
        t = np.zeros(N, dtype=np.float64)
        for i in range(N):
            count = bin(i).count('1')
            t[i] = 1.0 if count % 2 == 0 else -1.0
        try_candidate(t, greedy_time=6.0, sa_time=5.0)
    
    # Phase 3: Random full sequences
    for trial in range(8):
        if time.time() > deadline - 15:
            break
        h_rand = np.random.choice([-1, 1], size=N).astype(np.float64)
        try_candidate(h_rand, greedy_time=4.0, sa_time=4.0)
    
    # Phase 4: Chirp-based candidates
    if time.time() < deadline - 20:
        t = np.arange(N, dtype=np.float64)
        for root in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]:
            if time.time() > deadline - 15:
                break
            val = np.cos(2.0 * np.pi * root * t * t / N)
            h_chirp = np.sign(val)
            h_chirp[h_chirp == 0] = 1.0
            try_candidate(h_chirp, greedy_time=4.0, sa_time=3.0)
    
    # Phase 5: If time remains, do another round of SA on the best
    if best_h is not None and time.time() < deadline - 5:
        remaining = min(10.0, deadline - time.time() - 2)
        if remaining > 2:
            h_final, score_final = _simulated_annealing(best_h, time_budget=remaining, start_time=start_time)
            if score_final < best_score:
                best_score = score_final
                best_h = h_final.copy()
    
    if best_h is None:
        # Fallback: Thue-Morse
        t = np.zeros(N, dtype=np.float64)
        for i in range(N):
            count = bin(i).count('1')
            t[i] = 1.0 if count % 2 == 0 else -1.0
        best_h = t
    
    return best_h
