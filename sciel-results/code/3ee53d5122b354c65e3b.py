import numpy as np
import time

N = 400

def _thue_morse_sequence(n):
    """Generate Thue-Morse sequence of length n."""
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Thue-Morse: (-1)^(number of 1s in binary representation of i)
        # Using bit_count is faster in Python 3.10+, but bin().count is safe
        # For speed, we can generate recursively
        pass
    
    # Recursive generation is faster
    seq = np.array([1.0])
    while len(seq) < n:
        seq = np.concatenate([seq, -seq])
    return seq[:n]

def _make_legendre_sequence(p, offset=0, n=N):
    """Create a ±1 sequence of length n using Legendre symbols mod p."""
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x = (i + offset) % p
        if x == 0:
            seq[i] = 1.0
        else:
            ls = pow(x, (p - 1) // 2, p)
            seq[i] = 1.0 if ls == 1 else -1.0
    return seq

def _calc_3rd_autocorr(x):
    """Calculate the 3rd order autocorrelation values for all lags t."""
    n = len(x)
    A3 = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        L = n - 2 * t
        if L <= 0:
            break
        # Vectorized sum of products
        # x[0:L] * x[t:t+L] * x[2*t:2*t+L]
        A3[t] = np.sum(x[:L] * x[t:t+L] * x[2*t:2*t+L])
    return A3

def _score_3rd_autocorr(x):
    """Calculate the upper bound (max absolute value) of 3rd autocorrelation."""
    A3 = _calc_3rd_autocorr(x)
    # Exclude lag 0 (which is n) and only consider t > 0
    if len(A3) > 1:
        return np.max(np.abs(A3[1:]))
    return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_seq = None
    best_score = float('inf')
    
    # 1. Generate Initial Candidates
    
    # Thue-Morse
    seq_tm = _thue_morse_sequence(N)
    score_tm = _score_3rd_autocorr(seq_tm)
    if score_tm < best_score:
        best_score = score_tm
        best_seq = seq_tm.copy()
        
    # Legendre Symbols for primes near N
    primes_to_try = [397, 401, 409, 419, 421, 431, 433, 439, 443, 449]
    for p in primes_to_try:
        # Try a few offsets
        for offset in range(5):
            seq_leg = _make_legendre_sequence(p, offset)
            score_leg = _score_3rd_autocorr(seq_leg)
            if score_leg < best_score:
                best_score = score_leg
                best_seq = seq_leg.copy()
                
        if time.time() > deadline - 60:
            break
            
    # Random Sequences
    np.random.seed(42)
    for _ in range(10):
        seq_rand = np.random.choice([-1.0, 1.0], N)
        score_rand = _score_3rd_autocorr(seq_rand)
        if score_rand < best_score:
            best_score = score_rand
            best_seq = seq_rand.copy()

    if best_seq is None:
        best_seq = np.random.choice([-1.0, 1.0], N)
        best_score = _score_3rd_autocorr(best_seq)

    # 2. Local Search / Simulated Annealing
    
    current_seq = best_seq.copy()
    current_score = best_score
    best_seq = current_seq.copy()
    
    # Simulated Annealing parameters
    temperature = 1.0
    cooling = 0.999
    min_temperature = 0.01
    
    iteration = 0
    
    while time.time() < deadline - 5.0:
        # Pick a random index to flip
        idx = np.random.randint(N)
        
        # Flip
        old_val = current_seq[idx]
        current_seq[idx] = -old_val
        
        # Calculate new score
        new_score = _score_3rd_autocorr(current_seq)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (temperature > min_temperature and np.random.rand() < np.exp(-delta / temperature)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_seq = current_seq.copy()
            temperature *= cooling
        else:
            # Revert
            current_seq[idx] = old_val
            temperature *= cooling
            
        iteration += 1
        # Check time periodically
        if iteration % 1000 == 0:
            if time.time() > deadline - 10.0:
                break

    return best_seq
