import numpy as np
import time

N = 400

def _compute_metric(h):
    """
    Compute the upper bound on C3, defined here as the peak sidelobe 
    of the periodic autocorrelation normalized by N.
    """
    if len(h) != N:
        return np.inf
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    # R[0] is zero-lag (N). We look at non-zero lags.
    # Due to floating point errors, R[0] might be N + eps.
    # We take max of absolute values of R[1:]
    max_abs = np.max(np.abs(R[1:]))
    return max_abs / N

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    seq = np.zeros(n, dtype=np.float64)
    seq[0] = 1.0
    for i in range(1, n):
        # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
        # Or recursive: t(2n) = t(n), t(2n+1) = 1 - t(n)
        # Using bit count for simplicity in vectorized way is hard, so loop or use precomputed
        pass
    
    # Efficient generation
    t = np.zeros(n, dtype=np.int8)
    t[0] = 1
    for i in range(1, n):
        t[i] = 1 - t[i // 2]
    return 2.0 * t - 1.0 # Map 0->-1, 1->1

def _hadamard_row(n):
    """
    Generate a row of a Hadamard matrix of size >= n, truncated to n.
    Hadamard rows have perfect autocorrelation for full length.
    """
    # Find smallest power of 2 >= n
    m = 1
    while m < n:
        m *= 2
        
    # Generate Sylvester Hadamard matrix row 1 (or any non-zero row)
    # Row k of Sylvester Hadamard is given by (-1)^(popcount(i & k))
    # Let's pick a random row index k != 0
    np.random.seed(42)
    k = np.random.randint(1, m)
    
    i = np.arange(m)
    # Compute popcount of (i & k)
    # We can use a lookup table or bit manipulation
    # For speed, we can use numpy bitwise operations if we had popcount, but we don't.
    # However, m is 512. Loop is fast enough.
    seq = np.ones(m, dtype=np.float64)
    for idx in range(m):
        val = idx & k
        # Count bits
        c = 0
        v = val
        while v:
            c += v & 1
            v >>= 1
        if c % 2 != 0:
            seq[idx] = -1.0
            
    return seq[:n]

def _legendre_sequence(n):
    """Generate a Legendre sequence of length n."""
    def is_prime(m):
        if m < 2: return False
        if m == 2: return True
        if m % 2 == 0: return False
        for i in range(3, int(np.sqrt(m)) + 1, 2):
            if m % i == 0: return False
        return True
    
    p = n
    while not is_prime(p):
        p += 1
        
    seq = np.ones(p, dtype=np.float64)
    for i in range(1, p):
        ls = pow(i, (p - 1) // 2, p)
        if ls == p - 1:
            seq[i] = -1
            
    if p >= n:
        return seq[:n]
    else:
        result = np.ones(n, dtype=np.float64)
        result[:p] = seq
        return result

def _sa_search(h, deadline):
    """
    Simulated Annealing with block flip moves.
    """
    h = h.copy()
    current_score = _compute_metric(h)
    best_h = h.copy()
    best_score = current_score
    
    # Cooling schedule
    T_start = 0.5
    T_end = 0.001
    # We will run for a fixed number of iterations or until deadline
    # Estimate time per iteration
    start_t = time.time()
    
    # Determine number of iterations we can afford
    # Roughly 1000 iterations per second for FFT of size 400? 
    # Let's just run until deadline with a check every 100 iterations.
    
    T = T_start
    # Number of steps to decay T
    # We don't know total steps in advance, so we use a time-based decay or step-based.
    # Let's assume we can do ~5000 steps in 10 seconds. 
    # 150s -> 75000 steps.
    # Let's target 50000 steps.
    total_steps = 50000
    # If we run out of time, we stop.
    
    for step in range(total_steps):
        if time.time() > deadline - 5:
            break
            
        # Decay temperature
        # Exponential decay
        # T = T_start * (T_end / T_start) ** (step / total_steps)
        T = T_start * (T_end / T_start) ** (step / total_steps)
        
        # Select a block to flip
        # Block length L between 1 and 20
        L = np.random.randint(1, 21)
        start_idx = np.random.randint(0, N - L + 1)
        
        # Flip the block
        h[start_idx:start_idx+L] *= -1
        
        new_score = _compute_metric(h)
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            # Revert
            h[start_idx:start_idx+L] *= -1
            
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    np.random.seed(42)
    
    candidates = []
    
    # 1. Thue-Morse
    candidates.append(_thue_morse(N))
    
    # 2. Hadamard Row
    candidates.append(_hadamard_row(N))
    
    # 3. Legendre
    candidates.append(_legendre_sequence(N))
    
    # 4. Random
    candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # 5. Another Random
    np.random.seed(123)
    candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Evaluate candidates
    best_h = None
    best_score = np.inf
    
    for c in candidates:
        c = np.asarray(c, dtype=np.float64)
        if len(c) != N:
            continue
        s = _compute_metric(c)
        if s < best_score:
            best_score = s
            best_h = c.copy()
            
    if best_h is None:
        best_h = np.ones(N, dtype=np.float64)
        best_score = _compute_metric(best_h)
        
    # Run Simulated Annealing
    # We can run multiple restarts if time permits
    # But SA is stochastic, so one long run is often better than many short ones.
    # Let's do one long run from the best candidate.
    
    best_h, best_score = _sa_search(best_h, deadline)
    
    # If we have time left, try another restart from a different candidate
    if time.time() < deadline - 20:
        # Pick the second best candidate
        scores = []
        for c in candidates:
            c = np.asarray(c, dtype=np.float64)
            if len(c) == N:
                scores.append((_compute_metric(c), c))
        scores.sort(key=lambda x: x[0])
        
        if len(scores) > 1:
            h2, s2 = scores[1]
            h2, s2 = _sa_search(h2, deadline)
            if s2 < best_score:
                best_h = h2
                best_score = s2
                
    return best_h
