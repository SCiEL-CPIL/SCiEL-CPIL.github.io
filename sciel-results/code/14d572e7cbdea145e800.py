import numpy as np
import time

def _autocorr_and_score(h):
    n = len(h)
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h, size)
    power = F * np.conj(F)
    corr = np.fft.irfft(power, size)[:n]
    score = np.max(np.abs(corr[1:])) if n > 1 else 0.0
    return corr, score

def _lfsr_sequence(n, poly, init_state):
    degree = poly.bit_length() - 1
    mask = (1 << degree) - 1
    state = init_state & mask
    seq = []
    for _ in range(n):
        bit = state & 1
        seq.append(1.0 if bit else -1.0)
        fb = 0
        for i in range(degree):
            if (poly >> i) & 1:
                fb ^= (state >> i) & 1
        state = (state >> 1) | (fb << (degree - 1))
        state &= mask
    return np.array(seq)

def _sa_optimize(h_init, time_limit, seed=0):
    n = len(h_init)
    h = h_init.copy()
    corr, score = _autocorr_and_score(h)
    best_h = h.copy()
    best_score = score
    
    rng = np.random.RandomState(seed)
    T = 10.0
    T_min = 0.1
    cooling = 0.999 # Adjust based on steps
    
    start_time = time.time()
    steps = 0
    
    while time.time() < start_time + time_limit:
        # Pick random index
        i = rng.randint(n)
        
        # Calculate delta
        # v[k] = h[i-k] + h[i+k]
        # We need to handle bounds.
        # k ranges from 0 to n-1.
        # i-k >= 0 => k <= i
        # i+k < n => k < n-i
        
        # Create vector v of length n
        v = np.zeros(n)
        
        # Part 1: h[i-k] for k=0..i
        if i >= 0:
            v[0:i+1] = h[i::-1] # h[i], h[i-1], ..., h[0]
            
        # Part 2: h[i+k] for k=0..n-i-1
        if i < n:
            v[0:n-i] += h[i:] # h[i], h[i+1], ..., h[n-1]
            
        # Note: v[0] gets h[i] + h[i] = 2*h[i]
        # Delta C(k) = -2 * h[i] * v[k]
        
        delta_corr = -2.0 * h[i] * v
        
        new_corr = corr + delta_corr
        new_score = np.max(np.abs(new_corr[1:]))
        
        delta_score = new_score - score
        
        if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
            # Accept
            h[i] = -h[i]
            corr = new_corr
            score = new_score
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
        else:
            # Revert (do nothing to h, corr, score)
            pass
            
        steps += 1
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Check time more frequently in tight loops? 
        # The while condition is sufficient.
        
    return best_h, best_score

def construct():
    N = 400
    deadline = time.time() + 110
    
    best_h = None
    best_score = float('inf')
    
    # Generate candidates
    candidates = []
    
    # 1. LFSR candidates
    polys = [0x119, 0x12D, 0x15D, 0x169, 0x199]
    for poly in polys:
        for init in range(1, 8):
            h = _lfsr_sequence(N, poly, init)
            candidates.append(h)
            
    # 2. Random candidates
    rng = np.random.RandomState(42)
    for _ in range(5):
        h = rng.choice([1.0, -1.0], N)
        candidates.append(h)
        
    # 3. Alternating (worst case, but good for structure?)
    h_alt = np.ones(N)
    h_alt[1::2] = -1.0
    candidates.append(h_alt)
    
    # Optimize each candidate with a time slice
    # We have ~110s. Let's give each candidate a chance.
    # If we have 30 candidates, 3.6s each.
    # Or we can do a few full runs.
    # Let's do 10 restarts, each getting up to 10s.
    
    num_restarts = min(len(candidates), 10)
    time_per_restart = 10.0
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        h_init = candidates[i]
        h_opt, score_opt = _sa_optimize(h_init, time_per_restart, seed=i)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
            
        # If we still have time, do another random restart
        if time.time() < deadline - 15:
             h_rand = rng.choice([1.0, -1.0], N)
             h_opt, score_opt = _sa_optimize(h_rand, time_per_restart, seed=i+100)
             if score_opt < best_score:
                 best_score = score_opt
                 best_h = h_opt

    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
