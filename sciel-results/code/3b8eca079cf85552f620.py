import numpy as np
import time

N = 400

def _score(h):
    # Fast PSL calculation
    # Zero pad to 2N
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # Return max absolute value for lags 1 to N-1
    return np.max(np.abs(acf[1:N]))

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    best_h = None
    best_score = float('inf')
    
    # 1. Generate initial candidates
    candidates = []
    
    # Random
    rng = np.random.default_rng(42)
    for i in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # Deterministic patterns
    # Alternating
    candidates.append(np.array([1.0 if i%2==0 else -1.0 for i in range(N)]))
    # Cosine based
    t = np.arange(N)
    candidates.append(np.sign(np.cos(2 * np.pi * t / N)))
    candidates.append(np.sign(np.cos(2 * np.pi * t * 3 / N)))
    
    # Evaluate and pick best start
    for h in candidates:
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # 2. Local Search / Simulated Annealing
    # We will use a greedy local search first, then SA if time permits.
    
    def _local_search(h, time_limit):
        h = h.copy()
        n = len(h)
        score = _score(h)
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            # Compute acf once
            fft_size = 1
            while fft_size < 2 * n:
                fft_size *= 2
            H = np.fft.rfft(h, n=fft_size)
            acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
            current_peak = np.max(np.abs(acf[1:n]))
            
            best_flip_pos = -1
            best_flip_score = current_peak
            
            # Try all positions? N=400 is small.
            # But recomputing score for each is O(N log N).
            # 400 * O(400 log 400) is fine.
            for i in range(n):
                old_val = h[i]
                h[i] = -old_val
                s = _score(h)
                if s < best_flip_score:
                    best_flip_score = s
                    best_flip_pos = i
                h[i] = old_val
                
            if best_flip_pos >= 0:
                h[best_flip_pos] = -h[best_flip_pos]
                score = best_flip_score
                improved = True
        return h, score

    best_h, best_score = _local_search(best_h, deadline)
    
    # 3. Random Restarts with SA
    # If time remains, do a few SA runs
    
    def _simulated_annealing(h, time_limit):
        h = h.copy()
        n = len(h)
        current_score = _score(h)
        best_h_sa = h.copy()
        best_score_sa = current_score
        
        T = 10.0
        cooling = 0.999
        
        while time.time() < time_limit:
            # Pick random position
            i = np.random.randint(n)
            old_val = h[i]
            h[i] = -old_val
            new_score = _score(h)
            
            delta = new_score - current_score
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_h_sa = h.copy()
            else:
                h[i] = old_val
                
            T *= cooling
            if T < 0.1:
                break
                
        return best_h_sa, best_score_sa

    # Run SA from best current
    if time.time() < deadline - 10:
        h_sa, s_sa = _simulated_annealing(best_h, deadline - 5)
        if s_sa < best_score:
            best_score = s_sa
            best_h = h_sa
            
    # Try a few random restarts with SA
    restarts = 5
    for i in range(restarts):
        if time.time() > deadline - 20:
            break
        rng = np.random.default_rng(100+i)
        h_init = rng.choice([-1.0, 1.0], size=N)
        h_sa, s_sa = _simulated_annealing(h_init, deadline - 5)
        if s_sa < best_score:
            best_score = s_sa
            best_h = h_sa

    return best_h
