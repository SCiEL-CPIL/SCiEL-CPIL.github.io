import numpy as np
import time

N = 400
TIME_LIMIT = 120  # Safety margin before 150s kill (evaluator kills at 150s, we stop at 120s to be safe)
FFT_SIZE = 1024  # Next power of 2 >= 2*400 - 1 = 799

def _compute_acf_full(h):
    """
    Compute the aperiodic autocorrelation of h.
    Returns array of length N where acf[k] = sum_{i=0}^{N-1-k} h[i] h[i+k] for k=0..N-1.
    This covers lags 0 to N-1. The full autocorrelation is symmetric.
    """
    h_pad = np.zeros(FFT_SIZE)
    h_pad[:N] = h
    H = np.fft.fft(h_pad)
    P = H * np.conj(H)
    acf = np.fft.ifft(P).real
    # The first N lags are 0..N-1
    return acf[:N]

def _score(h):
    """
    Compute the upper bound on C3.
    Based on inspirations, this is dx * max |autoconvolution|.
    The autoconvolution is symmetric, so max over lags 0..N-1 is sufficient.
    """
    acf = _compute_acf_full(h)
    max_abs = np.max(np.abs(acf))
    dx = 0.5 / N
    return dx * max_abs

def _local_search(h, time_limit=10.0):
    """
    Local search using incremental ACF updates.
    """
    start_time = time.time()
    current_h = h.copy()
    current_acf = _compute_acf_full(current_h)
    current_score = _score(current_h)
    
    best_h = current_h.copy()
    best_score = current_score
    
    improved = True
    passes = 0
    max_passes = 10000
    
    while improved and passes < max_passes:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        passes += 1
        
        best_gain = 1e-12  # Small epsilon
        best_idx = -1
        
        for j in range(N):
            # Calculate delta for each lag k in 0..N-1
            # Delta A_k = -2 * h[j] * (h[j+k] * (j+k<N) + h[j-k] * (j-k>=0))
            
            delta_acf = np.zeros(N)
            hj = current_h[j]
            
            # Term 1: h[j+k] for k where j+k < N
            # k ranges from 0 to N-1-j
            # h[j+k] corresponds to current_h[j : N]
            if N - j > 0:
                delta_acf[:N-j] -= 2 * hj * current_h[j:N]
            
            # Term 2: h[j-k] for k where j-k >= 0
            # k ranges from 0 to j
            # h[j-k] corresponds to current_h[j::-1]
            if j + 1 > 0:
                delta_acf[:j+1] -= 2 * hj * current_h[j::-1]
            
            new_acf = current_acf + delta_acf
            new_max_abs = np.max(np.abs(new_acf))
            new_score = (0.5 / N) * new_max_abs
            
            gain = current_score - new_score
            
            if gain > best_gain:
                best_gain = gain
                best_idx = j
        
        if best_idx >= 0:
            # Apply the best flip
            current_h[best_idx] *= -1
            # Recompute ACF for accuracy (O(N log N) is fine, or we could apply delta, 
            # but recomputing avoids drift errors. Given N=400, FFT is ~10us. 
            # 400 flips * 10us = 4ms per pass. 10000 passes = 40s. 
            # However, we are doing 400 trials per pass. 
            # 400 trials * (O(N) update + O(N) max) = 400 * 800 ops = 320k ops per pass.
            # This is much faster than 400 FFTs.
            # So we should apply the delta to current_acf.
            
            # Re-apply the delta for best_idx to update current_acf
            hj_old = -current_h[best_idx] # The value before flip
            # Wait, we flipped current_h. 
            # Let's just recompute ACF. It's simpler and robust. 
            # Actually, applying delta is safer if we stored it. 
            # Let's recompute ACF. 1024 FFT is very fast.
            current_acf = _compute_acf_full(current_h)
            current_score = (0.5 / N) * np.max(np.abs(current_acf))
            
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
            improved = True
            
    return best_h, best_score

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return np.array(seq[:n], dtype=np.float64)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_h = None
    best_score = np.inf
    
    # 1. Thue-Morse Initialization
    tm = _thue_morse(N)
    tm_score = _score(tm)
    if tm_score < best_score:
        best_score = tm_score
        best_h = tm.copy()
    
    # 2. Random Initializations
    np.random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        h = np.random.choice([-1.0, 1.0], size=N)
        score = _score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # 3. Local Search from Best Start
    if best_h is not None:
        time_left = deadline - time.time() - 5
        if time_left > 5:
            best_h, best_score = _local_search(best_h, time_limit=time_left)
    
    # 4. Additional Restarts if Time Permits
    if time.time() < deadline - 20:
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            h = np.random.choice([-1.0, 1.0], size=N)
            score = _score(h)
            
            # Only do local search if this start is promising or we have lots of time
            # Since local search is fast, we can try more
            time_left = deadline - time.time() - 5
            if time_left > 5:
                h2, score2 = _local_search(h, time_limit=time_left)
                if score2 < best_score:
                    best_score = score2
                    best_h = h2
    
    # Final safety check
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Score: {_score(h)}")
