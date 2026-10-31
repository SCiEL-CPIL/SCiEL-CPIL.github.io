import numpy as np
import time

N = 400

def _compute_bound(f):
    """
    Computes the peak sidelobe magnitude of the linear autocorrelation.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    PSD = F * np.conj(F)
    R = np.fft.irfft(PSD, n=n_fft)
    # R[0] is the peak (energy)
    # R[1:N] are the sidelobes
    return np.max(np.abs(R[1:N]))

def _gen_m_sequence(k, poly_taps):
    """
    Generates an m-sequence of length 2^k - 1 using specified taps.
    poly_taps: list of bit positions for feedback (LSB is 0).
    """
    length = (1 << k) - 1
    state = 1
    seq = []
    mask = (1 << k) - 1
    
    for _ in range(length):
        seq.append(state & 1)
        # Calculate feedback
        fb = 0
        for tap in poly_taps:
            fb ^= (state >> tap) & 1
        state = (state >> 1) | (fb << (k - 1))
        state &= mask
    
    return 2 * np.array(seq, dtype=int) - 1

def _local_search(f, start_time, time_limit):
    """
    Performs local search by flipping bits to reduce the peak sidelobe.
    """
    current_f = f.copy()
    current_bound = _compute_bound(current_f)
    best_bound = current_bound
    best_f = current_f.copy()
    
    n = len(current_f)
    
    # To speed up, we can try a few random flips per iteration
    # Or systematically scan a subset
    
    while time.time() < time_limit:
        improved = False
        # Try flipping each bit once
        # This is O(N * N log N) per pass, which is 400 * 400log400 ~ 1.5M ops.
        # 1.5M ops is very fast in Python/NumPy? No, 400 FFTs is the cost.
        # 400 FFTs of size 800 is ~400 * 10us = 4ms.
        # So one full pass is ~4ms.
        # We can do ~2500 passes in 10 seconds.
        
        for i in range(n):
            if time.time() > time_limit:
                break
            
            # Flip bit i
            old_val = current_f[i]
            current_f[i] = -old_val
            
            new_bound = _compute_bound(current_f)
            
            if new_bound < current_bound:
                current_bound = new_bound
                if current_bound < best_bound:
                    best_bound = current_bound
                    best_f = current_f.copy()
                improved = True
            else:
                # Revert
                current_f[i] = old_val
        
        if not improved:
            # Stuck in local minimum, break out or do a random restart?
            # For now, break to save time for other candidates
            break
            
    return best_f, best_bound

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    candidates = []
    
    # 1. M-sequences with different primitive polynomials for k=9
    # Primitive polynomials of degree 9
    # x^9 + x^5 + 1 (taps 9,5,0 -> feedback from 5 and 0? No, standard form)
    # Let's use a few known ones.
    # x^9 + x^5 + 1
    candidates.append(_gen_m_sequence(9, [5, 0]))
    # x^9 + x^4 + 1
    candidates.append(_gen_m_sequence(9, [4, 0]))
    # x^9 + x^8 + x^6 + x^5 + x^3 + x^2 + x + 1 (not primitive probably)
    
    # 2. Golay sequence (length 512)
    g = np.array([1], dtype=int)
    for _ in range(9):
        g = np.concatenate([g, -g])
    candidates.append(g[:N])
    
    # 3. Random sequences
    for seed in [42, 123, 999]:
        np.random.seed(seed)
        candidates.append(np.random.choice([-1, 1], size=N))
        
    best_f = None
    best_score = float('inf')
    
    # Allocate time to each candidate
    # We have ~110s. Let's give each candidate a slice.
    # But local search converges fast.
    # Let's run local search on the best initial candidates first.
    
    # Initial evaluation to sort candidates
    initial_scores = []
    for f in candidates:
        score = _compute_bound(f)
        initial_scores.append((score, f))
        
    # Sort by initial score (lower is better)
    initial_scores.sort(key=lambda x: x[0])
    
    # Run local search on top candidates
    # Give 30s to the first, 20s to the second, etc.
    time_allocations = [30, 20, 15, 10, 5, 5]
    
    for i, (init_score, f) in enumerate(initial_scores):
        if i >= len(time_allocations):
            break
        if time.time() > deadline - 5: # Leave 5s for finalization
            break
            
        alloc = time_allocations[i]
        search_deadline = min(time.time() + alloc, deadline - 5)
        
        # If the initial score is already very good, maybe skip?
        # No, always try to improve.
        
        improved_f, improved_score = _local_search(f, time.time(), search_deadline)
        
        if improved_score < best_score:
            best_score = improved_score
            best_f = improved_f
    
    if best_f is None:
        best_f = initial_scores[0][1]
        
    return best_f
