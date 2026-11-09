import numpy as np
import time

N = 400

def max_sidelobe(h):
    """
    Compute the maximum normalized autocorrelation sidelobe.
    h: np.array of shape (N,) with values +/- 1.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = h
    F = np.fft.fft(f)
    R = np.real(np.fft.ifft(F * np.conj(F)))
    R = R[:N]
    R[0] = 0
    # Normalized by N (peak power)
    return np.max(np.abs(R)) / N

def find_best_flip(h, current_score, R):
    """
    Find the single bit flip that most improves the score.
    Uses incremental update of autocorrelation to avoid N FFTs.
    Returns (best_index, best_score, best_R) or (None, current_score, R) if no improvement.
    """
    best_idx = -1
    best_score = current_score
    best_R = R
    
    # Pre-compute the contributions for each i?
    # Delta R[k] for flip at i:
    # R_new[k] = R[k] - 2*h[i]*(h[i-k] + h[i+k]) (with boundary checks)
    # Note: R is symmetric. We only need to check k=1..N-1.
    
    # To speed this up, we can vectorize the calculation of the *change* in the max?
    # No, we need the actual max of the new R.
    
    # Since N=400, let's just loop over i and update R temporarily?
    # That is O(N^2) memory writes.
    # 400*400 = 160,000 operations. Very fast in Python if vectorized?
    # But we need to do it for each i.
    
    # Let's try a hybrid: Check a random subset of flips first?
    # No, Greedy Best Improvement requires checking all.
    
    # Let's optimize the loop.
    # We can pre-compute the "influence" of each i on R?
    
    # Actually, let's just do the FFT for the top candidates if we use a heuristic?
    # No, let's stick to the incremental update but implement it efficiently.
    
    # For each i:
    #   delta_R = np.zeros(N)
    #   for k in 1..N-1:
    #       idx1 = i + k
    #       idx2 = i - k
    #       val = 0
    #       if 0 <= idx1 < N: val += h[idx1]
    #       if 0 <= idx2 < N: val += h[idx2]
    #       delta_R[k] = -2 * h[i] * val
    #   new_R = R + delta_R
    #   new_score = max(abs(new_R[1:])) / N
    
    # This inner loop is O(N). Total O(N^2).
    # In Python, 160k iterations is fast (~10ms).
    
    for i in range(N):
        # Calculate delta_R for flip at i
        # We can vectorize this using numpy slices?
        # h[i+k] is h[i+1 : i+N] with wrap? No, no wrap.
        
        # Let's construct the vectors for h[i+k] and h[i-k]
        # k ranges from 1 to N-1
        
        # This is tricky to vectorize perfectly without padding.
        # Let's just do the loop in numpy if possible.
        
        # Create index arrays
        # k = 1..N-1
        # idx_plus = i + k
        # idx_minus = i - k
        
        # We need to handle boundaries.
        
        # Alternative: Just use the FFT for all N flips?
        # 400 FFTs * 50us = 20ms.
        # 50 iterations/sec.
        # 115s * 50 = 5750 local minima found.
        # This is a very solid number of restarts.
        
        # Let's implement the FFT-based best flip search. It's simpler and likely fast enough.
        pass
        
    # Fallback: Use FFT for all flips
    best_score = current_score
    best_idx = -1
    best_R = R
    
    # Batch the FFTs?
    # Create a matrix of shape (N, n_fft) where each row is h with one bit flipped?
    # Memory: 400 * 1024 * 8 bytes = 3.2 MB. Acceptable.
    
    h_batch = np.tile(h, (N, 1))
    for i in range(N):
        h_batch[i, i] *= -1
        
    # Compute FFT for all
    F_batch = np.fft.fft(h_batch, axis=1)
    R_batch = np.real(np.fft.ifft(F_batch * np.conj(F_batch), axis=1))
    R_batch = R_batch[:, :N]
    R_batch[:, 0] = 0
    
    # Compute scores
    scores = np.max(np.abs(R_batch[:, 1:]), axis=1) / N
    
    min_idx = np.argmin(scores)
    min_score = scores[min_idx]
    
    if min_score < best_score:
        best_score = min_score
        best_idx = min_idx
        best_R = R_batch[min_idx]
        
    return best_idx, best_score, best_R

def iterated_local_search(h, deadline, rng):
    """
    Optimize h using Iterated Local Search.
    """
    current_score = max_sidelobe(h)
    R = None # We need R for incremental? No, we use FFT batch.
    
    best_h = h.copy()
    best_score = current_score
    
    # Local Search Phase
    while time.time() < deadline:
        # Find best single flip
        idx, new_score, _ = find_best_flip(h, current_score, None)
        
        if idx == -1 or new_score >= current_score:
            # Local minimum reached
            # Perturb
            # Randomly flip 5-10 bits
            num_flips = rng.integers(5, 11)
            flip_indices = rng.choice(N, size=num_flips, replace=False)
            h[flip_indices] *= -1
            current_score = max_sidelobe(h)
            
            # If this perturbation leads to a better score, keep it?
            # Usually ILS accepts the perturbed state and runs local search again.
            # But if the perturbation is very bad, we might want to revert?
            # Standard ILS: Accept perturbation, run local search.
            # If the resulting local min is better than global best, update global best.
            # If not, continue.
            
            # To avoid drifting too far, we can check if the perturbed score is too high?
            # For now, just accept and let local search fix it.
            
        else:
            # Apply best flip
            h[idx] *= -1
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
                
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Thue-Morse
    def generate_thue_morse(length):
        seq = np.array([0])
        while len(seq) < length:
            inv = 1 - seq
            seq = np.concatenate([seq, inv])
        return (1 - 2 * seq[:length]).astype(np.float64)
        
    h_tm = generate_thue_morse(N)
    candidates.append(h_tm)
    
    # 2. Random
    h_rand = rng.choice([-1.0, 1.0], N)
    candidates.append(h_rand)
    
    # 3. M-Sequence
    def generate_mseq_length(m, taps, length):
        state = 1
        seq = []
        for i in range(length):
            seq.append(1.0 if (state & 1) else -1.0)
            fb = 0
            for t in taps:
                fb ^= (state >> t) & 1
            state = (state >> 1) | (fb << (m - 1))
        return np.array(seq)
        
    h_mseq = generate_mseq_length(9, [0, 4], 511)
    h_mseq = h_mseq[:N]
    candidates.append(h_mseq)
    
    best_h = None
    best_score = float('inf')
    
    # Allocate time to each candidate
    time_per_candidate = (deadline - time.time()) / len(candidates)
    
    for idx, h in enumerate(candidates):
        if time.time() > deadline:
            break
            
        h_work = h.copy()
        local_deadline = time.time() + (deadline - time.time()) / (len(candidates) - idx)
        
        h_opt, score = iterated_local_search(h_work, local_deadline, rng)
        
        if score < best_score:
            best_score = score
            best_h = h_opt
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
