import numpy as np
import time

N = 400

def compute_score(h):
    """
    Compute the maximum normalized autocorrelation sidelobe.
    h: np.array of shape (N,) with values +/- 1.
    """
    # np.correlate(h, h, 'full') gives length 2N-1
    # We want lags 0 to N-1.
    # R[k] = sum h[j] h[j+k]
    # np.correlate output index N-1+k corresponds to lag k.
    R = np.correlate(h, h, mode='full')
    R_sidelobes = R[N-1:N-1+N]
    R_sidelobes[0] = 0 # Ignore peak at lag 0
    return np.max(np.abs(R_sidelobes)) / N

def simulated_annealing(h_init, deadline, rng, max_steps=50000):
    """
    Perform Simulated Annealing with single-bit flips.
    """
    h = h_init.copy()
    current_score = compute_score(h)
    best_h = h.copy()
    best_score = current_score
    
    T_start = 0.5
    T_end = 0.001
    # Number of steps per temperature level
    steps_per_temp = 100
    
    # Total steps determined by time or max_steps
    # We will use a dynamic loop based on deadline
    
    T = T_start
    # Geometric cooling factor to reach T_end in roughly max_steps / steps_per_temp levels
    # Levels = max_steps / steps_per_temp
    # T_end = T_start * alpha^Levels
    # alpha = (T_end / T_start)^(1/Levels)
    levels = max(1, max_steps // steps_per_temp)
    alpha = (T_end / T_start) ** (1.0 / levels)
    
    step = 0
    while step < max_steps and time.time() < deadline:
        # Try a random flip
        i = rng.integers(0, N)
        old_val = h[i]
        h[i] = -old_val
        
        new_score = compute_score(h)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept move
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            # Reject move
            h[i] = old_val
            
        step += 1
        # Cool down every steps_per_temp steps
        if step % steps_per_temp == 0:
            T *= alpha
            if T < T_end:
                T = T_end
                # Optional: break if we want to stop early, but let's finish the budget
                # Actually, if T is very low, we are effectively doing local search.
                
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Random Start
    h_rand = rng.choice([-1.0, 1.0], N)
    candidates.append(h_rand)
    
    # 2. Thue-Morse Sequence
    def generate_thue_morse(length):
        seq = np.array([0])
        while len(seq) < length:
            inv = 1 - seq
            seq = np.concatenate([seq, inv])
        return (1 - 2 * seq[:length]).astype(np.float64)

    h_tm = generate_thue_morse(N)
    candidates.append(h_tm)

    # 3. M-Sequence Start (Length 511, truncated to 400)
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
        
    # Taps for x^9+x^4+1: bits 0 and 4
    h_mseq = generate_mseq_length(9, [0, 4], 511)
    h_mseq = h_mseq[:N]
    candidates.append(h_mseq)
    
    best_h = None
    best_score = float('inf')
    
    # We will run SA for each candidate, but limit the time per candidate.
    # Since SA is stochastic, running multiple short SA runs might be better than one long one?
    # Or one long SA run per candidate.
    # Let's split time evenly among candidates for the first pass.
    # Then, if time remains, continue SA from the best solution found so far.
    
    time_per_candidate = (deadline - time.time()) / len(candidates)
    
    for idx, h in enumerate(candidates):
        if time.time() > deadline:
            break
            
        h_work = h.copy()
        local_deadline = time.time() + time_per_candidate
        
        # Limit steps to ensure we don't overshoot deadline significantly in compute_score
        # 50000 steps is roughly 50000 * (O(N) correlate) which is fast for N=400.
        # np.correlate for N=400 is very fast.
        h_opt, score = simulated_annealing(h_work, local_deadline, rng, max_steps=20000)
        
        if score < best_score:
            best_score = score
            best_h = h_opt
            
    # If we have time left, do a final polish on the best solution found
    if best_h is not None and time.time() < deadline:
        # Run SA again from the best solution with a lower initial temperature to fine-tune
        h_work = best_h.copy()
        # Adjust deadline to be the global deadline
        # Use a higher max_steps to try to exploit the remaining time
        remaining_time = deadline - time.time()
        # Estimate steps possible: assume ~1ms per step (conservative for N=400 correlate)
        # Actually, let's just use a large number and let the deadline guard stop it.
        h_opt, score = simulated_annealing(h_work, deadline, rng, max_steps=100000)
        
        if score < best_score:
            best_score = score
            best_h = h_opt
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
