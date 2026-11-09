"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def _fast_overlap_score(h):
    """
    Computes the maximum overlap of h with shifted 1-h over linear shifts.
    Uses FFT for speed.
    """
    one_minus_h = 1.0 - h
    # Use FFT-based cross-correlation for speed
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = np.fft.rfft(h, size)
    G = np.fft.rfft(one_minus_h, size)
    # Cross-correlation: sum_i h[i] * one_minus_h[i+k] for k=0..n-1
    # This is the 'valid' mode of correlate
    corr = np.fft.irfft(H * np.conj(G), size)[:n]
    return np.max(corr)

def _deterministic_polish(binary, time_limit=2.0):
    """
    Deterministic hill-climbing: try all (i,j) swap pairs where h[i]=1, h[j]=0.
    Accept any improving move. Repeat until no improvement or time limit.
    """
    current = binary.copy()
    current_score = _fast_overlap_score(current)
    best = current.copy()
    best_score = current_score
    
    start_time = time.time()
    
    while (time.time() - start_time) < time_limit:
        improved = False
        ones = np.where(current == 1.0)[0]
        zeros = np.where(current == 0.0)[0]
        
        # Try all pairs, but limit per pass to avoid timeout
        max_pairs = 5000
        n_pairs = len(ones) * len(zeros)
        
        if n_pairs > max_pairs:
            # Sample a subset
            idx = np.random.choice(n_pairs, max_pairs, replace=False)
            i_list = []
            j_list = []
            for k in idx:
                i_list.append(ones[k // len(zeros)])
                j_list.append(zeros[k % len(zeros)])
        else:
            i_list = []
            j_list = []
            for i in ones:
                for j in zeros:
                    i_list.append(i)
                    j_list.append(j)
        
        for i, j in zip(i_list, j_list):
            if (time.time() - start_time) > time_limit:
                break
            
            current[i] = 0.0
            current[j] = 1.0
            new_score = _fast_overlap_score(current)
            
            if new_score < best_score - 1e-12:
                best_score = new_score
                best = current.copy()
                improved = True
            else:
                current[i] = 1.0
                current[j] = 0.0
        
        if not improved:
            break
    
    return best

def _local_search_binary(binary, time_limit=5.0):
    """
    Perform local search on a binary sequence to minimize max linear overlap.
    Uses swap-based moves to maintain sum = N/2.
    """
    current = binary.copy()
    current_score = _fast_overlap_score(current)
    best = current.copy()
    best_score = current_score
    
    start_time = time.time()
    
    while (time.time() - start_time) < time_limit:
        # Find a 1 and a 0 to swap
        ones = np.where(current == 1.0)[0]
        zeros = np.where(current == 0.0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        
        current[i] = 0.0
        current[j] = 1.0
        
        new_score = _fast_overlap_score(current)
        
        if new_score < best_score:
            best_score = new_score
            best = current.copy()
            current_score = new_score
        else:
            # Revert
            current[i] = 1.0
            current[j] = 0.0
            
    return best

def _structured_init():
    """
    Create a structured initial pattern: alternating blocks to reduce overlap.
    """
    h = np.zeros(N)
    # Pattern: groups of consecutive 1s and 0s with varying lengths
    # Start with a base pattern and perturb
    block_size = 4
    for i in range(N):
        block_idx = i // block_size
        if block_idx % 2 == 0:
            h[i] = 1.0
        else:
            h[i] = 0.0
    
    # Ensure exactly N/2 ones
    ones_count = np.sum(h == 1.0)
    if ones_count > N // 2:
        # Remove excess ones
        excess = ones_count - N // 2
        ones_indices = np.where(h == 1.0)[0]
        to_remove = np.random.choice(ones_indices, excess, replace=False)
        h[to_remove] = 0.0
    elif ones_count < N // 2:
        # Add needed ones
        needed = N // 2 - ones_count
        zeros_indices = np.where(h == 0.0)[0]
        to_add = np.random.choice(zeros_indices, needed, replace=False)
        h[to_add] = 1.0
    
    return h

def construct():
    """
    Returns a sequence of step heights in [0, 1] whose sum is exactly len/2.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Try structured initialization first
    best_binary = _structured_init()
    best_score = _fast_overlap_score(best_binary)
    
    # Polish with deterministic search
    time_left = deadline - (time.time() - start_time)
    if time_left > 3.0:
        polished = _deterministic_polish(best_binary, time_limit=min(3.0, time_left * 0.5))
        score_polished = _fast_overlap_score(polished)
        if score_polished < best_score:
            best_binary = polished
            best_score = score_polished
    
    # Random restarts with local search
    while (time.time() - start_time) < deadline:
        binary2 = np.zeros(N)
        ones_indices = np.random.choice(N, N//2, replace=False)
        binary2[ones_indices] = 1.0
        
        # Allocate remaining time minus a small buffer for this restart
        time_left = deadline - (time.time() - start_time)
        if time_left < 2.0:
            break
            
        # Use a fraction of the remaining time for local search
        search_time = min(4.0, time_left * 0.6)
        cand = _local_search_binary(binary2, time_limit=search_time)
        
        # Polish the candidate
        time_left2 = deadline - (time.time() - start_time)
        if time_left2 > 1.5:
            cand = _deterministic_polish(cand, time_limit=min(1.5, time_left2 * 0.4))
        
        score_cand = _fast_overlap_score(cand)
        
        if score_cand < best_score:
            best_binary = cand
            best_score = score_cand
            
    return best_binary

# EVOLVE-BLOCK-END
