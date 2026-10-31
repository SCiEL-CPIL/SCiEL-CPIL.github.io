import math
import time
import numpy as np

def _compute_score(u):
    """Compute score using FFT for O(M log M) evaluation."""
    u = sorted(set(u))
    if not u or u[0] != 0:
        u = [0] + list(u)
        u = sorted(set(u))
    
    max_u = max(u)
    m = 2 * max_u + 1
    
    if m <= 1:
        return 1.0
    
    # Create indicator vector
    indicator = np.zeros(2 * max_u + 1, dtype=np.float64)
    for x in u:
        indicator[x] = 1.0
    
    # Sumset: convolution
    fft_size = 1
    while fft_size < 2 * (2 * max_u + 1):
        fft_size *= 2
        
    fft_ind = np.fft.fft(indicator, n=fft_size)
    fft_sum = fft_ind * fft_ind
    sumset_counts = np.fft.ifft(fft_sum).real
    
    # Sumset size
    sumset_size = int(np.sum(sumset_counts[:2 * max_u + 1] > 0.5))
    
    # Diffset: correlation
    fft_corr = fft_ind * np.conj(fft_ind)
    diffset_counts = np.fft.ifft(fft_corr).real
    
    # Diffset size
    diffset_size = 0
    for lag in range(-max_u, max_u + 1):
        pos = lag % fft_size
        if diffset_counts[pos] > 0.5:
            diffset_size += 1
    
    if sumset_size == 0:
        return 1.0
        
    r = diffset_size / sumset_size
    score = 1 + math.log(r) / math.log(m)
    return score

def _greedy_grow(u, deadline):
    """Greedy growth of the set."""
    u = sorted(set(u))
    if 0 not in u:
        u = [0] + u
        u = sorted(set(u))
    
    current_score = _compute_score(u)
    best_score = current_score
    best_u = list(u)
    
    # Limit the number of growth steps to avoid infinite loops or excessive time
    max_steps = 100
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        max_u = max(u)
        # Candidate range: from 1 to 2 * max_u + 10
        # To keep it fast, we might limit this if max_u is very large
        upper_bound = 2 * max_u + 10
        if upper_bound > 1000:
            upper_bound = 1000 # Cap for safety
            
        candidates = [x for x in range(1, upper_bound + 1) if x not in u]
        
        if not candidates:
            break
            
        best_candidate = None
        best_candidate_score = current_score
        
        # Evaluate each candidate
        for c in candidates:
            new_u = sorted(u + [c])
            s = _compute_score(new_u)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = new_u
                
        if best_candidate is not None:
            u = best_candidate
            current_score = best_candidate_score
            if current_score > best_score:
                best_score = current_score
                best_u = list(u)
        else:
            break
            
    return best_u, best_score

def _local_search(u, deadline):
    """Local search to improve the set."""
    u = sorted(set(u))
    if 0 not in u:
        u = [0] + u
        u = sorted(set(u))
        
    current_score = _compute_score(u)
    best_score = current_score
    best_u = list(u)
    
    max_steps = 5000
    step = 0
    while step < max_steps:
        if time.time() > deadline:
            break
            
        improved = False
        mutation_type = step % 2
        
        if mutation_type == 0:
            # Try moving an element
            for i in range(1, len(u)):
                old_val = u[i]
                for delta in [-3, -2, -1, 1, 2, 3]:
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    if any(u[j] == new_val for j in range(len(u)) if j != i):
                        continue
                    new_u = list(u)
                    new_u[i] = new_val
                    new_u = sorted(set(new_u))
                    if 0 not in new_u:
                        new_u = [0] + new_u
                        new_u = sorted(set(new_u))
                    s = _compute_score(new_u)
                    if s > current_score:
                        u = new_u
                        current_score = s
                        if current_score > best_score:
                            best_score = current_score
                            best_u = list(u)
                        improved = True
                        break
                if improved:
                    break
        else:
            # Try adding an element
            max_u = max(u)
            candidates = set()
            candidates.add(max_u + 1)
            candidates.add(max_u + 2)
            candidates.add(max_u + 3)
            # Add in gaps
            for i in range(len(u) - 1):
                gap_start = u[i] + 1
                gap_end = u[i + 1] - 1
                if gap_end >= gap_start:
                    candidates.add((gap_start + gap_end) // 2)
                    candidates.add(gap_start)
                    candidates.add(gap_end)
                    
            for c in sorted(candidates):
                if c in u or c <= 0:
                    continue
                new_u = sorted(u + [c])
                s = _compute_score(new_u)
                if s > current_score:
                    u = new_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = list(u)
                    improved = True
                    break
            
        if not improved:
            # If no improvement in a full cycle, try a random jump or break
            # For now, just break to avoid wasting time
            if step > 100:
                break
                
        step += 1
        
    return best_u, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial candidates
    initial_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 44],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55],
    ]
    
    # Add some random starts
    import random
    random.seed(42)
    for n in [10, 20, 50]:
        random_set = {0}
        for _ in range(n):
            random_set.add(random.randint(1, n * 5))
        initial_candidates.append(sorted(random_set))
        
    best_set = [0, 1, 3]
    best_score = 0.0
    
    for u in initial_candidates:
        if time.time() > deadline:
            break
            
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
            u = sorted(set(u))
            
        # Greedy growth
        grown_set, grown_score = _greedy_grow(u, deadline)
        
        # Local search
        final_set, final_score = _local_search(grown_set, deadline)
        
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
            
    return best_set
