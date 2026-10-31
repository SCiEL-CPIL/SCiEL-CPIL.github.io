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

def _compute_score_incremental(u, base_fft_ind, base_fft_size, added_elem=None, removed_elem=None, moved_from=None, moved_to=None):
    """Compute score incrementally by updating FFT."""
    u = sorted(set(u))
    if not u or u[0] != 0:
        u = [0] + list(u)
        u = sorted(set(u))
    
    max_u = max(u)
    m = 2 * max_u + 1
    
    if m <= 1:
        return 1.0
    
    # Determine required FFT size
    needed_fft_size = 1
    while needed_fft_size < 2 * (2 * max_u + 1):
        needed_fft_size *= 2
    
    if needed_fft_size != base_fft_size:
        # Need full recompute
        return _compute_score(u)
    
    # Update indicator
    indicator = np.zeros(2 * max_u + 1, dtype=np.float64)
    for x in u:
        indicator[x] = 1.0
    
    fft_size = base_fft_size
    fft_ind = np.fft.fft(indicator, n=fft_size)
    fft_sum = fft_ind * fft_ind
    sumset_counts = np.fft.ifft(fft_sum).real
    sumset_size = int(np.sum(sumset_counts[:2 * max_u + 1] > 0.5))
    
    fft_corr = fft_ind * np.conj(fft_ind)
    diffset_counts = np.fft.ifft(fft_corr).real
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
    
    max_steps = 200
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        max_u = max(u)
        # Adaptive upper bound: allow larger sets
        upper_bound = min(4 * max_u + 20, 5000)
        
        candidates = [x for x in range(1, upper_bound + 1) if x not in u]
        
        if not candidates:
            break
            
        best_candidate = None
        best_candidate_score = current_score
        
        # Sample candidates if too many
        if len(candidates) > 200:
            import random
            random.seed(step)
            candidates = random.sample(candidates, 200)
        
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
    
    max_steps = 3000
    step = 0
    no_improve_count = 0
    
    while step < max_steps:
        if time.time() > deadline:
            break
            
        improved = False
        mutation_type = step % 3
        
        if mutation_type == 0:
            # Try moving an element
            for i in range(1, len(u)):
                old_val = u[i]
                for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
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
                        no_improve_count = 0
                        break
                if improved:
                    break
        elif mutation_type == 1:
            # Try adding an element
            max_u = max(u)
            candidates = set()
            for d in range(1, 8):
                candidates.add(max_u + d)
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
                    no_improve_count = 0
                    break
        else:
            # Try removing an element (to reduce max)
            if len(u) > 3:
                for i in range(len(u) - 1, 0, -1):
                    if u[i] == 0:
                        continue
                    new_u = [x for j, x in enumerate(u) if j != i]
                    if 0 not in new_u:
                        continue
                    s = _compute_score(new_u)
                    if s > current_score:
                        u = new_u
                        current_score = s
                        if current_score > best_score:
                            best_score = current_score
                            best_u = list(u)
                        improved = True
                        no_improve_count = 0
                        break
            
        if not improved:
            no_improve_count += 1
            if no_improve_count > 50:
                break
                
        step += 1
        
    return best_u, best_score

def _golomb_ruler_init(n):
    """Generate a Golomb ruler-like initial set."""
    # Use known Golomb rulers or construct greedily
    # Greedy Golomb ruler construction
    u = [0]
    used_diffs = set()
    
    for _ in range(n - 1):
        best_pos = None
        best_min_diff = float('inf')
        
        for pos in range(1, 1000):
            if pos in u:
                continue
            # Check if this creates new differences
            valid = True
            min_new_diff = float('inf')
            for existing in u:
                d = abs(pos - existing)
                if d in used_diffs:
                    valid = False
                    break
                min_new_diff = min(min_new_diff, d)
            
            if valid and min_new_diff < best_min_diff:
                best_min_diff = min_new_diff
                best_pos = pos
        
        if best_pos is not None:
            u.append(best_pos)
            for existing in u[:-1]:
                used_diffs.add(abs(best_pos - existing))
        else:
            break
    
    return sorted(u)

def _construct_initial_candidates():
    """Generate diverse initial candidates."""
    candidates = []
    
    # Small Golomb rulers
    for n in [3, 4, 5, 6, 7, 8, 9, 10]:
        gr = _golomb_ruler_init(n)
        if gr and gr[0] == 0:
            candidates.append(gr)
    
    # Known good structures
    candidates.append([0, 1, 3])
    candidates.append([0, 1, 4, 6])
    candidates.append([0, 1, 4, 9, 11])
    candidates.append([0, 1, 4, 10, 12, 17])
    candidates.append([0, 1, 4, 10, 12, 17, 25])
    candidates.append([0, 1, 4, 10, 12, 17, 25, 34])
    
    # Random sets with Golomb-like properties
    import random
    random.seed(123)
    for trial in range(5):
        n = random.randint(5, 15)
        u = {0}
        used_diffs = set()
        for _ in range(n - 1):
            attempts = 0
            while attempts < 100:
                pos = random.randint(1, 200)
                if pos in u:
                    attempts += 1
                    continue
                valid = True
                for existing in u:
                    d = abs(pos - existing)
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    u.add(pos)
                    for existing in list(u)[:-1]:
                        used_diffs.add(abs(pos - existing))
                    break
                attempts += 1
        candidates.append(sorted(u))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    initial_candidates = _construct_initial_candidates()
    
    best_set = [0, 1, 3]
    best_score = 0.0
    
    for u in initial_candidates:
        if time.time() > deadline:
            break
            
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
            u = sorted(set(u))
            
        # Quick score check
        s = _compute_score(u)
        if s > best_score:
            best_score = s
            best_set = list(u)
        
        # Greedy growth
        grown_set, grown_score = _greedy_grow(u, deadline)
        
        # Local search
        final_set, final_score = _local_search(grown_set, deadline)
        
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
            
    # Final polish: try more aggressive local search on best
    if time.time() < deadline - 5:
        polished, polished_score = _local_search(best_set, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_set = polished
    
    return best_set
