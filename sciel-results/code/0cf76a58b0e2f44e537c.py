import time
import random
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 115.0

def _compute_set_sizes_fft(u):
    """Compute |U+U| and |U-U| using FFT for exact counts."""
    if not u:
        return 0, 0
    m = max(u)
    # For sums: need range [0, 2m]
    # For diffs: need range [-m, m], shift by m to [0, 2m]
    size = 2 * m + 1
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    # Create indicator vector
    vec = np.zeros(fft_size, dtype=np.float64)
    for x in u:
        vec[x] = 1.0
    
    # Sumset: convolution of vec with itself
    fft_vec = np.fft.fft(vec)
    sum_fft = fft_vec * fft_vec
    sum_conv = np.fft.ifft(sum_fft).real
    # |U+U| = number of positions in [0, 2m] with count > 0
    n_sums = int(np.sum(sum_conv[:size] > 0.5))
    
    # Difference set: convolution of vec with reversed vec
    # rev_vec[i] = vec[fft_size - 1 - i] for circular, but we want linear
    # Actually: (U-U) corresponds to autocorrelation
    # diff[x] = sum_y vec[y] * vec[y+x] for x in [-m, m]
    # This is the autocorrelation
    diff_fft = fft_vec * np.conj(fft_vec)
    diff_conv = np.fft.ifft(diff_fft).real
    # diff_conv[k] for k=0..m gives counts for difference k
    # diff_conv[fft_size-k] gives counts for difference -k
    # |U-U| = number of k in [-m, m] with count > 0
    n_diffs = int(np.sum(diff_conv[:m+1] > 0.5))  # k=0 to m
    if m > 0:
        n_diffs += int(np.sum(diff_conv[fft_size-m:fft_size] > 0.5))  # k=-m to -1
    # Note: k=0 is counted once in diff_conv[0]
    
    return n_sums, n_diffs

def _score(u):
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    n_sums, n_diffs = _compute_set_sizes_fft(u)
    
    if n_sums == 0:
        return 0.0
        
    ratio = n_diffs / float(n_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _score_fast(u_list):
    """Score using precomputed or fast path for small sets."""
    if not u_list or len(u_list) < 2:
        return 0.0
    m = max(u_list)
    if m < 1:
        return 0.0
    # For very small sets, brute force is faster than FFT setup
    if m <= 500:
        sums = set()
        diffs = set()
        for a in u_list:
            for b in u_list:
                sums.add(a + b)
                diffs.add(a - b)
        n_sums = len(sums)
        n_diffs = len(diffs)
        if n_sums == 0:
            return 0.0
        ratio = n_diffs / float(n_sums)
        if ratio <= 0:
            return 0.0
        return 1.0 + math.log(ratio) / math.log(2 * m + 1)
    else:
        return _score(u_list)

def _construct_golomb(n):
    """
    Construct a Golomb ruler of length n using a simple greedy algorithm.
    """
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    u = [0]
    diffs = set()
    for i in range(1, n):
        best_pos = -1
        max_pos = (i+1)**2 + 10
        for pos in range(1, max_pos + 1):
            valid = True
            for a in u:
                d = abs(pos - a)
                if d in diffs:
                    valid = False
                    break
            if valid:
                best_pos = pos
                break
        
        if best_pos == -1:
            u.append(i)
        else:
            for a in u:
                diffs.add(abs(best_pos - a))
            u.append(best_pos)
            
    return sorted(u)

def _simulated_annealing(u, timeout=5.0):
    """
    Simulated annealing with lattice gas interpretation.
    Elements are particles with pairwise interactions determined by sumset/difference set structure.
    """
    u = sorted(set(u))
    if len(u) < 2:
        return u, _score(u)
    
    current_score = _score_fast(u)
    best_u = u[:]
    best_score = current_score
    start_time = time.time()
    n = len(u)
    max_val = max(u)
    
    # Temperature schedule
    T_start = 0.5
    T_end = 0.001
    total_steps = 2000
    
    for step in range(total_steps):
        if time.time() > start_time + timeout or time.time() > _DEADLINE - 3.0:
            break
        
        # Linear cooling
        frac = step / float(total_steps)
        T = T_start * (T_end / T_start) ** frac
        
        # Choose a random element to move
        i = random.randint(0, n - 1)
        old_val = u[i]
        
        # Generate new position: local move or jump
        if random.random() < 0.7:
            # Local perturbation
            step_size = max(1, int(0.1 * max_val))
            new_val = old_val + random.randint(-step_size, step_size)
        else:
            # Larger jump
            new_val = random.randint(0, int(max_val * 1.5))
        
        if new_val < 0:
            new_val = 0
        
        # Check collision
        if new_val != old_val:
            if new_val in u:
                continue
        
        new_u = u[:]
        new_u[i] = new_val
        new_u = sorted(set(new_u))
        if len(new_u) != n:
            continue
        
        new_score = _score_fast(new_u)
        delta = new_score - current_score
        
        # Metropolis acceptance
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-10)):
            u = new_u
            current_score = new_score
            max_val = max(u)
            if current_score > best_score:
                best_score = current_score
                best_u = u[:]
    
    return best_u, best_score

def _local_search(u, timeout=2.0):
    u = sorted(set(u))
    if len(u) < 2:
        return u, _score(u)
        
    current_score = _score_fast(u)
    best_u = u[:]
    best_score = current_score
    start_time = time.time()
    n = len(u)
    max_val = max(u)
    
    while time.time() < start_time + timeout and time.time() < _DEADLINE - 2.0:
        improved = False
        for i in range(n):
            if time.time() > start_time + timeout or time.time() > _DEADLINE - 2.0:
                break
            old_val = u[i]
            search_range = max(5, max_val // 3)
            candidates = set()
            for delta in range(-search_range, search_range + 1, max(1, search_range // 15)):
                new_val = old_val + delta
                if new_val >= 0:
                    candidates.add(new_val)
            for _ in range(3):
                candidates.add(random.randint(0, max_val + search_range))
                
            for new_val in candidates:
                if new_val in u and new_val != old_val:
                    continue
                new_u = u[:]
                new_u[i] = new_val
                new_u = sorted(set(new_u))
                if len(new_u) != n:
                    continue
                s = _score_fast(new_u)
                if s > current_score:
                    u = new_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    max_val = max(u)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            if random.random() < 0.15:
                idx = random.randint(0, n-1)
                u[idx] = random.randint(0, max_val + 10)
                u = sorted(set(u))
                if len(u) < n:
                    u = best_u[:]
                else:
                    current_score = _score_fast(u)

    return best_u, best_score

def construct():
    global _DEADLINE
    random.seed(42)
    
    best_u = [0, 1]
    best_s = _score(best_u)
    
    # Strategy: try various cardinalities, using both local search and simulated annealing
    # Larger n can give better ratios if the structure is right
    # Focus on n from 3 to ~20, with more time for promising ranges
    
    # Phase 1: Quick scan of small n
    for n in range(3, 12):
        if time.time() > _DEADLINE - 20.0:
            break
        
        for r in range(3):
            if time.time() > _DEADLINE - 15.0:
                break
            
            if r == 0:
                u_init = _construct_golomb(n)
            else:
                max_span = n * n * 2
                u_init = [0]
                for _ in range(n - 1):
                    u_init.append(random.randint(1, max_span))
                u_init = sorted(set(u_init))
                while len(u_init) < n:
                    u_init.append(random.randint(1, max_span))
                    u_init = sorted(set(u_init))
            
            u_opt, s_opt = _local_search(u_init, timeout=1.5)
            
            if s_opt > best_s:
                best_s = s_opt
                best_u = u_opt[:]
    
    # Phase 2: Simulated annealing on promising sizes
    # Use larger n where FFT is beneficial
    for n in [8, 10, 12, 15, 18, 20, 25]:
        if time.time() > _DEADLINE - 15.0:
            break
        
        time_budget = max(2.0, min(5.0, (_DEADLINE - time.time()) / 3.0 - 5.0))
        
        for r in range(4):
            if time.time() > _DEADLINE - 10.0:
                break
            
            if r == 0:
                # Structured init: arithmetic progression with gaps
                max_span = n * n
                u_init = [0]
                for i in range(1, n):
                    u_init.append(i * (max_span // n))
                u_init = sorted(set(u_init))
            elif r == 1:
                # Random with moderate span
                max_span = n * n * 3
                u_init = [0]
                for _ in range(n - 1):
                    u_init.append(random.randint(1, max_span))
                u_init = sorted(set(u_init))
                while len(u_init) < n:
                    u_init.append(random.randint(1, max_span))
                    u_init = sorted(set(u_init))
            else:
                # Start from best found so far, add/remove elements
                if len(best_u) >= 2:
                    u_init = best_u[:]
                    # Add some random elements
                    max_span = max(best_u) * 2 + 10
                    for _ in range(min(3, n - len(u_init))):
                        u_init.append(random.randint(1, max_span))
                    u_init = sorted(set(u_init))
                    if len(u_init) > n:
                        u_init = u_init[:n]
                    while len(u_init) < n:
                        u_init.append(random.randint(1, max_span))
                        u_init = sorted(set(u_init))
                else:
                    max_span = n * n * 2
                    u_init = [0]
                    for _ in range(n - 1):
                        u_init.append(random.randint(1, max_span))
                    u_init = sorted(set(u_init))
                    while len(u_init) < n:
                        u_init.append(random.randint(1, max_span))
                        u_init = sorted(set(u_init))
            
            u_opt, s_opt = _simulated_annealing(u_init, timeout=time_budget)
            
            if s_opt > best_s:
                best_s = s_opt
                best_u = u_opt[:]
    
    # Phase 3: Final polish on the best solution found
    if time.time() < _DEADLINE - 5.0:
        remaining = _DEADLINE - time.time() - 3.0
        if remaining > 1.0:
            u_opt, s_opt = _local_search(best_u, timeout=min(remaining, 5.0))
            if s_opt > best_s:
                best_s = s_opt
                best_u = u_opt[:]
    
    return best_u
