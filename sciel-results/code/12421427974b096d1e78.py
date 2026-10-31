import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_set(u):
    """
    Compute the score for a set of integers u.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    n = len(u_arr)
    
    # Compute all pairwise sums and differences
    sums = u_arr[:, None] + u_arr[None, :]
    diffs = u_arr[:, None] - u_arr[None, :]
    
    unique_sums = np.unique(sums)
    unique_diffs = np.unique(diffs)
    
    if len(unique_sums) == 0:
        return 0.0
        
    ratio = len(unique_diffs) / float(len(unique_sums))
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _fft_score(u):
    """
    Fast scoring using FFT to count unique sums and differences.
    This is O(M log M) where M is the span, allowing evaluation of larger sets.
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    
    # Size of FFT needs to be at least 2*max(U) + 1 for sums if we don't shift?
    # Actually, sum of two numbers in [0, M] is in [0, 2M].
    # Diff of two numbers in [0, M] is in [-M, M].
    
    size = 2 * m + 1
    n_fft = 1
    while n_fft < size:
        n_fft *= 2
        
    # Indicator vector
    f = np.zeros(n_fft, dtype=np.float64)
    f[u_arr] = 1.0
    
    # FFT
    F = np.fft.fft(f)
    
    # Sums: Convolution of f with f.
    conv = np.fft.ifft(F * F).real
    unique_sums = np.sum(conv[:size] > 0.5)
    
    # Diffs: Autocorrelation.
    autocorr = np.fft.ifft(F * np.conj(F)).real
    
    mask = np.zeros(n_fft, dtype=bool)
    mask[0] = True
    mask[1:m+1] = True
    mask[n_fft-m:n_fft] = True
    
    unique_diffs = np.sum(autocorr[mask] > 0.5)
    
    ratio = unique_diffs / float(unique_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _sidon_initial_set(n, max_val):
    """
    Generate a Sidon-like set (all pairwise sums unique) as initial candidate.
    Uses greedy construction.
    """
    if n <= 1:
        return [0]
    
    u = [0]
    used_sums = {0}
    
    for i in range(1, n):
        # Try to find a value that doesn't create sum collisions
        best_val = None
        for val in range(1, max_val + 1):
            if val in u:
                continue
            # Check if adding val creates any sum collision
            valid = True
            for existing in u:
                s = existing + val
                if s in used_sums:
                    valid = False
                    break
            if valid:
                best_val = val
                break
        
        if best_val is None:
            # If no perfect Sidon value found, pick a random one
            candidates = [v for v in range(1, max_val + 1) if v not in u]
            if candidates:
                best_val = random.choice(candidates)
            else:
                best_val = max(u) + 1
        
        u.append(best_val)
        for existing in u[:-1]:
            used_sums.add(existing + best_val)
    
    return sorted(u)

def _structured_initial_set(n, max_val):
    """
    Generate structured sets that might have good sum/diff ratios.
    Mixes different strategies.
    """
    strategy = random.random()
    
    if strategy < 0.3:
        # Sidon-like
        return _sidon_initial_set(n, max_val)
    elif strategy < 0.5:
        # Arithmetic progression with perturbations
        step = max(1, max_val // (n + 1))
        u = set([0])
        for i in range(1, n):
            base = i * step
            perturb = random.randint(-step // 4, step // 4)
            val = max(1, base + perturb)
            u.add(val)
        return sorted(u)
    elif strategy < 0.7:
        # Geometric-like with noise
        ratio = (max_val) ** (1.0 / max(1, n - 1))
        u = set([0])
        for i in range(1, n):
            val = int(ratio ** i) + random.randint(-max(1, int(ratio ** i * 0.1)), max(1, int(ratio ** i * 0.1)))
            val = max(1, min(val, max_val))
            u.add(val)
        return sorted(u)
    else:
        # Random
        u = {0}
        for _ in range(n - 1):
            val = random.randint(1, max_val)
            u.add(val)
        return sorted(u)

def _local_search(u, timeout=5.0, use_fft=True):
    """
    Improve a set using local search.
    Moves: Replace an element with a nearby one that improves score.
    Uses FFT for faster evaluation if enabled.
    """
    u = sorted(set(u))
    if len(u) < 2:
        return [0, 1], 0.0
    
    score_func = _fft_score if use_fft else _score_from_set
    current_score = score_func(u)
    best_u = u[:]
    best_score = current_score
    start_time = time.time()
    
    n = len(u)
    max_val = max(u)
    
    for step in range(20000):
        if time.time() > start_time + timeout or time.time() > _DEADLINE - 5.0:
            break
            
        if n < 2:
            break
        
        # Pick a random element to perturb (excluding 0)
        idx = random.randint(1, n - 1)
        old_val = u[idx]
        
        # Determine search range - adaptive based on current state
        # If ratio is close to optimal, use smaller steps for fine-tuning
        # If ratio is far from optimal, use larger steps for restructuring
        ratio_estimate = current_score - 1.0
        if ratio_estimate > 0.01:
            # Good ratio, fine-tune
            search_range = max(1, int(max_val * 0.02) + 5)
        else:
            # Poor ratio, restructure
            search_range = max(1, int(max_val * 0.1) + 20)
        
        # Generate diverse candidates
        candidates = []
        
        # Small steps (fine tuning)
        step_size = max(1, search_range // 8)
        for delta in range(-search_range, search_range + 1, step_size):
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val < 0 or new_val in u:
                continue
            candidates.append(new_val)
            
        # Medium jumps
        for _ in range(5):
            jump = random.randint(search_range, search_range * 3)
            direction = random.choice([-1, 1])
            new_val = old_val + direction * jump
            if 0 <= new_val <= max_val * 2 and new_val not in u:
                candidates.append(new_val)
                
        # Large jumps (random restarts within local neighborhood)
        for _ in range(3):
            new_val = random.randint(1, max_val * 2)
            if new_val not in u:
                candidates.append(new_val)
                
        if not candidates:
            continue
            
        random.shuffle(candidates)
        
        improved = False
        # Limit tries per step to keep pace
        for new_val in candidates[:15]: 
            new_u = u[:]
            new_u[idx] = new_val
            new_u = sorted(set(new_u))
            
            if len(new_u) != n:
                continue
                
            s = score_func(new_u)
            if s > current_score:
                u = new_u
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_u = u[:]
                improved = True
                # Update max_val if it changed significantly
                new_max = max(u)
                if new_max > max_val:
                    max_val = new_max
                break
                
        if not improved:
            # Try adding an element if size is small and span allows
            if n < 40 and random.random() < 0.08:
                # Add a random element in a gap
                if n > 1:
                    idx_add = random.randint(0, n-2)
                    low = u[idx_add]
                    high = u[idx_add+1]
                    if high - low > 1:
                        new_val = random.randint(low+1, high-1)
                        if new_val not in u:
                            test_u = sorted(u + [new_val])
                            s = score_func(test_u)
                            if s > current_score:
                                u = test_u
                                current_score = s
                                if current_score > best_score:
                                    best_score = current_score
                                    best_u = u[:]
                                n += 1
            # Try removing an element if size is large
            elif n > 10 and random.random() < 0.06:
                idx_rm = random.randint(1, n - 1)
                test_u = u[:]
                test_u.pop(idx_rm)
                s = score_func(test_u)
                if s > current_score:
                    u = test_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    n -= 1
                    max_val = max(u) if u else 1
                    
    return best_u, best_score

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_u = [0, 1]
    best_s = 0.0
    
    # Strategy:
    # 1. Use FFT-based scoring to allow exploring larger sets (n up to ~50-100) 
    #    and larger spans without timing out.
    # 2. Initialize with diverse structured sets (Sidon, arithmetic, geometric, random).
    # 3. Perform local search to optimize the ratio.
    # 4. The key insight is that "Golomb" (max diff) is not always optimal.
    #    Sometimes allowing some diff collisions while reducing span or sum collisions yields a better ratio.
    
    # Try a range of sizes
    sizes_to_try = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    
    for n in sizes_to_try:
        if time.time() > _DEADLINE - 15.0:
            break
            
        # Number of restarts for this size
        # Larger n -> fewer restarts
        restarts = max(1, 25 - n // 4)
        
        for r in range(restarts):
            if time.time() > _DEADLINE - 10.0:
                break
                
            # Initialize with diverse strategies
            # Span heuristic: for Sidon-like sets, span ~ n^2. 
            # But we want to explore beyond that.
            min_span = int(n ** 1.5)
            max_span = int(n ** 2.5)
            
            init_max_val = random.randint(min_span, max_span)
            
            # Use structured initialization
            u_init = _structured_initial_set(n, init_max_val)
            
            # Optimize
            u_opt, s_opt = _local_search(u_init, timeout=4.0, use_fft=True)
            
            if s_opt > best_s:
                best_s = s_opt
                best_u = u_opt[:]
                
    # Final polish with the best found
    if time.time() < _DEADLINE - 5.0:
        # Use exact scoring for final verification and potentially fine-tuning
        if len(best_u) < 50:
            u_final, s_final = _local_search(best_u, timeout=10.0, use_fft=False)
            if s_final > best_s:
                best_s = s_final
                best_u = u_final[:]
        else:
            # For larger sets, stick with FFT optimized version
            # But verify score with exact method to be safe?
            s_exact = _score_from_set(best_u)
            if s_exact > best_s:
                best_s = s_exact
            if time.time() < _DEADLINE - 2.0:
                 u_final, s_final = _local_search(best_u, timeout=5.0, use_fft=False)
                 if s_final > best_s:
                     best_s = s_final
                     best_u = u_final[:]

    # Sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u
# EVOLVE-BLOCK-END
