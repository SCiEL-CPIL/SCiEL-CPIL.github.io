"""Program for maximizing 3-AP differences in a 32-element set within 1..512."""

import random
import time
import sys

def count_aps(s):
    """
    Calculate the number of distinct differences d > 0 such that
    a, a+d, a+2d are all in the set s.
    """
    s_set = set(s)
    diffs = set()
    for i in range(len(s)):
        a = s[i]
        for j in range(len(s)):
            if i == j: continue
            b = s[j]
            if b <= a: continue
            d = b - a
            c = a + 2 * d
            if c in s_set:
                diffs.add(d)
    return len(diffs)

def _generate_structured_candidates(N=32, MAX_VAL=512):
    """Generate diverse structured initial candidates."""
    candidates = []
    
    # 1. Arithmetic progression
    step = MAX_VAL // (N - 1)
    ap = [1 + i * step for i in range(N)]
    candidates.append(ap)
    
    # 2. Quadratic residue-like pattern
    quad = sorted(set((i * i) % MAX_VAL + 1 for i in range(1, MAX_VAL + 1)))
    if len(quad) >= N:
        candidates.append(random.sample(quad, N))
    
    # 3. Mixed: half in low range, half in high range
    low = random.sample(range(1, MAX_VAL // 2 + 1), N // 2)
    high = random.sample(range(MAX_VAL // 2 + 1, MAX_VAL + 1), N - N // 2)
    candidates.append(sorted(low + high))
    
    # 4. Clustered: groups of 4 elements
    clusters = []
    for c in range(N // 4):
        base = random.randint(1, MAX_VAL - 15)
        cluster = [base + random.randint(0, 15) for _ in range(4)]
        clusters.extend(cluster)
    if len(clusters) == N:
        candidates.append(sorted(set(clusters)))
    
    # 5. Random
    candidates.append(random.sample(range(1, MAX_VAL + 1), N))
    
    # 6. Geometric-like spacing
    geom = [1]
    for i in range(1, N):
        next_val = geom[-1] + random.randint(5, 20)
        if next_val > MAX_VAL:
            break
        geom.append(next_val)
    if len(geom) >= N:
        candidates.append(geom[:N])
    
    return candidates

def _local_search_optimize(s, N=32, MAX_VAL=512, time_limit=10.0):
    """Apply local search to optimize a set."""
    start_time = time.time()
    current = set(s)
    current_score = count_aps(list(current))
    best = list(current)
    best_score = current_score
    
    # Tabu list to avoid cycling
    tabu = set()
    tabu_len = 50
    
    while time.time() - start_time < time_limit:
        # Try all possible single-element swaps
        best_swap = None
        best_swap_score = current_score
        
        elements = list(current)
        for i in range(len(elements)):
            old_val = elements[i]
            # Try moving to a few candidate positions
            candidates = [random.randint(1, MAX_VAL) for _ in range(10)]
            # Also try values near existing elements
            for elem in random.sample(elements, min(5, len(elements))):
                candidates.append(elem + random.randint(-10, 10))
                candidates.append(elem + random.randint(20, 50))
            
            for new_val in candidates:
                if new_val < 1 or new_val > MAX_VAL:
                    continue
                if new_val == old_val:
                    continue
                if new_val in current:
                    continue
                
                # Check if this swap is tabu
                swap_key = (old_val, new_val)
                if swap_key in tabu:
                    continue
                
                # Test the swap
                current.remove(old_val)
                current.add(new_val)
                score = count_aps(list(current))
                
                if score > best_swap_score:
                    best_swap_score = score
                    best_swap = (old_val, new_val)
                
                # Revert
                current.remove(new_val)
                current.add(old_val)
        
        if best_swap:
            old_val, new_val = best_swap
            current.remove(old_val)
            current.add(new_val)
            current_score = best_swap_score
            
            if current_score > best_score:
                best_score = current_score
                best = list(current)
            
            # Update tabu
            tabu.add(best_swap)
            if len(tabu) > tabu_len:
                # Remove oldest (approximate by clearing half)
                tabu = set(list(tabu)[:tabu_len // 2])
        else:
            # No improving swap found, try random perturbation
            idx = random.randint(0, N - 1)
            old_val = elements[idx]
            new_val = random.randint(1, MAX_VAL)
            if new_val in current and new_val != old_val:
                continue
            
            current.remove(old_val)
            current.add(new_val)
            current_score = count_aps(list(current))
            
            if current_score > best_score:
                best_score = current_score
                best = list(current)
    
    return best, best_score

def construct():
    """
    Use structured initialization + local search to find a 32-element set
    in [1, 512] that maximizes the number of distinct 3-AP differences.
    """
    N = 32
    MAX_VAL = 512
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Generate diverse initial candidates
    candidates = _generate_structured_candidates(N, MAX_VAL)
    
    best_set = None
    best_score = -1
    
    # Evaluate each candidate and optimize the best ones
    for candidate in candidates:
        if time.time() - start_time > time_limit - 15:
            break
        
        # Quick evaluation
        score = count_aps(candidate)
        
        # If promising, do local search
        if score > best_score or best_set is None:
            optimized, opt_score = _local_search_optimize(
                candidate, N, MAX_VAL, 
                time_limit=min(8.0, time_limit - (time.time() - start_time))
            )
            
            if opt_score > best_score:
                best_score = opt_score
                best_set = optimized
    
    # If we still have time, do additional random restarts
    if best_set is None:
        best_set = random.sample(range(1, MAX_VAL + 1), N)
        best_score = count_aps(best_set)
    
    # Final refinement if time permits
    if time.time() - start_time < time_limit - 5:
        refined, ref_score = _local_search_optimize(
            best_set, N, MAX_VAL,
            time_limit=time_limit - (time.time() - start_time) - 2
        )
        if ref_score > best_score:
            best_score = ref_score
            best_set = refined
    
    return best_set
