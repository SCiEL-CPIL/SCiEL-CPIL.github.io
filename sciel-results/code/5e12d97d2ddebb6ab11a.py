import math
import time
import random
import numpy as np

def _compute_score(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not U or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sum_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
    P = len(sum_set)
    
    # Compute difference set U-U
    diff_set = set()
    for a in U:
        for b in U:
            diff_set.add(a - b)
    D = len(diff_set)
    
    if P == 0:
        return 0.0
    
    denom = math.log(2 * max_u + 1, 2)
    if denom == 0:
        return 0.0
    
    score = 1.0 + math.log(D / P, 2) / denom
    return score

def _score_set_numpy(U):
    """
    Fast score calculation using numpy for small ranges.
    If max(U) is small, we can use bitsets.
    """
    if not U or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # If max_u is large, fall back to python sets
    if max_u > 5000:
        return _compute_score(U)
        
    # Use bitmasks for sumset and diffset
    # Sumset: convolution of indicator vector with itself
    # Diffset: cross-correlation? Or just shift and OR.
    
    # Indicator vector
    vec = np.zeros(2 * max_u + 1, dtype=bool)
    for u in U:
        vec[u] = True
        
    # Sumset
    # We can use FFT or just simple shifts if U is small
    # For now, use np.bitwise_or on shifts
    sum_mask = np.zeros(2 * max_u + 1, dtype=bool)
    diff_mask = np.zeros(2 * max_u + 1, dtype=bool)
    
    # This is O(|U|^2) in numpy which is still slow if |U| is large
    # But bitwise ops are fast.
    
    # Actually, simple python sets are often faster for small |U| < 20 due to overhead
    # Let's stick to python sets for robustness and speed in the general case, 
    # but optimize the inner loop.
    
    # Optimized Python Set Calculation
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    P = len(sum_set)
    D = len(diff_set)
    
    if P == 0:
        return 0.0
        
    denom = math.log(2 * max_u + 1, 2)
    if denom == 0:
        return 0.0
        
    score = 1.0 + math.log(D / P, 2) / denom
    return score

def _simulated_annealing(start_set, time_limit, temp_start=1.0, temp_end=0.01):
    """
    Simulated Annealing with diverse moves.
    """
    current_set = sorted(start_set)
    current_score = _compute_score(current_set)
    best_set = current_set[:]
    best_score = current_score
    
    T = temp_start
    total_time = time_limit
    start_time = time.time()
    
    # Determine move range based on current max
    max_u = max(current_set) if current_set else 10
    move_range = max(20, max_u // 2)
    
    while time.time() - start_time < total_time:
        # Generate a move
        move_type = random.choice(['add', 'remove', 'swap', 'shift'])
        new_set = current_set[:]
        
        if move_type == 'add':
            # Add a random element
            # Bias towards elements close to existing ones or far away
            if random.random() < 0.5:
                # Close to existing
                base = random.choice(current_set)
                new_elem = base + random.randint(1, move_range)
            else:
                # Random
                new_elem = random.randint(1, max_u + move_range)
            
            if new_elem not in new_set:
                new_set.append(new_elem)
                new_set.sort()
                
        elif move_type == 'remove':
            if len(new_set) > 1:
                # Remove a random element (not 0)
                idx = random.randint(1, len(new_set) - 1)
                new_set.pop(idx)
                
        elif move_type == 'swap':
            if len(new_set) > 1:
                # Remove one, add another
                idx = random.randint(1, len(new_set) - 1)
                removed = new_set.pop(idx)
                new_elem = removed + random.randint(-move_range, move_range)
                if new_elem < 1:
                    new_elem = random.randint(1, max_u + move_range)
                if new_elem not in new_set:
                    new_set.append(new_elem)
                    new_set.sort()
                else:
                    new_set.append(removed)
                    new_set.sort()
                    
        elif move_type == 'shift':
            # Shift a random element
            if len(new_set) > 1:
                idx = random.randint(1, len(new_set) - 1)
                old = new_set[idx]
                new_val = old + random.randint(-move_range, move_range)
                if new_val >= 1 and new_val not in new_set:
                    new_set[idx] = new_val
                    new_set.sort()
                else:
                    # Revert if invalid
                    pass
                    
        # Evaluate
        if not new_set or 0 not in new_set:
            new_score = 0.0
        else:
            new_score = _compute_score(new_set)
            
        # Acceptance
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = new_set[:]
        else:
            delta = current_score - new_score
            if delta == 0:
                accept = True
            else:
                # Avoid division by zero or underflow
                if T < 1e-9:
                    accept = False
                else:
                    accept = random.random() < math.exp(-delta / T)
            
            if accept:
                current_set = new_set
                current_score = new_score
                
        # Cool down
        T *= 0.999
        if T < temp_end:
            T = temp_end
            
    return best_set, best_score

def _local_search_polish(U, time_limit):
    """
    High-precision polish: try all single and double moves.
    """
    current_set = sorted(U)
    current_score = _compute_score(current_set)
    best_set = current_set[:]
    best_score = current_score
    
    start_time = time.time()
    
    # Single move search
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        max_u = max(current_set)
        
        # Try adding
        for c in range(1, max_u + 50):
            if c in current_set:
                continue
            new_set = current_set + [c]
            new_set.sort()
            s = _compute_score(new_set)
            if s > current_score:
                current_set = new_set
                current_score = s
                best_set = current_set[:]
                best_score = s
                changed = True
                break # Restart search with new set
        
        if changed:
            continue
            
        # Try removing
        for i in range(1, len(current_set)):
            new_set = current_set[:i] + current_set[i+1:]
            s = _compute_score(new_set)
            if s > current_score:
                current_set = new_set
                current_score = s
                best_set = current_set[:]
                best_score = s
                changed = True
                break
        
        if changed:
            continue
            
        # Try swapping (replace i with j)
        for i in range(1, len(current_set)):
            old = current_set[i]
            for j in range(1, max_u + 50):
                if j == old or j in current_set:
                    continue
                new_set = current_set[:]
                new_set[i] = j
                new_set.sort()
                s = _compute_score(new_set)
                if s > current_score:
                    current_set = new_set
                    current_score = s
                    best_set = current_set[:]
                    best_score = s
                    changed = True
                    break
            if changed:
                break
                
    return best_set, best_score

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 7],
        [0, 1, 4, 9],
        [0, 2, 3, 5],
        [0, 1, 5, 9],
        [0, 3, 5, 7],
        [0, 1, 2, 5, 10],
        [0, 1, 4, 6, 10],
        [0, 1, 3, 6, 10],
    ]
    
    best_set = None
    best_score = 0.0
    
    # Evaluate initial candidates
    for c in candidates:
        s = _compute_score(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    # Phase 1: Simulated Annealing
    if time.time() < deadline:
        # Run SA for a portion of the time
        sa_time = 60
        sa_set, sa_score = _simulated_annealing(best_set, sa_time)
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_set[:]
            
    # Phase 2: Local Search Polish
    if time.time() < deadline:
        polish_time = deadline - time.time() - 5
        if polish_time > 0:
            pol_set, pol_score = _local_search_polish(best_set, polish_time)
            if pol_score > best_score:
                best_score = pol_score
                best_set = pol_set[:]
                
    return sorted(set(best_set))
