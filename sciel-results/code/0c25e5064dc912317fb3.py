"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 12, 'm': 40}

def _is_strict_subset(a, b):
    """Check if a is a strict subset of b. Assumes a and b are frozensets."""
    if len(a) >= len(b):
        return False
    return a < b

def _initial_family(n, m):
    """Fast initialization: create a few chains and fill rest randomly."""
    family = []
    used = set()
    
    # Start with empty set
    family.append([])
    used.add(frozenset())
    
    # Build one long chain
    all_elements = list(range(n))
    random.shuffle(all_elements)
    
    for i in range(1, n + 1):
        s = frozenset(all_elements[:i])
        if s not in used:
            family.append(list(s))
            used.add(s)
            
    # Fill rest with random sets
    while len(family) < m:
        size = random.randint(0, n)
        s = frozenset(random.sample(all_elements, size))
        if s not in used:
            family.append(list(s))
            used.add(s)
            
    return family[:m]

def _calc_initial_score(family):
    """Calculate score for initial family."""
    m = len(family)
    fs = [frozenset(s) for s in family]
    count = 0
    for i in range(m):
        for j in range(i+1, m):
            if _is_strict_subset(fs[i], fs[j]):
                count += 1
            elif _is_strict_subset(fs[j], fs[i]):
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = P['n']
    m = P['m']
    
    # Initial family
    family = _initial_family(n, m)
    current = [list(s) for s in family]
    current_fs = [frozenset(s) for s in current]
    current_score = _calc_initial_score(current)
    
    best_family = [list(s) for s in current]
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        idx = random.randint(0, m - 1)
        old_set = current[idx]
        old_fs = current_fs[idx]
        old_size = len(old_set)
        
        new_set = None
        strategy = random.random()
        
        if strategy < 0.4:
            # Add element
            if old_size < n:
                candidates = [b for b in range(n) if b not in old_set]
                if candidates:
                    new_elem = random.choice(candidates)
                    new_set = list(old_set) + [new_elem]
        elif strategy < 0.8:
            # Remove element
            if old_size > 0:
                new_set = list(old_set)
                new_set.pop(random.randint(0, len(new_set) - 1))
        else:
            # Swap element
            if old_size > 0:
                candidates_out = [b for b in range(n) if b not in old_set]
                if candidates_out:
                    new_set = list(old_set)
                    pos = random.randint(0, len(new_set) - 1)
                    new_elem = random.choice(candidates_out)
                    new_set[pos] = new_elem
        
        if new_set is None:
            continue
            
        new_fs = frozenset(new_set)
        
        # Check for duplicates
        is_dup = False
        for i, fs in enumerate(current_fs):
            if fs == new_fs:
                if i != idx:
                    is_dup = True
                    break
        if is_dup:
            continue
            
        # Calculate delta score
        delta = 0
        for j in range(m):
            if j == idx:
                continue
            if _is_strict_subset(old_fs, current_fs[j]):
                delta -= 1
            elif _is_strict_subset(current_fs[j], old_fs):
                delta -= 1
            
            if _is_strict_subset(new_fs, current_fs[j]):
                delta += 1
            elif _is_strict_subset(current_fs[j], new_fs):
                delta += 1
                
        new_score = current_score + delta
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current[idx] = new_set
            current_fs[idx] = new_fs
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_family = [list(s) for s in current]
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_family
# EVOLVE-BLOCK-END
