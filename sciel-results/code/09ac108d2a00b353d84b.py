# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _is_sidon_fast(s):
    """Check if s is a Sidon set. Returns True if all positive differences are distinct."""
    diffs = set()
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n):
            d = s[j] - s[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _covers_1_to_30_fast(s):
    """Check if all d in 1..30 are present as differences."""
    diffs = set()
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(s[j] - s[i])
    return all(x in diffs for x in range(1, 31))

def _valid(s):
    if len(s) < 1 or len(s) > 32:
        return False
    if 0 not in s:
        return False
    if max(s) > 10**6:
        return False
    if not _is_sidon_fast(s):
        return False
    if not _covers_1_to_30_fast(s):
        return False
    return True

def _greedy_construct():
    """
    Greedy construction trying to keep elements small.
    """
    a = [0]
    
    def get_diffs(s):
        return {s[j] - s[i] for i in range(len(s)) for j in range(i + 1, len(s))}
        
    for d in range(1, 31):
        if d in get_diffs(a):
            continue
        found = False
        # Try to add a single element x such that x - v = d or v - x = d
        # We want to minimize the new max element.
        candidates = []
        for v in a:
            # x = v + d
            x1 = v + d
            # x = v - d
            x2 = v - d
            if x1 >= 0 and x1 not in a:
                candidates.append(x1)
            if x2 >= 0 and x2 not in a:
                candidates.append(x2)
                
        candidates = sorted(set(candidates))
        
        # Sort candidates by the resulting max element to prefer compact sets
        candidates.sort(key=lambda x: max(a + [x]))
        
        for x in candidates:
            trial = sorted(a + [x])
            if _is_sidon_fast(trial):
                a = trial
                found = True
                break
        
        if not found:
            # If no single element works, add two elements y, y+d
            # This is rare for small d, but handle it
            y = max(a) + 1
            while time.time() < _DEADLINE:
                trial = sorted(a + [y, y + d])
                if _is_sidon_fast(trial):
                    a = trial
                    break
                y += 1
    return a

def _local_search():
    """
    Local search to minimize max element.
    """
    best = None
    best_val = 10**9
    rng = random.Random(42)
    
    # Start from greedy
    start_set = _greedy_construct()
    if not _valid(start_set):
        # Fallback to a random valid set if greedy fails (unlikely)
        return None
    
    current = start_set[:]
    current_val = max(current)
    
    # If greedy is already good, try to improve
    if current_val < best_val and _valid(current):
        best = current[:]
        best_val = current_val
        
    steps = 0
    max_steps = 50000
    
    while time.time() < _DEADLINE and steps < max_steps:
        steps += 1
        # Pick a random non-zero element to move
        idx = rng.randint(1, len(current) - 1)
        old_val = current[idx]
        
        # Try moving it to a nearby value
        # Bias towards smaller values to reduce max
        delta = rng.randint(-20, 10)
        new_val = old_val + delta
        
        if new_val <= 0:
            continue
            
        # Check for duplicates
        if new_val in current and new_val != old_val:
            continue
            
        # Create candidate
        candidate = current[:]
        candidate[idx] = new_val
        candidate.sort()
        
        # Check Sidon and Coverage
        if _is_sidon_fast(candidate) and _covers_1_to_30_fast(candidate):
            val = max(candidate)
            if val < current_val:
                current = candidate
                current_val = val
                if val < best_val:
                    best_val = val
                    best = current[:]
            else:
                # Accept worse moves with small probability (Simulated Annealing style)
                # Or just reject to keep it simple and fast
                pass
                
    return best

def construct():
    t0 = time.time()
    
    # 1. Greedy
    g = _greedy_construct()
    best = g
    best_val = max(g)
    
    # 2. Local Search
    ls = _local_search()
    if ls is not None and max(ls) < best_val:
        best = ls
        best_val = max(ls)
        
    return best
# EVOLVE-BLOCK-END
