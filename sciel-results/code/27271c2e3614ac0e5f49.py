"""Improved program: search for 32-element subset of 1..512 maximizing distinct 3-term AP differences."""
import time
import random
import numpy as np

def _score_set(s):
    """Return the number of distinct d>0 such that a, a+d, a+2d in s for some a."""
    s_set = set(s)
    good = set()
    n = len(s)
    # For each pair (a, a+d) in s, check if a+2d in s
    # Efficient: iterate over all pairs
    for i in range(n):
        for j in range(i+1, n):
            d = s[j] - s[i]
            if d > 0 and (s[j] + d) in s_set:
                good.add(d)
    return len(good)

def _greedy_construct(n=32, ub=512, time_limit=110.0):
    """Greedy + local search construction."""
    t0 = time.time()
    
    # Start with a good initial set
    # Try several random starts and pick the best
    best_set = None
    best_score = 0
    
    # Initial: use a structured set, then refine
    # Start with evenly spaced or random
    for attempt in range(20):
        if time.time() - t0 > time_limit * 0.3:
            break
        if attempt == 0:
            # Evenly spaced
            s = list(range(1, n+1))
        elif attempt == 1:
            # Powers of 2-ish
            s = [1]
            while len(s) < n:
                s.append(s[-1] * 2)
            s = s[:n]
            # Scale to fit in 1..512
            scale = 512 // max(s)
            if scale < 1:
                scale = 1
            s = sorted(set(x * scale + 1 for x in s))
            while len(s) < n:
                s.append(max(s) + 1)
            s = sorted(s[:n])
        else:
            s = sorted(random.sample(range(1, ub+1), n))
        
        sc = _score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)
    
    if best_set is None:
        best_set = list(range(1, n+1))
        best_score = _score_set(best_set)
    
    # Local search: try swapping elements
    improved = True
    while improved and (time.time() - t0) < time_limit:
        improved = False
        s = list(best_set)
        s_set = set(s)
        current_score = best_score
        
        # Try each element, try replacing with a random other value
        for i in range(n):
            if time.time() - t0 > time_limit:
                break
            old_val = s[i]
            # Try a few random replacements
            for _ in range(5):
                new_val = random.randint(1, ub)
                if new_val in s_set and new_val != old_val:
                    continue
                # Remove old, add new
                s_set.remove(old_val)
                s_set.add(new_val)
                s[i] = new_val
                new_score = _score_set(s)
                if new_score > current_score:
                    current_score = new_score
                    best_set = list(s)
                    improved = True
                else:
                    # Revert
                    s[i] = old_val
                    s_set.remove(new_val)
                    s_set.add(old_val)
    
    return best_set

def _simulated_annealing(n=32, ub=512, time_limit=115.0):
    """Simulated annealing to improve the set."""
    t0 = time.time()
    
    # Start from a good initial set
    s = _greedy_construct(n, ub, time_limit * 0.2)
    current_score = _score_set(s)
    best_set = list(s)
    best_score = current_score
    
    temp = 50.0
    cooling = 0.999
    
    while (time.time() - t0) < time_limit:
        # Pick a random element and perturb it
        i = random.randint(0, n-1)
        old_val = s[i]
        
        # Try to move to a nearby value or a random value
        if random.random() < 0.5:
            delta = random.randint(-10, 10)
            new_val = old_val + delta
        else:
            new_val = random.randint(1, ub)
        
        if new_val < 1 or new_val > ub:
            continue
        if new_val in s and s.index(new_val) != i:
            continue
        
        s[i] = new_val
        new_score = _score_set(s)
        
        delta_score = new_score - current_score
        if delta_score > 0 or random.random() < np.exp(delta_score / max(temp, 0.1)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = list(s)
        else:
            s[i] = old_val
        
        temp *= cooling
    
    return best_set

def construct():
    """Return exactly 32 distinct integers in 1..512 maximizing distinct 3-term AP differences."""
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin before 150s kill
    
    # Phase 1: Greedy construction
    s = _greedy_construct(32, 512, 30.0)
    best_score = _score_set(s)
    best_set = list(s)
    
    # Phase 2: Simulated annealing for refinement
    if time.time() < deadline - 20.0:
        s2 = _simulated_annealing(32, 512, deadline - time.time() - 5.0)
        sc2 = _score_set(s2)
        if sc2 > best_score:
            best_score = sc2
            best_set = list(s2)
    
    # Phase 3: Final local polish
    if time.time() < deadline - 2.0:
        improved = True
        s = list(best_set)
        current_score = best_score
        while improved and (time.time() < deadline - 2.0):
            improved = False
            s_set = set(s)
            for i in range(32):
                if time.time() > deadline - 2.0:
                    break
                old_val = s[i]
                for _ in range(3):
                    new_val = random.randint(1, 512)
                    if new_val in s_set and new_val != old_val:
                        continue
                    s_set.remove(old_val)
                    s_set.add(new_val)
                    s[i] = new_val
                    new_score = _score_set(s)
                    if new_score > current_score:
                        current_score = new_score
                        best_set = list(s)
                        improved = True
                    else:
                        s[i] = old_val
                        s_set.remove(new_val)
                        s_set.add(old_val)
    
    # Ensure valid output
    result = sorted(set(best_set))
    if len(result) != 32:
        # Fix: add or remove to get exactly 32
        if len(result) < 32:
            existing = set(result)
            for x in range(1, 513):
                if x not in existing and len(result) < 32:
                    result.append(x)
        else:
            result = result[:32]
        result = sorted(result)
    
    return result
