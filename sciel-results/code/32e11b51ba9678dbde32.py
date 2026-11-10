"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _compatible(a, b):
    """Check if a and b can coexist: (a*b) % (a+b) != 0"""
    return (a * b) % (a + b) != 0

def _is_compatible_with_set(x, s):
    """Check if x is compatible with every element in set s"""
    for y in s:
        if x == y:
            continue
        if (x * y) % (x + y) == 0:
            return False
    return True

def _construct_greedy_randomized(rng, time_limit):
    """Randomized greedy: randomly order candidates, add if compatible."""
    candidates = list(range(1, 257))
    rng.shuffle(candidates)
    result = []
    for x in candidates:
        if _is_compatible_with_set(x, result):
            result.append(x)
    return result

def _construct_odds_plus_greedy_even(rng, time_limit):
    """Start with all odds, then greedily add compatible evens."""
    result = list(range(1, 257, 2))  # all odds
    evens = list(range(2, 257, 2))
    rng.shuffle(evens)
    for x in evens:
        if _is_compatible_with_set(x, result):
            result.append(x)
    return result

def _local_search(rng, time_limit, s):
    """Try to improve set s by adding elements or swapping."""
    s = list(s)
    s_set = set(s)
    candidates = [x for x in range(1, 257) if x not in s_set]
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Try adding
        rng.shuffle(candidates)
        for x in candidates:
            if _is_compatible_with_set(x, s):
                s.append(x)
                s_set.add(x)
                candidates.remove(x)
                improved = True
                break
        if improved:
            continue
        
        # Try swaps: remove one, add two
        n = len(s)
        for i in range(n):
            if time.time() >= time_limit:
                break
            removed = s[i]
            s_trial = s[:i] + s[i+1:]
            s_trial_set = s_set - {removed}
            added = 0
            for x in candidates:
                if x in s_trial_set:
                    continue
                if _is_compatible_with_set(x, s_trial):
                    s_trial.append(x)
                    s_trial_set.add(x)
                    added += 1
                    if added >= 2:
                        break
            if added >= 2:
                s = s_trial
                s_set = s_trial_set
                candidates = [x for x in range(1, 257) if x not in s_set]
                improved = True
                break
    
    return s

def _local_search_swap1(rng, time_limit, s):
    """Try to improve by removing one element and adding one that allows more additions."""
    s = list(s)
    s_set = set(s)
    
    for iteration in range(200):
        if time.time() >= time_limit:
            break
        best_gain = 0
        best_swap = None
        
        n = len(s)
        # Sample a subset of removal candidates for speed
        removals = list(range(n))
        rng.shuffle(removals)
        removals = removals[:min(50, n)]
        
        for i in removals:
            removed = s[i]
            s_trial = s[:i] + s[i+1:]
            s_trial_set = s_set - {removed}
            
            # Try to add as many as possible from candidates
            added = []
            for x in range(1, 257):
                if x in s_trial_set:
                    continue
                if _is_compatible_with_set(x, s_trial):
                    s_trial.append(x)
                    s_trial_set.add(x)
                    added.append(x)
            
            gain = len(added) - 1
            if gain > best_gain:
                best_gain = gain
                best_swap = (i, added)
        
        if best_swap is not None and best_gain > 0:
            i, added = best_swap
            removed = s[i]
            s = s[:i] + s[i+1:]
            for x in added:
                s.append(x)
    
    return s

def construct():
    deadline = time.time() + 80  # 80s safety margin for 90s limit
    
    rng = random.Random(42)
    
    best = None
    best_size = 0
    
    # Phase 1: Multiple randomized greedy constructions
    for i in range(50):
        if time.time() > deadline - 30:
            break
        s = _construct_odds_plus_greedy_even(rng, deadline - 30)
        if len(s) > best_size:
            best_size = len(s)
            best = s
    
    # Also try pure randomized greedy
    for i in range(20):
        if time.time() > deadline - 30:
            break
        s = _construct_greedy_randomized(rng, deadline - 30)
        if len(s) > best_size:
            best_size = len(s)
            best = s
    
    if best is None:
        best = list(range(1, 257, 2))
    
    # Phase 2: Local search on best solution
    if time.time() < deadline - 10:
        s = _local_search(rng, deadline - 10, best)
        if len(s) > best_size:
            best_size = len(s)
            best = s
    
    # Phase 3: More local search iterations
    if time.time() < deadline - 5:
        s = _local_search_swap1(rng, deadline - 5, best)
        if len(s) > best_size:
            best_size = len(s)
            best = s
    
    # Final: try more local search
    if time.time() < deadline - 2:
        s = _local_search(rng, deadline - 2, best)
        if len(s) > best_size:
            best_size = len(s)
            best = s
    
    return best
# EVOLVE-BLOCK-END
