import time
import random
from itertools import combinations

def count_conflicts(S):
    """Count the number of unordered pair-sum collisions in S.
    A collision is two distinct pairs (i,j) and (k,l) with i<=j, k<=l,
    (i,j) != (k,l), and S[i]+S[j] == S[k]+S[l].
    More collisions means the set is 'harder' to extract a large Sidon subset from."""
    sums = {}
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    # Total collisions = sum over all sums of C(count, 2)
    total = 0
    for c in sums.values():
        if c > 1:
            total += c * (c - 1) // 2
    return total

def local_search(S, max_iter=5000, time_budget=None, start_time=None):
    """Stochastic local search: swap elements to maximize conflict count."""
    S = sorted(S)
    current_conflicts = count_conflicts(S)
    best_S = list(S)
    best_conflicts = current_conflicts
    n = len(S)
    universe = list(range(129))
    
    for it in range(max_iter):
        if time_budget is not None and start_time is not None:
            if time.time() - start_time > time_budget:
                break
        
        # Pick a random element to swap out
        idx = random.randrange(n)
        old_val = S[idx]
        
        # Pick a random replacement from universe not in S
        while True:
            new_val = random.choice(universe)
            if new_val not in S:
                break
        
        # Perform swap
        S[idx] = new_val
        new_conflicts = count_conflicts(S)
        
        if new_conflicts > best_conflicts:
            best_conflicts = new_conflicts
            best_S = list(S)
        elif new_conflicts >= current_conflicts:
            current_conflicts = new_conflicts
        else:
            # Revert
            S[idx] = old_val
    
    return best_S, best_conflicts

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_S = None
    best_conflicts = -1
    
    # Initial candidates: try several structured starts
    initial_sets = []
    # Consecutive
    initial_sets.append(list(range(12)))
    # Arithmetic progression with large step
    initial_sets.append([i * 11 for i in range(12)])
    # Random starts
    for _ in range(10):
        initial_sets.append(sorted(random.sample(range(129), 12)))
    
    for S0 in initial_sets:
        if time.time() > deadline:
            break
        S, conflicts = local_search(S0, max_iter=3000, time_budget=deadline - time.time(), start_time=start_time)
        if conflicts > best_conflicts:
            best_conflicts = conflicts
            best_S = S
    
    # Additional random restarts if time remains
    while time.time() < deadline - 2.0:
        S0 = sorted(random.sample(range(129), 12))
        S, conflicts = local_search(S0, max_iter=2000, time_budget=deadline - time.time() - 2.0, start_time=start_time)
        if conflicts > best_conflicts:
            best_conflicts = conflicts
            best_S = S
    
    if best_S is None:
        return list(range(12))
    
    return sorted(best_S)
