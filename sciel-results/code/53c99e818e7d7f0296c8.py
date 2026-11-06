import time
import random
from itertools import combinations

def _any_triple_divisible(S_list, x):
    """Check if x divides the sum of any 3 distinct elements in S_list."""
    m = len(S_list)
    if m < 3:
        return False
    for i in range(m - 2):
        a = S_list[i]
        for j in range(i + 1, m - 1):
            b = S_list[j]
            ab = a + b
            for k in range(j + 1, m):
                if (ab + S_list[k]) % x == 0:
                    return True
    return False

def _can_add(S, x):
    """
    Check if adding x to set S maintains the validity condition.
    Condition: No element in S U {x} divides the sum of any 3 distinct OTHER elements in S U {x}.
    """
    S_list = list(S)
    m = len(S_list)
    
    # 1) Check if x divides the sum of any 3 distinct elements in S.
    if m >= 3:
        if _any_triple_divisible(S_list, x):
            return False

    # 2) Check for each y in S, if y divides (x + u + v) for any distinct u,v in S \ {y}.
    #    Note: u and v must be distinct from y and from each other.
    #    (x is not in S, so u,v come from S \ {y})
    
    for y in S_list:
        target = (-x) % y
        # Find if any pair u,v in S \ {y} sums to target mod y.
        
        # Create list of elements in S excluding y
        others = [v for v in S_list if v != y]
        
        seen_residues = set()
        for u in others:
            ru = u % y
            needed_rv = (target - ru) % y
            
            if needed_rv in seen_residues:
                return False
            
            seen_residues.add(ru)
            
    return True

def _is_valid(S):
    """Full validation."""
    S_list = list(S)
    m = len(S_list)
    for a in S_list:
        others = [v for v in S_list if v != a]
        if len(others) < 3:
            continue
        if _any_triple_divisible(others, a):
            return False
    return True

def _greedy(order, deadline):
    current_set = set()
    S_list = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(current_set, x):
            current_set.add(x)
            S_list.append(x)
    return current_set, S_list

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 192
    best_set = set()
    
    # Strategy 1: Descending Greedy
    order = list(range(n, 0, -1))
    current_set, _ = _greedy(order, deadline)
    if len(current_set) > len(best_set):
        best_set = current_set.copy()
        
    # Strategy 2: Randomized Greedy (Multiple Restarts)
    rng = random.Random(42)
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Bias towards larger numbers
        order = list(range(n, 0, -1))
        rng.shuffle(order)
        
        current_set, _ = _greedy(order, deadline)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
    # Local Search / Improvement
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try adding any unselected element
        for x in range(1, n + 1):
            if time.time() > deadline - 5.0:
                break
            if x not in best_set:
                if _can_add(best_set, x):
                    best_set.add(x)
                    improved = True
                    break
        
        # If no direct add, try 1-out 2-in swap
        if not improved:
            for _ in range(20):
                if time.time() > deadline - 5.0:
                    break
                if len(best_set) < 2:
                    break
                a = rng.choice(list(best_set))
                trial_set = best_set - {a}
                
                added = False
                for x in range(1, n + 1):
                    if x in trial_set:
                        continue
                    if _can_add(trial_set, x):
                        trial_set.add(x)
                        added = True
                        for y in range(1, n + 1):
                            if y in trial_set:
                                continue
                            if _can_add(trial_set, y):
                                trial_set.add(y)
                                best_set = trial_set
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break

    # Final Validation
    if _is_valid(best_set):
        return list(best_set)
    
    # Fallback
    return [192]
