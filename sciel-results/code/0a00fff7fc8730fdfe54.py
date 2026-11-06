"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from functools import lru_cache

def _valid_set(S):
    """Check if set S is valid: no element divides sum of any 3 distinct OTHER elements."""
    S_list = list(S)
    n = len(S_list)
    if n < 4:
        return True
    # For each element a in S, check that a does not divide sum of any 3 distinct other elements
    for i, a in enumerate(S_list):
        others = [x for j, x in enumerate(S_list) if j != i]
        # Check all combinations of 3 from others
        for combo in combinations(others, 3):
            s = combo[0] + combo[1] + combo[2]
            if s % a == 0:
                return False
    return True

def _can_add(S, x):
    """Check if x can be added to set S while maintaining validity."""
    S_list = list(S)
    # Check: for each existing element a in S, a must not divide sum of any 3 distinct others in S∪{x}
    # The new triples involving x are: (x, b, c) where b,c are distinct elements in S
    # For each a in S: a must not divide (x + b + c) for any distinct b,c in S
    # Also: x must not divide (a + b + c) for any distinct a,b,c in S
    # And: for each a in S, the old condition still holds (no change since we only add triples with x)
    
    # Check x doesn't divide sum of any 3 distinct elements in S
    for combo in combinations(S_list, 3):
        s = combo[0] + combo[1] + combo[2]
        if s % x == 0:
            return False
    
    # Check each a in S doesn't divide (x + b + c) for distinct b,c in S
    for a in S_list:
        for i, b in enumerate(S_list):
            for j, c in enumerate(S_list):
                if i != j and b != c:
                    s = x + b + c
                    if s % a == 0:
                        return False
    return True

def _greedy_largest_first():
    """Greedy: try elements from largest to smallest."""
    selected = []
    for x in range(192, 0, -1):
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _greedy_smallest_first():
    """Greedy: try elements from smallest to largest."""
    selected = []
    for x in range(1, 193):
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_search(initial_set, max_iters=5000, time_limit=8.0):
    """Local search: try swapping elements to improve cardinality."""
    S = list(initial_set)
    best_size = len(S)
    start_time = time.time()
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        # Try to add a random element not in S
        candidates = [x for x in range(1, 193) if x not in S]
        if not candidates:
            break
        x = random.choice(candidates)
        if _can_add(S, x):
            S.append(x)
            best_size = max(best_size, len(S))
            continue
        
        # Try swapping: remove one element, add another
        if len(S) > 0:
            i = random.randint(0, len(S) - 1)
            removed = S[i]
            S_no_removed = [v for j, v in enumerate(S) if j != i]
            
            # Try adding a candidate
            candidates2 = [x for x in range(1, 193) if x not in S_no_removed and x != removed]
            random.shuffle(candidates2)
            for x in candidates2[:20]:  # Limit trials
                if _can_add(S_no_removed, x):
                    S = S_no_removed + [x]
                    best_size = max(best_size, len(S))
                    break
    
    return S

def _multi_start_greedy(num_starts=5, time_limit=15.0):
    """Try multiple greedy strategies with different orderings."""
    best = []
    start_time = time.time()
    
    # Standard greedy largest first
    best = _greedy_largest_first()
    
    # Standard greedy smallest first
    s2 = _greedy_smallest_first()
    if len(s2) > len(best):
        best = s2
    
    # Randomized greedy with different seeds
    for seed in range(num_starts):
        if time.time() - start_time > time_limit:
            break
        random.seed(seed * 12345 + 7)
        S = []
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        for x in candidates:
            if _can_add(S, x):
                S.append(x)
        if len(S) > len(best):
            best = S
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Phase 1: Multiple greedy strategies
    best = _multi_start_greedy(num_starts=10, time_limit=10.0)
    
    if time.time() > deadline:
        return best
    
    # Phase 2: Local search from best solution
    best = _local_search(best, max_iters=10000, time_limit=10.0)
    
    if time.time() > deadline:
        return best
    
    # Phase 3: More local search with different random seeds
    for seed in range(5):
        if time.time() > deadline:
            break
        random.seed(seed * 999 + 42)
        S = _local_search(list(best), max_iters=5000, time_limit=5.0)
        if len(S) > len(best):
            best = S
    
    # Phase 4: Try to add more elements if possible
    if time.time() < deadline - 2.0:
        S = list(best)
        for x in range(1, 193):
            if x not in S and _can_add(S, x):
                S.append(x)
        if len(S) > len(best):
            best = S
    
    return best
# EVOLVE-BLOCK-END
