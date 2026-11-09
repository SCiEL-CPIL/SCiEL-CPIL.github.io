"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 128}
DEADLINE = 85.0  # safety margin before 90s kill

def _check_validity(S, n=128):
    """Check if set S is valid: no three distinct members a,b,c with a+b, a+c, b+c all in S."""
    S_set = set(S)
    for a, b, c in combinations(sorted(S), 3):
        if (a + b) in S_set and (a + c) in S_set and (b + c) in S_set:
            return False
    return True

def _is_valid(S_set, a, b, c, n=128):
    """Check if adding triple (a,b,c) would create a violation."""
    return (a + b) in S_set and (a + c) in S_set and (b + c) in S_set

def _can_add(S_set, x, n=128):
    """Check if x can be added to S without creating a violation."""
    for a, b in combinations(sorted(S_set), 2):
        if _is_valid(S_set, a, b, x, n):
            return False
    return True

def _greedy_construct(n=128):
    """Greedy construction: add numbers from high to low if possible."""
    S = set()
    for x in range(n, 0, -1):
        if _can_add(S, x, n):
            S.add(x)
    return S

def _local_search(initial_set, max_iters=500, deadline=None):
    """Local search: try swaps to improve set size."""
    S = set(initial_set)
    best_size = len(S)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        # Try adding elements
        for x in range(1, 129):
            if x not in S:
                if _can_add(S, x, 128):
                    S.add(x)
                    improved = True
                    break
        
        if not improved:
            # Try swaps: remove y, add x
            removed = None
            added = None
            for y in list(S):
                if removed is not None:
                    break
                for x in range(1, 129):
                    if x not in S and x != y:
                        S_test = S - {y} | {x}
                        if _check_validity(list(S_test), 128):
                            removed = y
                            added = x
                            break
                if removed is not None:
                    break
            
            if removed is not None and added is not None:
                S.remove(removed)
                S.add(added)
                improved = True
        
        if not improved:
            break
        
        if len(S) > best_size:
            best_size = len(S)
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with greedy construction
    S_greedy = _greedy_construct(128)
    best = S_greedy
    best_size = len(S_greedy)
    
    # Try local search with random perturbations
    rng = random.Random(42)
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        # Perturb: start from greedy with some random removals
        S_current = set(best)
        for _ in range(rng.randint(1, 5)):
            if S_current:
                S_current.discard(rng.choice(list(S_current)))
        
        S_new = _local_search(S_current, max_iters=200, deadline=deadline - 2)
        
        if len(S_new) > best_size:
            best = S_new
            best_size = len(S_new)
        
        # Also try different greedy orders
        for start_val in [64, 65, 66, 67, 68]:
            if time.time() > deadline - 5:
                break
            S_g = set(range(start_val, 129))
            if _check_validity(list(S_g), 128):
                S_new = _local_search(S_g, max_iters=100, deadline=deadline - 2)
                if len(S_new) > best_size:
                    best = S_new
                    best_size = len(S_new)
    
    return list(best)
# EVOLVE-BLOCK-END
