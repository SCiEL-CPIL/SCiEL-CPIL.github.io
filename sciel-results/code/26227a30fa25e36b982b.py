"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 128
TIME_LIMIT = 80.0  # Safety margin before 90s kill

# Precompute forbidden triples: (a, b, c) with a < b < c, all in 1..N,
# such that a+b, a+c, b+c are all in 1..N.
# A valid set cannot contain any such triple.
# We'll represent this as a list of forbidden triples.

def _precompute_forbidden_triples():
    """Precompute all forbidden triples (a,b,c) with a<b<c where all pair sums are in [1, N]."""
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            ab = a + b
            if ab > N:
                break
            for c in range(b + 1, N + 1):
                ac = a + c
                if ac > N:
                    break
                bc = b + c
                if bc > N:
                    break
                # Check that a, b, c are distinct (they are by construction)
                # and that a+b, a+c, b+c are all in [1, N] (checked above)
                triples.append((a, b, c))
    return triples

_FORBIDDEN_TRIPLES = _precompute_forbidden_triples()

# For each element x, precompute which other elements it appears in a forbidden triple with.
# conflict_pairs[x] = set of (y, z) pairs such that {x, y, z} is a forbidden triple
# Actually, for greedy construction, we need: given current set S, can we add x?
# x can be added iff no forbidden triple is fully contained in S ∪ {x}.
# A forbidden triple (a,b,c) becomes fully contained if:
#   - x is one of {a,b,c} and the other two are already in S
#   - x is not one of {a,b,c} but... wait, if x is not in the triple, then the triple was already 
#     either fully in S (invalid, shouldn't happen) or not fully in S (adding x doesn't complete it)
# So only case 1 matters: x completes a forbidden triple.

# For each x, store the set of pairs (y, z) with y < z, y != x, z != x, such that {x, y, z} is forbidden.
# When adding x, we need to check: for each (y, z) in conflict_pairs[x], if y in S and z in S, conflict.

def _precompute_conflict_pairs():
    """For each x, compute set of (y, z) pairs that form a forbidden triple with x."""
    conflict = {x: set() for x in range(1, N + 1)}
    for (a, b, c) in _FORBIDDEN_TRIPLES:
        conflict[a].add((b, c))
        conflict[b].add((a, c))
        conflict[c].add((a, b))
    return conflict

_CONFLICT_PAIRS = _precompute_conflict_pairs()

def _can_add(S_set, x):
    """Check if x can be added to S without creating a forbidden triple."""
    if x in S_set:
        return False
    for (y, z) in _CONFLICT_PAIRS[x]:
        if y in S_set and z in S_set:
            return False
    return True

def _greedy_random_order():
    """Greedy construction with random vertex ordering."""
    S = set()
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    for x in candidates:
        if _can_add(S, x):
            S.add(x)
    return S

def _greedy_weighted_order():
    """Greedy with weighted random ordering - prefer elements with fewer conflicts."""
    S = set()
    candidates = list(range(1, N + 1))
    # Weight by inverse of number of conflicts
    weights = [1.0 / max(1, len(_CONFLICT_PAIRS[x])) for x in candidates]
    # Use weighted sampling
    while candidates:
        total_w = sum(weights)
        r = random.random() * total_w
        cum = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                x = candidates.pop(i)
                weights.pop(i)
                if _can_add(S, x):
                    S.add(x)
                break
    return S

def _local_search(S, time_limit=10.0):
    """Local search: try removing one element and adding others."""
    start = time.time()
    best_S = set(S)
    best_size = len(best_S)
    current_S = set(S)
    
    while time.time() - start < time_limit:
        # Try removing a random element and then greedily adding
        if not current_S:
            break
        to_remove = random.choice(list(current_S))
        trial_S = set(current_S)
        trial_S.remove(to_remove)
        
        # Try to add elements
        candidates = [x for x in range(1, N + 1) if x not in trial_S]
        random.shuffle(candidates)
        for x in candidates:
            if _can_add(trial_S, x):
                trial_S.add(x)
        
        if len(trial_S) > best_size:
            best_S = set(trial_S)
            best_size = len(trial_S)
        
        # Accept move
        if len(trial_S) >= len(current_S) or random.random() < 0.3:
            current_S = trial_S
    
    return best_S

def _is_valid(S):
    """Verify that set S has no forbidden triple."""
    S_set = set(S)
    for (a, b, c) in _FORBIDDEN_TRIPLES:
        if a in S_set and b in S_set and c in S_set:
            return False
    return True

def construct():
    start_time = time.time()
    
    best_S = set()
    best_size = 0
    
    # Multiple restarts with different strategies
    while time.time() - start_time < TIME_LIMIT:
        remaining = TIME_LIMIT - (time.time() - start_time)
        
        # Try greedy with random order
        S = _greedy_random_order()
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
        
        # Try greedy with weighted order
        S = _greedy_weighted_order()
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
        
        # Local search if time permits
        if remaining > 15.0:
            ls_time = min(remaining - 10.0, 10.0)
            S = _local_search(best_S, time_limit=ls_time)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
        
        # Early termination if we find a very good solution
        if best_size >= 85:
            break
    
    # Final verification
    if not _is_valid(best_S):
        # Fallback: upper half is always valid (sums exceed N)
        best_S = set(range(N // 2 + 1, N + 1))
    
    return list(best_S)
# EVOLVE-BLOCK-END
