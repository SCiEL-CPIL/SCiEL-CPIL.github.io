import time
import random
import sys

N = 256
_DEADLINE = 55.0

def _sidon_sums(S):
    """Return the set of all pairwise sums a+b for a<=b in S, or None if not Sidon."""
    sums = set()
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i, n):
            s = a + S[j]
            if s in sums:
                return None
            sums.add(s)
    return sums

def _is_sidon(S):
    return _sidon_sums(S) is not None

def _is_maximal(S, N, sums=None):
    """Check if S is inclusion-maximal Sidon in [1,N] using blocked-set method."""
    if sums is None:
        sums = _sidon_sums(S)
        if sums is None:
            return False
    
    S_set = set(S)
    
    # x is blocked if 2x in sums OR exists a in S such that x+a in sums
    blocked = set()
    
    # From 2x in sums: x = s//2 if s is even
    for s in sums:
        if s % 2 == 0:
            x = s // 2
            if 1 <= x <= N:
                blocked.add(x)
                
    # From x + a in sums => x = s - a
    for s in sums:
        for a in S:
            x = s - a
            if 1 <= x <= N:
                blocked.add(x)
                
    # Check if all x in [1, N] not in S are in blocked
    for x in range(1, N + 1):
        if x not in S_set and x not in blocked:
            return False
            
    return True

def _sidon_and_maximal(S, N):
    if not S:
        return False
    sums = _sidon_sums(S)
    if sums is None:
        return False
    return _is_maximal(S, N, sums)

def _greedy_sidon(order=None):
    """Build a Sidon set greedily given a specific candidate order."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    selected.sort()
    return selected

def _greedy_sidon_random(seed):
    """Build a Sidon set greedily with random candidate order."""
    rng = random.Random(seed)
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    return _greedy_sidon(candidates)

def _greedy_sidon_structured(seed, pattern):
    """Build a Sidon set with structured ordering."""
    rng = random.Random(seed)
    if pattern == 0:
        # Random
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
    elif pattern == 1:
        # Large to small
        candidates = list(range(N, 0, -1))
    elif pattern == 2:
        # Small to large
        candidates = list(range(1, N + 1))
    elif pattern == 3:
        # Alternating high/low
        low = list(range(1, N // 2 + 1))
        high = list(range(N // 2 + 1, N + 1))
        candidates = []
        for i in range(N // 2 + 1):
            if i < len(high):
                candidates.append(high[i])
            if i < len(low):
                candidates.append(low[i])
    elif pattern == 4:
        # Prime-like spacing
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
    elif pattern == 5:
        # Quadratic ordering
        candidates = sorted(range(1, N + 1), key=lambda x: (x * x) % (N + 1))
    else:
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
    return _greedy_sidon(candidates)

def _try_minimize_by_removal(S, N, deadline=None):
    """Try to remove elements while keeping Sidon + maximal. Exhaustive removal search."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.time() > deadline:
            return S
        sums = _sidon_sums(S)
        if sums is None:
            break
        n = len(S)
        # Try removing each element; prefer removing elements that are "less critical"
        # Sort by index for determinism, but try all
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                return S
            candidate = S[:i] + S[i+1:]
            if _sidon_and_maximal(candidate, N):
                S = candidate
                improved = True
                break  # Restart the loop with the new S
    return S

def _add_then_remove(S, N, seed, deadline=None):
    """Try adding a random element (if Sidon-preserving), then minimize."""
    rng = random.Random(seed)
    S = list(S)
    sums = _sidon_sums(S)
    if sums is None:
        return S
    
    # Try adding a random element
    S_set = set(S)
    candidates = [x for x in range(1, N + 1) if x not in S_set]
    rng.shuffle(candidates)
    
    for x in candidates[:10]:  # Try up to 10 random additions
        if deadline is not None and time.time() > deadline:
            break
        new_S = S + [x]
        new_sums = _sidon_sums(new_S)
        if new_sums is None:
            continue
        # Now try to minimize
        minimized = _try_minimize_by_removal(new_S, N, deadline=deadline)
        if len(minimized) < len(S):
            return minimized
    return S

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1
    
    def _update_best(S):
        nonlocal best_set, best_size
        if S and _sidon_and_maximal(S, N) and len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Phase 1: Generate diverse initial candidates and minimize
    # Use multiple structured patterns and random orders
    patterns = [0, 1, 2, 3, 4, 5]
    
    for pattern in patterns:
        for seed in range(300):
            elapsed = time.time() - start_time
            if elapsed > _DEADLINE - 8:
                break
                
            S = _greedy_sidon_structured(seed, pattern)
            if len(S) >= best_size:
                continue
                
            S_min = _try_minimize_by_removal(S, N, deadline=_DEADLINE - 5)
            _update_best(S_min)
            
            if elapsed > _DEADLINE - 10:
                break
    
    # Phase 2: From the best set, try add-then-remove moves
    if best_set is not None:
        for seed in range(500):
            elapsed = time.time() - start_time
            if elapsed > _DEADLINE - 3:
                break
            
            improved = _add_then_remove(best_set, N, seed, deadline=_DEADLINE - 2)
            _update_best(improved)
    
    # Phase 3: If still no good result, fall back to deterministic
    if best_set is None:
        S = _greedy_sidon(list(range(1, N + 1)))
        S_min = _try_minimize_by_removal(S, N)
        _update_best(S_min)
        if best_set is None:
            best_set = S
    
    # Final safety: ensure we return a valid set
    if best_set is None or not _sidon_and_maximal(best_set, N):
        S = _greedy_sidon(list(range(1, N + 1)))
        S_min = _try_minimize_by_removal(S, N)
        if _sidon_and_maximal(S_min, N):
            best_set = S_min
        else:
            best_set = S
    
    return best_set
