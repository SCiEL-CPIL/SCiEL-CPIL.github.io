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
    """Check if S is inclusion-maximal Sidon in [1,N]."""
    if sums is None:
        sums = _sidon_sums(S)
        if sums is None:
            return False
    
    S_set = set(S)
    
    # Compute blocked set B
    # x is blocked if 2x in sums OR exists a in S such that x+a in sums
    # x in sums/2 (even) OR x in sums - a
    
    blocked = set()
    
    # From 2x in sums
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

def _greedy_sidon(randomize=False, seed=None):
    """Build a Sidon set greedily."""
    if seed is not None:
        random.seed(seed)
    selected = []
    sums = set()
    candidates = list(range(1, N + 1))
    if randomize:
        random.shuffle(candidates)
    for x in candidates:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    selected.sort()
    return selected

def _try_minimize_by_removal(S, N, deadline=None):
    """Try to remove elements while keeping Sidon + maximal."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        sums = _sidon_sums(S)
        if sums is None:
            break
        for i in range(len(S)):
            if deadline is not None and time.time() > deadline:
                return S
            candidate = S[:i] + S[i+1:]
            if _sidon_and_maximal(candidate, N):
                S = candidate
                improved = True
                break
    return S

def _local_search_swap(S, N, max_swaps=50, deadline=None):
    """Try to improve Sidon set by swapping elements."""
    S = list(S)
    S_set = set(S)
    for _ in range(max_swaps):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try removing one and adding another
        for i in range(len(S)):
            if deadline is not None and time.time() > deadline:
                break
            # Try all possible new elements
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                new_S = S[:i] + S[i+1:] + [x]
                # Check Sidon
                sums = _sidon_sums(new_S)
                if sums is None:
                    continue
                # Check Maximal
                if _is_maximal(new_S, N, sums):
                    S = new_S
                    S_set = set(S)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return S

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1

    # Phase 1: Try several random greedy constructions
    for seed in range(20000):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 5:
            break
            
        # Vary the strategy slightly
        if seed % 2 == 0:
            S = _greedy_sidon(randomize=True, seed=seed)
        else:
            # Reverse greedy?
            S = _greedy_sidon(randomize=False)
            random.seed(seed)
            random.shuffle(S)
            # Re-greedy from shuffled? No, greedy is order dependent.
            # Let's just use random shuffle for greedy
            S = _greedy_sidon(randomize=True, seed=seed + 100000)

        if len(S) < best_size:
            S_min = _try_minimize_by_removal(S, N, deadline=_DEADLINE - 3)
            if _sidon_and_maximal(S_min, N) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min
                # Try local search
                S_swapped = _local_search_swap(S_min, N, max_swaps=10, deadline=_DEADLINE - 2)
                if _sidon_and_maximal(S_swapped, N) and len(S_swapped) < best_size:
                    best_size = len(S_swapped)
                    best_set = S_swapped

    if best_set is None:
        S = _greedy_sidon(randomize=False)
        S_min = _try_minimize_by_removal(S, N)
        if _sidon_and_maximal(S_min, N):
            best_set = S_min
        else:
            best_set = S

    return best_set
