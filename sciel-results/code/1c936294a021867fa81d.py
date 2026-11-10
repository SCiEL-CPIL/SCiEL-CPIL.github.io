"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
_DEADLINE = 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START
def _compute_sums(S):
    """Compute all pair sums a+b for a<=b in S. Returns (sums_set, is_sidon)."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return sums, False
            sums.add(s)
    return sums, True

def _is_maximal_fast(S, N, sums=None):
    """Check if S is inclusion-maximal Sidon in [1,N]. Uses precomputed sums if given."""
    S_set = set(S)
    if sums is None:
        sums, ok = _compute_sums(S)
        if not ok:
            return False
    
    # For x to be addable: 2x not in sums AND for all a in S, x+a not in sums
    # x is blocked iff 2x in sums OR exists a in S with x+a in sums
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if x is blocked
        if 2 * x in sums:
            continue  # blocked by diagonal
        # Check if any x+a is in sums
        blocked = False
        for a in S:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            return False  # x can be added, not maximal
    return True

def _sidon_and_maximal_fast(S, N):
    if not S:
        return False
    sums, ok = _compute_sums(S)
    if not ok:
        return False
    return _is_maximal_fast(S, N, sums)

def _greedy_sidon(randomize=False, seed=None):
    """Build a Sidon set greedily. If randomize, shuffle candidate order."""
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

def _try_minimize_by_removal(S, N):
    """Try to remove elements while keeping Sidon + maximal. Returns minimal size set found."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        # Try removing each element; compute sums once per candidate
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            sums, ok = _compute_sums(candidate)
            if ok and _is_maximal_fast(candidate, N, sums):
                S = candidate
                improved = True
                break
    return S

def _local_search_swap(S, N):
    """Try swapping elements to reduce size while maintaining maximality."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        for i in range(len(S)):
            for j in range(len(S)):
                if i == j:
                    continue
                # Try removing S[i] and adding S[j] (if S[j] not already in S)
                if S[j] in S:
                    continue
                candidate = S[:i] + S[i+1:] + [S[j]]
                candidate.sort()
                sums, ok = _compute_sums(candidate)
                if ok and _is_maximal_fast(candidate, N, sums) and len(candidate) < len(S):
                    S = candidate
                    improved = True
                    break
            if improved:
                break
    return S

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1
    
    # Phase 1: Try several random greedy constructions, then minimize
    for seed in range(5000):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 3:  # leave 3s safety
            break
        S = _greedy_sidon(randomize=True, seed=seed)
        if len(S) < best_size:
            # Try to minimize further
            S_min = _try_minimize_by_removal(S, N)
            if _sidon_and_maximal_fast(S_min, N) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min
            if elapsed > _DEADLINE - 8:
                break
    
    # If nothing found, use deterministic greedy
    if best_set is None:
        S = _greedy_sidon(randomize=False)
        S_min = _try_minimize_by_removal(S, N)
        if _sidon_and_maximal_fast(S_min, N):
            best_set = S_min
        else:
            best_set = S
    
    return best_set
# EVOLVE-BLOCK-END
