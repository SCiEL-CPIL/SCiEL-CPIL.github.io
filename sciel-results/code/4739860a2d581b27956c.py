"""Small inclusion-maximal Sidon sets in [1,256]."""
import random
import time
from itertools import combinations

N = 256

def _all_sums(S):
    """Return set of all unordered pair sums a+b with a<=b for S."""
    sums = set()
    n = len(S)
    for i in range(n):
        a = S[i]
        sums.add(2 * a)
        for j in range(i + 1, n):
            sums.add(a + S[j])
    return sums

def _is_sidon(S):
    """Check if S is a Sidon set (all unordered sums distinct)."""
    sums = set()
    n = len(S)
    for i in range(n):
        a = S[i]
        if 2 * a in sums:
            return False
        sums.add(2 * a)
        for j in range(i + 1, n):
            s = a + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(S):
    """Check if S is inclusion-maximal Sidon set in [1,N]."""
    S_set = set(S)
    sums = _all_sums(S)
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if adding x would create a collision
        conflict = False
        for a in S:
            if (x + a) in sums:
                conflict = True
                break
        if not conflict and (2 * x) in sums:
            conflict = True
        if not conflict:
            return False
    return True

def _greedy_construct(seed):
    """Greedy construction with random ordering."""
    rng = random.Random(seed)
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    selected = []
    sums = set()
    
    for x in candidates:
        # Check if x can be added
        ok = True
        for a in selected:
            if (x + a) in sums:
                ok = False
                break
        if ok and (2 * x) in sums:
            ok = False
        if not ok:
            continue
        # Add x
        selected.append(x)
        for a in selected[:-1]:  # don't double-add 2x, but do add x+a for previous
            sums.add(x + a)
        sums.add(2 * x)
    
    # Now try to remove elements to minimize size while staying maximal
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            if len(selected) <= 1:
                break
            # Try removing selected[i]
            candidate = selected[:i] + selected[i+1:]
            if _is_sidon(candidate) and _is_maximal(candidate):
                selected = candidate
                changed = True
                break
    
    return sorted(selected)

def _local_search(S, seed, max_iters=5000):
    """Local search to minimize size of maximal Sidon set."""
    rng = random.Random(seed)
    S = list(S)
    best = S[:]
    best_len = len(S)
    
    for it in range(max_iters):
        if len(S) >= best_len:
            continue
        # Try removing a random element
        idx = rng.randint(0, len(S) - 1)
        candidate = S[:idx] + S[idx+1:]
        if _is_sidon(candidate) and _is_maximal(candidate):
            S = candidate
            if len(S) < best_len:
                best_len = len(S)
                best = S[:]
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s
    
    best = None
    best_len = N + 1
    
    # First, get a baseline greedy solution
    baseline = _greedy_construct(42)
    best = baseline
    best_len = len(baseline)
    
    # Try multiple random greedy constructions
    seed = 0
    while time.time() < deadline - 5.0:
        candidate = _greedy_construct(seed)
        if len(candidate) < best_len:
            best_len = len(candidate)
            best = candidate
        seed += 1
    
    # Local search on the best solution found
    if time.time() < deadline - 2.0:
        improved = _local_search(best, seed=123, max_iters=10000)
        if len(improved) < best_len:
            best = improved
            best_len = len(best)
    
    # More local search iterations if time permits
    while time.time() < deadline - 1.0:
        candidate = _local_search(best, seed=random.randint(1, 10**6), max_iters=5000)
        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
    
    return sorted(best)

N = 256
