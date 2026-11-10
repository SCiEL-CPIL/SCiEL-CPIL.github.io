"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 256
_DEADLINE = 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START

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

def _is_maximal(S, N, sums=None):
    """Check if S is inclusion-maximal Sidon in [1,N]. Reuse precomputed sums if given."""
    if sums is None:
        sums = _sidon_sums(S)
        if sums is None:
            return False
    S_set = set(S)
    for x in range(1, N + 1):
        if x in S_set:
            continue
        if (2 * x) in sums:
            continue
        blocked = False
        for a in S:
            if (x + a) in sums:
                blocked = True
                break
        if not blocked:
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

def _try_minimize_by_removal(S, N, deadline=None):
    """Try to remove elements while keeping Sidon + maximal. Returns minimal size set found."""
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

def _local_search_aggressive(S, N, deadline=None):
    """Aggressive local search: try removals first, then swaps to maintain maximality."""
    S = list(S)
    S_set = set(S)
    sums = _sidon_sums(S)
    if sums is None:
        return S
    
    # Phase 1: Try removing elements
    while True:
        if deadline is not None and time.time() > deadline:
            return S
        removed = False
        for i in range(len(S)):
            if deadline is not None and time.time() > deadline:
                return S
            candidate = S[:i] + S[i+1:]
            if _sidon_and_maximal(candidate, N):
                S = candidate
                S_set = set(S)
                sums = _sidon_sums(S)
                removed = True
                break
        if not removed:
            break
    
    # Phase 2: Try swaps to find smaller configurations
    for _ in range(20):
        if deadline is not None and time.time() > deadline:
            return S
        improved = False
        for i in range(len(S)):
            if deadline is not None and time.time() > deadline:
                return S
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                new_S = S[:i] + S[i+1:] + [x]
                if _sidon_and_maximal(new_S, N):
                    S = new_S
                    S_set = set(S)
                    sums = _sidon_sums(S)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return S

def _structured_initial(seed, pattern_type=0):
    """Generate structured initial Sidon sets with different patterns."""
    rng = random.Random(seed)
    
    if pattern_type == 0:
        # Sparse arithmetic progression
        step = rng.choice([3, 4, 5, 6, 7, 8])
        S = list(range(1, N + 1, step))
    elif pattern_type == 1:
        # Random subset with density control
        density = rng.uniform(0.08, 0.15)
        S = [x for x in range(1, N + 1) if rng.random() < density]
    elif pattern_type == 2:
        # Quadratic residues pattern
        S = [(i * i) % (N + 1) + 1 for i in range(1, 20)]
        S = list(set(S))
        S = [x for x in S if 1 <= x <= N]
    else:
        # Mixed: some structured, some random
        base = list(range(1, N + 1, 5))
        extra = [x for x in range(1, N + 1) if rng.random() < 0.1]
        S = list(set(base + extra))
    
    # Ensure it's Sidon by filtering
    filtered = []
    sums = set()
    for x in sorted(S):
        added = [x + a for a in filtered] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            filtered.append(x)
            sums.update(added)
    
    return filtered if filtered else _greedy_sidon(randomize=True, seed=seed)

def _conflict_guided_search(S, N, deadline=None):
    """Use conflict analysis to guide which elements to swap."""
    S = list(S)
    S_set = set(S)
    sums = _sidon_sums(S)
    if sums is None:
        return S
    
    # For each element in S, compute how many conflicts it creates
    conflict_scores = {}
    for i, a in enumerate(S):
        conflicts = 0
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if removing a would allow x to be added
            # Simplified: count how many x's are blocked by a
            for b in S:
                if b != a and (x + b) in sums and (2 * x) not in sums:
                    conflicts += 1
                    break
        conflict_scores[a] = conflicts
    
    # Try to swap high-conflict elements
    sorted_by_conflict = sorted(S, key=lambda x: conflict_scores.get(x, 0), reverse=True)
    
    for _ in range(10):
        if deadline is not None and time.time() > deadline:
            return S
        improved = False
        for a in sorted_by_conflict[:5]:  # Only try top 5 high-conflict elements
            if deadline is not None and time.time() > deadline:
                return S
            if a not in S:
                continue
            idx = S.index(a)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                new_S = S[:idx] + S[idx+1:] + [x]
                if _sidon_and_maximal(new_S, N):
                    S = new_S
                    S_set = set(S)
                    sums = _sidon_sums(S)
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
    
    # Phase 1: Try structured initializations
    for seed in range(500):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 10:
            break
        pattern_type = seed % 4
        S = _structured_initial(seed, pattern_type)
        if len(S) < best_size:
            S_min = _local_search_aggressive(S, N, deadline=_DEADLINE - 5)
            if _sidon_and_maximal(S_min, N) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min
                # Try conflict-guided improvement
                S_conf = _conflict_guided_search(S_min, N, deadline=_DEADLINE - 3)
                if _sidon_and_maximal(S_conf, N) and len(S_conf) < best_size:
                    best_size = len(S_conf)
                    best_set = S_conf
            if elapsed > _DEADLINE - 15:
                break
    
    # Phase 2: Random greedy with aggressive local search
    for seed in range(2000):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 5:
            break
        S = _greedy_sidon(randomize=True, seed=seed)
        if len(S) < best_size:
            S_min = _local_search_aggressive(S, N, deadline=_DEADLINE - 3)
            if _sidon_and_maximal(S_min, N) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min
            if elapsed > _DEADLINE - 8:
                break
    
    # Phase 3: Final polish on best solution
    if best_set is not None:
        elapsed = time.time() - start_time
        if elapsed < _DEADLINE - 2:
            S_final = _local_search_aggressive(best_set, N, deadline=_DEADLINE - 1)
            if _sidon_and_maximal(S_final, N) and len(S_final) <= best_size:
                best_set = S_final
    
    # Fallback if nothing found
    if best_set is None:
        S = _greedy_sidon(randomize=False)
        S_min = _local_search_aggressive(S, N, deadline=_DEADLINE - 1)
        if _sidon_and_maximal(S_min, N):
            best_set = S_min
        else:
            best_set = S
    
    return best_set
# EVOLVE-BLOCK-END
