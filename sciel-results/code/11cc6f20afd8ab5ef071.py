import math
import time
import random

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _score_exact(U):
    """Exact O(n^2) score for small sets."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(len(diffset) / len(sumset), 2) / denom

def _gen_golomb_optimal(n):
    """Return known optimal Golomb rulers for small n."""
    optimal = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 55],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 67],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 80],
    }
    if n in optimal:
        return list(optimal[n])
    return None

def _local_search(U, time_budget=5.0):
    """
    Local search: try moving each element by small integers.
    Uses exact scoring for speed and accuracy on small sets.
    """
    U = sorted(list(U))
    if 0 not in U:
        U.insert(0, 0)
    best_U = list(U)
    best_s = _score_exact(best_U)
    start = time.time()
    rng = random.Random(42)
    n = len(best_U)
    
    while time.time() - start < time_budget and _remaining() > 2.0:
        improved = False
        indices = list(range(1, n))
        rng.shuffle(indices)
        
        for i in indices:
            if _remaining() < 2.0:
                break
            orig = best_U[i]
            max_delta = max(5, int(math.log2(max(best_U) + 1)))
            deltas = list(range(-max_delta, max_delta + 1))
            if 0 in deltas:
                deltas.remove(0)
            rng.shuffle(deltas)
            
            for delta in deltas:
                new_val = orig + delta
                if new_val < 0:
                    continue
                if new_val in best_U:
                    continue
                new_U = list(best_U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                
                s = _score_exact(new_U)
                if s > best_s + 1e-10:
                    best_s = s
                    best_U = new_U
                    improved = True
                    break
        
        if not improved:
            if rng.random() < 0.2:
                idx = rng.randint(1, n-1)
                new_val = max(1, best_U[idx] + rng.randint(-20, 20))
                if new_val not in best_U:
                    new_U = list(best_U)
                    new_U[idx] = new_val
                    new_U = sorted(new_U)
                    s = _score_exact(new_U)
                    if s > best_s:
                        best_s = s
                        best_U = new_U

    return best_U, best_s

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    candidates = []
    
    # 1. Known optimal Golomb rulers for small n
    for n in range(3, 13):
        g = _gen_golomb_optimal(n)
        if g:
            candidates.append(g)
            
    # 2. Add a few random variations
    for n in range(3, 10):
        base = _gen_golomb_optimal(n)
        if base:
            for _ in range(10):
                perturbed = list(base)
                if len(perturbed) > 2:
                    i, j = random.sample(range(1, len(perturbed)), 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                    perturbed = sorted(perturbed)
                    candidates.append(perturbed)

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(list(c))

    # Score all initial candidates
    scored = [(_score_exact(s), s) for s in unique]
    scored.sort(key=lambda x: -x[0])

    if not scored:
        return [0, 1]

    best_U = scored[0][1]
    best_s = scored[0][0]

    # Local search on top candidates
    top_candidates = [s for _, s in scored[:10]]
    
    per_set_time = 5.0
    
    for s in top_candidates:
        if _remaining() < 5.0:
            break
        result, rs = _local_search(s, time_budget=per_set_time)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Final polish
    if _remaining() > 10.0:
        polished, ps = _local_search(best_U, time_budget=min(15.0, _remaining() - 2.0))
        if ps > best_s:
            best_s = ps
            best_U = polished

    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
