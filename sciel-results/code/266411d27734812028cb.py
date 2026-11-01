import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _score_small(U):
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

def _is_golomb(U):
    """Check if U is a Golomb ruler (all pairwise differences are unique)."""
    diffs = set()
    n = len(U)
    for i in range(n):
        for j in range(i + 1, n):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

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

def _local_search(U, time_budget=10.0):
    """
    Enhanced local search with simulated annealing, larger move sets,
    and occasional add/remove moves to adjust set size.
    """
    U = sorted(list(U))
    if 0 not in U:
        U.insert(0, 0)
    best_U = list(U)
    best_s = _score_small(best_U)
    start = time.time()
    rng = random.Random(42)
    n = len(best_U)
    
    # Simulated annealing parameters
    T = 0.5  # Lower initial temperature for more directed search
    T_min = 0.0001
    cooling_rate = 0.99
    
    current_U = list(best_U)
    current_s = best_s
    
    # Precompute common deltas
    small_deltas = [-3, -2, -1, 1, 2, 3]
    medium_deltas = [-8, -7, -6, -5, -4, 4, 5, 6, 7, 8]
    large_deltas = [-15, -12, -10, -8, 8, 10, 12, 15]
    
    while time.time() - start < time_budget and _remaining() > 2.0:
        improved = False
        indices = list(range(1, n))
        rng.shuffle(indices)
        
        for i in indices:
            if _remaining() < 2.0:
                break
            orig = current_U[i]
            
            # Try small, medium, and large moves with different probabilities
            move_type = rng.random()
            if move_type < 0.5:
                deltas = small_deltas
            elif move_type < 0.85:
                deltas = medium_deltas
            else:
                deltas = large_deltas
            
            for delta in deltas:
                new_val = orig + delta
                if new_val < 0:
                    continue
                if new_val in current_U:
                    continue
                new_U = list(current_U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                
                if _is_golomb(new_U):
                    s = _score_small(new_U)
                    delta_s = s - current_s
                    if delta_s > 0:
                        current_s = s
                        current_U = new_U
                        if s > best_s:
                            best_s = s
                            best_U = new_U
                        improved = True
                        break
                    else:
                        if T > T_min and rng.random() < math.exp(delta_s / T):
                            current_s = s
                            current_U = new_U
                            improved = True
                            break
        
        # Occasionally try add/remove moves
        if rng.random() < 0.15 and _remaining() > 2.0:
            if len(current_U) > 3 and rng.random() < 0.4:
                # Try removing a non-zero element
                idx = rng.randint(1, len(current_U) - 1)
                test_U = [x for j, x in enumerate(current_U) if j != idx]
                if _is_golomb(test_U):
                    s = _score_small(test_U)
                    if s > current_s or (T > T_min and rng.random() < math.exp((s - current_s) / T)):
                        current_U = test_U
                        current_s = s
                        if s > best_s:
                            best_s = s
                            best_U = list(current_U)
                        improved = True
            elif len(current_U) < 15:
                # Try adding a new element
                max_val = max(current_U) if current_U else 0
                for add_val in [max_val + 1, max_val + 2, max_val + 3, max_val + 5, max_val + 7, max_val + 10]:
                    if add_val <= 0:
                        continue
                    new_U = sorted(current_U + [add_val])
                    if _is_golomb(new_U):
                        s = _score_small(new_U)
                        if s > current_s or (T > T_min and rng.random() < math.exp((s - current_s) / T)):
                            current_U = new_U
                            current_s = s
                            if s > best_s:
                                best_s = s
                                best_U = list(current_U)
                            improved = True
                            break
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        if not improved and rng.random() < 0.2:
            # Random perturbation to escape local minima
            idx = rng.randint(1, max(1, len(current_U) - 1))
            new_val = max(1, current_U[idx] + rng.randint(-10, 10))
            if new_val not in current_U:
                new_U = list(current_U)
                new_U[idx] = new_val
                new_U = sorted(new_U)
                if _is_golomb(new_U):
                    s = _score_small(new_U)
                    if s > current_s:
                        current_U = new_U
                        current_s = s
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
            
    # 2. Add random variations of small Golomb rulers
    for n in range(3, 10):
        base = _gen_golomb_optimal(n)
        if base:
            for _ in range(8):
                perturbed = list(base)
                if len(perturbed) > 2:
                    i, j = random.sample(range(1, len(perturbed)), 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                    perturbed = sorted(perturbed)
                    if _is_golomb(perturbed):
                        candidates.append(perturbed)

    # 3. Generate some random Golomb rulers by greedy construction
    for _ in range(5):
        g = [0]
        for _ in range(random.randint(5, 10)):
            # Try to add a new element
            found = False
            for attempt in range(200):
                new_val = random.randint(1, 100)
                if new_val not in g:
                    test_g = sorted(g + [new_val])
                    if _is_golomb(test_g):
                        g = test_g
                        found = True
                        break
            if not found:
                break
        if len(g) >= 4:
            candidates.append(g)

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(list(c))

    # Score all initial candidates
    scored = [(_score_small(s), s) for s in unique]
    scored.sort(key=lambda x: -x[0])

    if not scored:
        return [0, 1]

    best_U = scored[0][1]
    best_s = scored[0][0]

    # Local search on top candidates
    top_candidates = [s for _, s in scored[:12]]
    
    per_set_time = 4.0
    
    for s in top_candidates:
        if _remaining() < 5.0:
            break
        result, rs = _local_search(s, time_budget=per_set_time)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Final polish on the best found with extended time budget
    if _remaining() > 10.0:
        polished, ps = _local_search(best_U, time_budget=min(25.0, _remaining() - 2.0))
        if ps > best_s:
            best_s = ps
            best_U = polished
    
    # Additional refinement: try adding/removing elements to adjust set size
    if _remaining() > 5.0:
        max_val = max(best_U) if best_U else 0
        for add_val in [max_val + 1, max_val + 2, max_val + 3, max_val + 5, max_val + 8, max_val + 12]:
            if add_val <= 0:
                continue
            new_U = sorted(best_U + [add_val])
            if _is_golomb(new_U):
                s = _score_small(new_U)
                if s > best_s:
                    best_s = s
                    best_U = new_U
        
        if len(best_U) > 3:
            test_U = best_U[:-1]
            if _is_golomb(test_U):
                s = _score_small(test_U)
                if s > best_s:
                    best_s = s
                    best_U = test_U

    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
