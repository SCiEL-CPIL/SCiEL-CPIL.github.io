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
    Enhanced local search with simulated annealing and larger move sets.
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
    T = 1.0  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.995
    
    current_U = list(best_U)
    current_s = best_s
    
    while time.time() - start < time_budget and _remaining() > 2.0:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(1, n))
        rng.shuffle(indices)
        
        for i in indices:
            if _remaining() < 2.0:
                break
            orig = current_U[i]
            # Try larger range of shifts
            for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
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
                        # Accept improvement
                        current_s = s
                        current_U = new_U
                        if s > best_s:
                            best_s = s
                            best_U = new_U
                        improved = True
                        break
                    else:
                        # Simulated annealing: accept worse solutions with probability exp(delta_s/T)
                        if T > T_min and rng.random() < math.exp(delta_s / T):
                            current_s = s
                            current_U = new_U
                            improved = True
                            break
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        if not improved:
            # Random restart/perturbation if stuck
            if rng.random() < 0.3:
                idx = rng.randint(1, n-1)
                new_val = max(1, current_U[idx] + rng.randint(-8, 8))
                if new_val not in current_U:
                    new_U = list(current_U)
                    new_U[idx] = new_val
                    new_U = sorted(new_U)
                    if _is_golomb(new_U):
                        s = _score_small(new_U)
                        if s > current_s:
                            current_s = s
                            current_U = new_U
                            if s > best_s:
                                best_s = s
                                best_U = new_U

    return best_U, best_s

def _generate_structured_candidates(max_u_limit=100, n_limit=15):
    """
    Generate a diverse set of structured candidates from the inspiration.
    Includes APs, Geometric, Fibonacci, Blocks, and Random sets.
    """
    candidates = []
    
    # 1. Arithmetic Progressions
    for n in range(3, n_limit):
        for max_u in range(n, max_u_limit, 5):
            if (max_u) % (n - 1) == 0:
                step = max_u // (n - 1)
                if step >= 1:
                    ap = [i * step for i in range(n)]
                    candidates.append(ap)
    
    # 2. Powers of 2 / Geometric
    for base in [2, 3]:
        geo = [0, 1]
        while geo[-1] * base < max_u_limit:
            next_val = geo[-1] * base
            if next_val >= max_u_limit:
                break
            geo.append(next_val)
        if len(geo) > 2:
            candidates.append(geo)
            
    # 3. Fibonacci-like
    fib = [0, 1]
    while fib[-1] < max_u_limit:
        fib.append(fib[-1] + fib[-2])
    for n in range(3, min(n_limit, len(fib))):
        if fib[n-1] < max_u_limit:
            candidates.append(fib[:n])
            
    # 4. Blocks
    for block_size in [2, 3]:
        blocks = []
        for start in range(0, max_u_limit, block_size):
            for i in range(block_size):
                if start + i < max_u_limit:
                    blocks.append(start + i)
        if blocks and 0 in blocks and len(blocks) > 3:
            candidates.append(blocks)
            
    # 5. Random structured sets (deterministic seed)
    rng = random.Random(123)
    for _ in range(50):
        max_u = rng.randint(10, max_u_limit)
        n = rng.randint(4, min(10, max_u))
        inner = list(range(1, max_u))
        if len(inner) < n - 2:
            continue
        selected = rng.sample(inner, n - 2)
        cand = sorted([0] + selected + [max_u])
        # Only add if it's a Golomb ruler to keep the search space relevant?
        # Or just add raw? The local search handles non-Golomb?
        # The current program's local search checks _is_golomb.
        # If we pass non-Golomb, _is_golomb will fail on the first move usually, or the score will be low.
        # Let's filter for Golomb to be efficient, or just let the scorer handle it.
        # Actually, _score_small works for any set. _local_search restricts moves to Golomb.
        # So if we start with non-Golomb, we might be stuck if we can't easily become Golomb.
        # Better to start with Golomb or near-Golomb.
        if _is_golomb(cand):
            candidates.append(cand)

    return candidates

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
            
    # 2. Add a few random variations of small Golomb rulers
    for n in range(3, 10):
        base = _gen_golomb_optimal(n)
        if base:
            for _ in range(5):
                perturbed = list(base)
                if len(perturbed) > 2:
                    i, j = random.sample(range(1, len(perturbed)), 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                    perturbed = sorted(perturbed)
                    if _is_golomb(perturbed):
                        candidates.append(perturbed)

    # 3. Add structured candidates from the inspiration
    structured = _generate_structured_candidates(max_u_limit=150, n_limit=15)
    candidates.extend(structured)
    
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
    # We have more candidates now. Allocate time carefully.
    # Top 15 candidates.
    top_candidates = [s for _, s in scored[:15]]
    
    # Calculate time per candidate
    # We want to finish in ~100s. 
    # 15 candidates * 3s each = 45s. Safe.
    per_set_time = 3.0
    
    for s in top_candidates:
        if _remaining() < 4.0:
            break
        result, rs = _local_search(s, time_budget=per_set_time)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Final polish on the best found with extended time budget
    if _remaining() > 10.0:
        polished, ps = _local_search(best_U, time_budget=min(20.0, _remaining() - 2.0))
        if ps > best_s:
            best_s = ps
            best_U = polished
    
    # Additional refinement: try adding/removing elements to adjust set size
    if _remaining() > 5.0:
        # Try adding a new element near the end
        max_val = max(best_U) if best_U else 0
        for add_val in [max_val + 1, max_val + 2, max_val + 3, max_val + 5, max_val + 8, max_val + 10]:
            if add_val <= 0:
                continue
            new_U = sorted(best_U + [add_val])
            if _is_golomb(new_U):
                s = _score_small(new_U)
                if s > best_s:
                    best_s = s
                    best_U = new_U
        
        # Try removing the largest element if it doesn't help
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
