"""Construct a finite set U of nonnegative integers containing 0 to maximize
the sumset/difference-set exponent lower bound.

Score = 1 + log(|U-U| / |U+U|) / log(2*max(U) + 1)
"""
import time
import math
import random
import itertools

# Deadline guard: evaluator kills at 150s, use 100s to be safe
_START_TIME = time.time()
DEADLINE = 100.0  # seconds


def _score(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute sumset and difference set
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom


def _geometric_set(base, m):
    """Return U = {floor(base^k) for k=0..m} as a sorted list of unique nonneg ints."""
    U = set()
    val = 1  # base^0
    for k in range(m + 1):
        U.add(int(round(val)))
        val *= base
    # Ensure 0 is in the set
    U.add(0)
    # Remove any duplicates and sort
    return sorted(U)


def _evaluate_geometric_sweep():
    """Sweep over base b in [1.5, 4.0] and m in [2, 12] to find best geometric set."""
    best_score = -1.0
    best_set = None
    
    # Coarse sweep over base
    bases = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.5, 4.0]
    
    for b in bases:
        for m in range(2, 13):
            if time.time() - _START_TIME > DEADLINE - 5:
                return best_set, best_score
            U = _geometric_set(b, m)
            if len(U) < 2:
                continue
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Fine sweep around the best base found
    if best_set is not None and best_score > 0:
        # Refine around the best base
        for b in [1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75, 2.85, 2.95, 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8, 3.9]:
            for m in range(2, 13):
                if time.time() - _START_TIME > DEADLINE - 5:
                    return best_set, best_score
                U = _geometric_set(b, m)
                if len(U) < 2:
                    continue
                s = _score(U)
                if s > best_score:
                    best_score = s
                    best_set = U
    
    return best_set, best_score


def _evaluate_powers_of_two():
    """Evaluate powers of 2 sets for various n, return the best."""
    best_score = -1.0
    best_set = None
    for n in range(2, 20):
        U = [0] + [1 << i for i in range(n)]
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U
    return best_set, best_score


def _try_local_search(initial_set, time_limit=30.0):
    """Local search: try swapping or adding/removing elements."""
    best_set = list(initial_set)
    best_score = _score(best_set)
    deadline = time.time() + time_limit
    max_val = max(best_set) if best_set else 1
    max_val = max(max_val, 10)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding a new element
        candidates = set(range(1, max_val + 5))
        for c in range(max_val + 1, max_val + 20):
            candidates.add(c)
        
        for c in candidates:
            if c in best_set:
                continue
            new_set = sorted(best_set + [c])
            s = _score(new_set)
            if s > best_score + 1e-9:
                best_set = new_set
                best_score = s
                improved = True
                max_val = max(best_set)
        
        if time.time() >= deadline:
            break
        
        # Try removing an element (only if size > 2)
        if len(best_set) > 2:
            for i, elem in enumerate(best_set):
                if elem == 0:
                    continue
                new_set = best_set[:i] + best_set[i+1:]
                s = _score(new_set)
                if s > best_score + 1e-9:
                    best_set = new_set
                    best_score = s
                    improved = True
                    max_val = max(best_set) if best_set else 1
        
        if time.time() >= deadline:
            break
    
    return best_set, best_score


def _try_random_search(deadline):
    """Random local search with known good starting points."""
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Known good sets from inspirations
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 9, 11],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 7, 14],
        [0, 1, 3, 8, 13],
        [0, 2, 3, 7, 12],
        [0, 1, 4, 11, 13],
        [0, 1, 4, 9, 13],
    ]
    
    for c in candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_set = c
    
    random.seed(42)
    
    while time.time() < deadline - 5:
        n = random.randint(4, 8)
        max_val = random.randint(15, 60)
        
        other_elements = random.sample(range(1, max_val), n - 1)
        current = sorted([0] + other_elements)
        
        current_score = _score(current)
        if current_score < 0:
            continue
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(current)):
                if i == 0:
                    continue
                old_val = current[i]
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    if new_val <= 0 or new_val >= max_val:
                        continue
                    new_set = sorted(set([0] + [current[j] if j != i else new_val for j in range(len(current))]))
                    if len(new_set) != len(current):
                        continue
                    new_score = _score(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current[:]
                        break
        
        if time.time() < deadline - 5:
            if len(current) < 8:
                for new_val in range(1, max_val + 5):
                    if new_val not in current:
                        new_set = sorted(current + [new_val])
                        new_score = _score(new_set)
                        if new_score > current_score:
                            current = new_set
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_set = current[:]
                            break
            
            if len(current) > 4:
                for i in range(1, len(current)):
                    new_set = [current[j] for j in range(len(current)) if j != i]
                    new_score = _score(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current[:]
                        break
    
    return best_set, best_score


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_set = None
    best_score = -1.0
    
    # Phase 1: Evaluate geometric progression sets (new strategy)
    geo_set, geo_score = _evaluate_geometric_sweep()
    if geo_set is not None and geo_score > best_score:
        best_set = geo_set
        best_score = geo_score
    
    # Phase 2: Evaluate powers of 2
    if time.time() - _START_TIME < DEADLINE - 10:
        pow2_set, pow2_score = _evaluate_powers_of_two()
        if pow2_set is not None and pow2_score > best_score:
            best_set = pow2_set
            best_score = pow2_score
    
    # Phase 3: Local search from the best set found so far
    if best_set is not None and time.time() - _START_TIME < DEADLINE - 15:
        remaining = DEADLINE - (time.time() - _START_TIME) - 15
        improved_set, improved_score = _try_local_search(best_set, time_limit=min(remaining, 30))
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    
    # Phase 4: Random search if time permits
    if time.time() - _START_TIME < DEADLINE - 10:
        remaining = DEADLINE - (time.time() - _START_TIME) - 10
        rand_set, rand_score = _try_random_search(time.time() + min(remaining, 40))
        if rand_score > best_score:
            best_set = rand_set
            best_score = rand_score
    
    # Safety: ensure 0 is in the set
    if best_set is None:
        best_set = [0, 1, 4, 9, 11]
    if 0 not in best_set:
        best_set = [0] + sorted(best_set)
    
    return best_set
