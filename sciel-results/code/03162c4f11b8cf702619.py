"""Construct a finite set U of nonnegative integers containing 0 to maximize
the sumset/difference-set exponent lower bound.

Score = 1 + log(|U-U| / |U+U|) / log(2*max(U) + 1)
"""
import time
import itertools
import random
import math

# Deadline guard
_START_TIME = time.time()
DEADLINE = 100.0  # seconds, leave safety margin for the 150s evaluator limit


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


def _powers_of_two_set(n):
    """Return {0, 1, 2, 4, ..., 2^(n-1)}."""
    return [0] + [1 << i for i in range(n)]


def _evaluate_powers_of_two():
    """Evaluate powers of 2 sets for various n, return the best."""
    best_score = -1.0
    best_set = None
    for n in range(2, 20):
        U = _powers_of_two_set(n)
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
        # Also try some larger candidates
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


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with the best powers-of-2 construction
    best_set, best_score = _evaluate_powers_of_two()
    
    if time.time() - _START_TIME > DEADLINE - 5:
        return best_set
    
    # Try local search from the best powers-of-2 set
    try:
        improved_set, improved_score = _try_local_search(best_set, time_limit=DEADLINE - (time.time() - _START_TIME) - 10)
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    except Exception:
        pass
    
    # Try a few random perturbations if time permits
    if time.time() - _START_TIME < DEADLINE - 10:
        random.seed(42)
        base = list(best_set)
        for _ in range(20):
            if time.time() - _START_TIME > DEADLINE - 5:
                break
            # Randomly perturb: swap an element
            cand = list(base)
            if len(cand) > 2:
                idx = random.randint(1, len(cand) - 1)
                old_val = cand[idx]
                new_val = random.randint(1, max(cand) + 5)
                if new_val not in cand:
                    cand[idx] = new_val
                    cand = sorted(set(cand))
                    if 0 in cand:
                        s = _score(cand)
                        if s > best_score:
                            best_set = cand
                            best_score = s
            # Try adding a random element
            cand2 = list(base)
            new_val = random.randint(1, max(base) + 10)
            if new_val not in cand2:
                cand2 = sorted(cand2 + [new_val])
                s = _score(cand2)
                if s > best_score:
                    best_set = cand2
                    best_score = s
    
    # Safety: ensure 0 is in the set
    if 0 not in best_set:
        best_set = [0] + sorted(best_set)
    
    return best_set
