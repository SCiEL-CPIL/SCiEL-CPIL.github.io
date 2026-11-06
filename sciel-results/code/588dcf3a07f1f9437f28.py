import time
import random
import numpy as np

_BASE = 51975
_DEADLINE = 115.0
_START = time.time()

def _divisors(n):
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

_DIVS = _divisors(_BASE)
_DIVS_SORTED = sorted(_DIVS)

def _coverage_mask(classes):
    """Return boolean array of covered positions."""
    covered = np.zeros(_BASE, dtype=bool)
    for m, r in classes:
        covered[r::_BASE//m+1 if r + (_BASE//m)*m < _BASE else None] = True  # wrong
    # Simpler:
    covered = np.zeros(_BASE, dtype=bool)
    for m, r in classes:
        idx = np.arange(r, _BASE, m)
        covered[idx] = True
    return covered

def _uncovered_count(classes):
    covered = _coverage_mask(classes)
    return int(_BASE - covered.sum())

def _greedy_build(order):
    """Greedy: process moduli in given order, pick best residue."""
    covered = np.zeros(_BASE, dtype=bool)
    classes = []
    for m in order:
        best_r = 0
        best_gain = -1
        for r in range(m):
            idx = np.arange(r, _BASE, m)
            gain = int((~covered[idx]).sum())
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        idx = np.arange(best_r, _BASE, m)
        covered[idx] = True
    return classes

def _local_search(classes, max_rounds=200):
    """Local search: for each modulus, find best residue given others fixed."""
    classes = [c[:] for c in classes]
    n = len(classes)
    
    for round_num in range(max_rounds):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for i in range(n):
            m = classes[i][0]
            # Compute coverage from all other classes
            other_covered = np.zeros(_BASE, dtype=bool)
            for j in range(n):
                if j == i:
                    continue
                mj, rj = classes[j]
                idx = np.arange(rj, _BASE, mj)
                other_covered[idx] = True
            # For each residue r, compute uncovered if we use (m, r)
            best_r = classes[i][1]
            best_uncovered = _BASE + 1
            for r in range(m):
                idx = np.arange(r, _BASE, m)
                new_covered = other_covered.copy()
                new_covered[idx] = True
                uncovered = int(_BASE - new_covered.sum())
                if uncovered < best_uncovered:
                    best_uncovered = uncovered
                    best_r = r
            if best_r != classes[i][1]:
                classes[i][1] = best_r
                improved = True
        if not improved:
            break
    return classes

def construct():
    best_classes = None
    best_score = _BASE + 1
    
    # Multiple restarts with different orderings
    rng = random.Random(42)
    orderings = [
        _DIVS_SORTED,
        sorted(_DIVS, reverse=True),
    ]
    for _ in range(5):
        orderings.append(rng.sample(_DIVS, len(_DIVS)))
    
    for order in orderings:
        if time.time() > _DEADLINE - 10:
            break
        classes = _greedy_build(order)
        score = _uncovered_count(classes)
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
        
        # Local search
        classes_ls = _local_search(classes)
        score_ls = _uncovered_count(classes_ls)
        if score_ls < best_score:
            best_score = score_ls
            best_classes = [c[:] for c in classes_ls]
    
    if best_classes is None:
        best_classes = _greedy_build(_DIVS_SORTED)
    
    return best_classes
