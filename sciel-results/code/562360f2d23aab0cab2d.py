"""Construct a finite set U of nonnegative integers containing 0 to maximize
the sumset/difference-set exponent lower bound.

Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)

Strategy:
  - Build U via a greedy construction that favors elements with low pairwise
    collision in U+U while keeping U-U large.
  - Use a base-b representation with carefully chosen digits to control
    additive structure.
  - Start from a known good seed and greedily add/remove elements to improve score.
"""

import math
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))
    if len(U) < 2:
        return 0.0
    maxU = max(U)
    if maxU == 0:
        return 0.0
    # Compute |U+U| and |U-U|
    # For efficiency, use set operations
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    n_sum = len(sumset)
    n_diff = len(diffset)
    if n_sum == 0:
        return 0.0
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(n_diff / n_sum) / denom


def _greedy_build(max_elem, n_target, deadline):
    """
    Greedily build a set U starting from {0}, adding elements that maximize
    the marginal improvement in score.
    """
    U = {0}
    # We'll try to build up to n_target elements, each <= max_elem
    # Candidate pool: all integers 0..max_elem
    # To be efficient, we only consider a subset of candidates
    
    # Precompute: for each candidate c, we want to know how it affects |U+U| and |U-U|
    # Adding c to U: new sumset adds {c+u for u in U}, new diffset adds {c-u, u-c for u in U}
    
    # We'll use a heuristic: prefer elements with large "spread" (far from existing elements)
    # but also check actual score improvement for promising candidates.
    
    # Start with a structured base
    # Use a geometric progression base for good difference set properties
    base = [0]
    step = 1
    while step <= max_elem:
        base.append(step)
        step *= 3  # base-3 gives good difference set properties
    
    # Filter to <= max_elem
    base = [b for b in base if b <= max_elem]
    
    # Now greedily add elements
    U_set = set(base)
    
    # Candidate elements: try all integers 1..max_elem not already in U
    # But that's too many. Use a smarter approach:
    # Try elements that are "far" from existing ones in a structured way.
    
    # Generate candidates: multiples of various steps
    candidates = set()
    for step in range(1, max_elem + 1, max(1, max_elem // 500)):
        for k in range(1, max_elem // step + 1):
            c = k * step
            if c <= max_elem and c not in U_set:
                candidates.add(c)
    
    # Also add some random candidates
    for _ in range(200):
        c = random.randint(1, max_elem)
        candidates.add(c)
    
    # Sort candidates by heuristic: prefer larger elements (they increase max(U) and diffset)
    candidates = sorted(candidates, key=lambda c: -c)
    
    # Greedily add elements
    best_score = _score(list(U_set))
    
    for c in candidates:
        if time.time() > deadline:
            break
        if len(U_set) >= n_target:
            break
        # Test adding c
        new_U = U_set | {c}
        new_score = _score(list(new_U))
        if new_score > best_score:
            U_set = new_U
            best_score = new_score
    
    return list(U_set), best_score


def _local_search(U, deadline, max_iter=200):
    """
    Local search: try replacing one element with another to improve score.
    """
    U_set = set(U)
    best_score = _score(list(U_set))
    maxU = max(U_set)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        
        # Try removing an element (except 0)
        for u in list(U_set):
            if u == 0:
                continue
            new_U = U_set - {u}
            if len(new_U) < 2:
                continue
            new_score = _score(list(new_U))
            if new_score > best_score + 1e-12:
                U_set = new_U
                best_score = new_score
                improved = True
                break
        
        if improved:
            continue
        
        # Try adding an element
        # Candidates: random + structured
        cands = set()
        for _ in range(50):
            c = random.randint(1, maxU)
            cands.add(c)
        for step in range(1, maxU + 1, max(1, maxU // 100)):
            for k in range(1, maxU // step + 1):
                c = k * step
                if c <= maxU:
                    cands.add(c)
        
        for c in cands:
            if c in U_set:
                continue
            new_U = U_set | {c}
            new_score = _score(list(new_U))
            if new_score > best_score + 1e-12:
                U_set = new_U
                best_score = new_score
                improved = True
                maxU = max(U_set)
                break
        
        if not improved:
            # Try swapping: remove one, add another
            for u in list(U_set):
                if u == 0:
                    continue
                for c in list(cands):
                    if c in U_set or c == u:
                        continue
                    new_U = (U_set - {u}) | {c}
                    new_score = _score(list(new_U))
                    if new_score > best_score + 1e-12:
                        U_set = new_U
                        best_score = new_score
                        improved = True
                        maxU = max(U_set)
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return list(U_set), best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    # Try several constructions with different parameters
    configs = [
        (100, 20),
        (200, 30),
        (500, 40),
        (1000, 50),
        (2000, 60),
        (5000, 80),
        (10000, 100),
        (20000, 120),
    ]
    
    for max_elem, n_target in configs:
        if time.time() > deadline:
            break
        
        # Greedy build
        U, sc = _greedy_build(max_elem, n_target, deadline)
        if sc > best_score:
            best_U = U
            best_score = sc
        
        # Local search refinement
        if time.time() < deadline - 30:
            U, sc = _local_search(U, deadline, max_iter=100)
            if sc > best_score:
                best_U = U
                best_score = sc
    
    # Final polish: try more local search iterations
    if time.time() < deadline - 5:
        U, sc = _local_search(best_U, deadline, max_iter=500)
        if sc > best_score:
            best_U = U
            best_score = sc
    
    return sorted(best_U)

# EVOLVE-BLOCK-END
