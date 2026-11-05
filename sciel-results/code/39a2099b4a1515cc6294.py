"""B_3[1] set construction: maximize cardinality with unique unordered triple sums.

Strategy:
  1. Fast numpy-based greedy to get a strong baseline quickly.
  2. Multi-phase randomized search with different candidate orderings.
  3. Local search: swap-based improvement to escape greedy local optima.
"""
import time
import random
import sys

import numpy as np

def _greedy_numpy(candidates, deadline):
    """
    Greedy insertion using numpy for fast conflict checking.
    candidates: list or array of candidate values to try (in order).
    Returns the constructed set as a Python list.
    """
    S = []
    # existing_sums: a set of all triple sums so far
    existing_sums = set()
    
    for x in candidates:
        if deadline and time.time() > deadline:
            break
        
        valid = True
        n = len(S)
        
        # Check (x, x, x)
        s = 3 * x
        if s in existing_sums:
            valid = False
        else:
            # Check (x, x, y) for y in S
            for y in S:
                s = 2 * x + y
                if s in existing_sums:
                    valid = False
                    break
                s = x + 2 * y
                if s in existing_sums:
                    valid = False
                    break
            if valid:
                # Check (x, y, z) for y <= z in S
                for i in range(n):
                    for j in range(i, n):
                        s = x + S[i] + S[j]
                        if s in existing_sums:
                            valid = False
                            break
                    if not valid:
                        break
        
        if valid:
            S.append(x)
            # Add new sums
            existing_sums.add(3 * x)
            for y in S:
                existing_sums.add(2 * x + y)
                existing_sums.add(x + 2 * y)
            for i in range(n):
                for j in range(i, n):
                    existing_sums.add(x + S[i] + S[j])
    
    return S

def _check_conflict_fast(x, S, existing_sums):
    """
    Check if adding x to S creates any conflict.
    Returns True if valid.
    """
    n = len(S)
    
    # Check (x, x, x)
    if 3 * x in existing_sums:
        return False
    
    # Check (x, x, y) for y in S
    for y in S:
        if 2 * x + y in existing_sums:
            return False
        if x + 2 * y in existing_sums:
            return False
    
    # Check (x, y, z) for y <= z in S
    for i in range(n):
        for j in range(i, n):
            if x + S[i] + S[j] in existing_sums:
                return False
    
    return True

def _add_element(x, S, existing_sums):
    """Add x to S and update existing_sums in place."""
    S.append(x)
    existing_sums.add(3 * x)
    n = len(S) - 1  # index of x is n
    for i in range(n):
        existing_sums.add(2 * x + S[i])
        existing_sums.add(x + 2 * S[i])
        for j in range(i, n):
            existing_sums.add(x + S[i] + S[j])

def _remove_element(idx, S, existing_sums):
    """Remove S[idx] from S and rebuild existing_sums (expensive but correct)."""
    x = S[idx]
    S.pop(idx)
    # Rebuild existing_sums from scratch
    existing_sums.clear()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                existing_sums.add(S[i] + S[j] + S[k])

def _local_search_swap(S, existing_sums, deadline, rng, max_swaps=200):
    """
    Try to improve S by swapping elements: remove one, add another.
    This allows escaping greedy local optima.
    Returns improved (S, existing_sums) or the same if no improvement.
    """
    S = list(S)
    existing_sums = set(existing_sums)
    in_S = set(S)
    best_size = len(S)
    
    for _ in range(max_swaps):
        if deadline and time.time() > deadline:
            break
        
        # Try a random swap: remove a random element, try to add a random new one
        if len(S) < 2:
            break
        
        idx_remove = rng.randint(0, len(S) - 1)
        x_remove = S[idx_remove]
        
        # Try to find a replacement
        # Pick a random candidate not in S
        candidates = [x for x in range(1, 4097) if x not in in_S]
        if not candidates:
            break
        
        # Try a few random candidates
        tried = 0
        while tried < 10:
            if deadline and time.time() > deadline:
                break
            x_add = rng.choice(candidates)
            tried += 1
            
            # Temporarily remove x_remove
            _remove_element(idx_remove, S, existing_sums)
            in_S.discard(x_remove)
            
            if _check_conflict_fast(x_add, S, existing_sums):
                _add_element(x_add, S, existing_sums)
                in_S.add(x_add)
                # Size stays same, but we've diversified
                break
            else:
                # Restore
                _add_element(x_remove, S, existing_sums)
                in_S.add(x_remove)
    
    return S, existing_sums

def _greedy_with_backtrack(candidates, deadline, rng, backtrack_depth=3):
    """
    Greedy with limited backtracking: if stuck, try a few alternative choices.
    """
    S = []
    existing_sums = set()
    
    for x in candidates:
        if deadline and time.time() > deadline:
            break
        
        if _check_conflict_fast(x, S, existing_sums):
            S.append(x)
            _add_element(x, S, existing_sums)
    
    return S

def _multi_phase_greedy(deadline, rng):
    """
    Try multiple greedy passes with different orderings.
    """
    best_S = []
    
    # Phase 1: ascending
    S = _greedy_numpy(list(range(1, 4097)), deadline)
    if len(S) > len(best_S):
        best_S = S
    
    # Phase 2: descending
    if time.time() < deadline:
        S = _greedy_numpy(list(range(4096, 0, -1)), deadline)
        if len(S) > len(best_S):
            best_S = S
    
    # Phase 3: random shuffles
    for attempt in range(50):
        if time.time() > deadline:
            break
        candidates = list(range(1, 4097))
        rng.shuffle(candidates)
        S = _greedy_numpy(candidates, deadline)
        if len(S) > len(best_S):
            best_S = S
    
    # Phase 4: biased toward small numbers
    for attempt in range(30):
        if time.time() > deadline:
            break
        # Geometric distribution: P(x) ~ 1/x^alpha
        alpha = rng.uniform(0.5, 3.0)
        weights = np.array([1.0 / (i + 1) ** alpha for i in range(4096)])
        weights /= weights.sum()
        candidates = np.random.choice(4096, size=4096, replace=False, p=weights) + 1
        S = _greedy_numpy(candidates.tolist(), deadline)
        if len(S) > len(best_S):
            best_S = S
    
    return best_S

def _extend_greedily(S, existing_sums, deadline):
    """
    Try to extend S by adding elements from 1..4096 not in S.
    """
    S = list(S)
    existing_sums = set(existing_sums)
    in_S = set(S)
    
    # Try candidates in ascending order (smaller = fewer conflicts)
    for x in range(1, 4097):
        if deadline and time.time() > deadline:
            break
        if x in in_S:
            continue
        if _check_conflict_fast(x, S, existing_sums):
            S.append(x)
            _add_element(x, S, existing_sums)
            in_S.add(x)
    
    return S, existing_sums

def _local_search_full(S, existing_sums, deadline, rng, max_iters=500):
    """
    More aggressive local search: try to add elements, and if that fails,
    try removing one element and adding two others (net +1).
    """
    S = list(S)
    existing_sums = set(existing_sums)
    in_S = set(S)
    
    # First, try simple extension
    S, existing_sums = _extend_greedily(S, existing_sums, deadline)
    in_S = set(S)
    
    # Now try swap-based improvements
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Try to add a random element
        candidates = [x for x in range(1, 4097) if x not in in_S]
        if not candidates:
            break
        
        x_add = rng.choice(candidates)
        if _check_conflict_fast(x_add, S, existing_sums):
            S.append(x_add)
            _add_element(x_add, S, existing_sums)
            in_S.add(x_add)
        else:
            # Try a swap: remove a random element, try to add x_add
            if len(S) > 1:
                idx_remove = rng.randint(0, len(S) - 1)
                x_remove = S[idx_remove]
                
                _remove_element(idx_remove, S, existing_sums)
                in_S.discard(x_remove)
                
                if _check_conflict_fast(x_add, S, existing_sums):
                    _add_element(x_add, S, existing_sums)
                    in_S.add(x_add)
                else:
                    # Restore
                    _add_element(x_remove, S, existing_sums)
                    in_S.add(x_remove)
    
    return S, existing_sums

def _is_valid_triples(S):
    """Check if all unordered triple sums (with replacement) are unique."""
    n = len(S)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = S[i] + S[j] + S[k]
                if s in seen:
                    return False
                seen.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (90s limit)
    rng = random.Random(42)
    
    best_S = []
    
    # Phase 1: Multi-phase greedy to get a strong baseline
    best_S = _multi_phase_greedy(deadline, rng)
    
    # Phase 2: Extend the best set
    if time.time() < deadline:
        existing_sums = set()
        n = len(best_S)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    existing_sums.add(best_S[i] + best_S[j] + best_S[k])
        best_S, existing_sums = _extend_greedily(best_S, existing_sums, deadline)
    
    # Phase 3: Local search to improve further
    if time.time() < deadline:
        existing_sums = set()
        n = len(best_S)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    existing_sums.add(best_S[i] + best_S[j] + best_S[k])
        best_S, existing_sums = _local_search_full(best_S, existing_sums, deadline, rng)
    
    # Phase 4: More random restarts with local search
    no_improve = 0
    while time.time() < deadline and no_improve < 30:
        seed = rng.randint(0, 10**9)
        rng2 = random.Random(seed)
        
        # Random candidate ordering
        candidates = list(range(1, 4097))
        rng2.shuffle(candidates)
        S = _greedy_numpy(candidates, deadline)
        
        # Extend
        if time.time() < deadline:
            existing_sums = set()
            n = len(S)
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        existing_sums.add(S[i] + S[j] + S[k])
            S, existing_sums = _extend_greedily(S, existing_sums, deadline)
        
        # Local search
        if time.time() < deadline:
            S, existing_sums = _local_search_full(S, existing_sums, deadline, rng2)
        
        if len(S) > len(best_S):
            best_S = S
            no_improve = 0
        else:
            no_improve += 1
    
    # Final verification
    if not _is_valid_triples(best_S):
        # Fallback: rebuild from scratch with a simple greedy
        best_S = _greedy_numpy(list(range(1, 4097)), deadline)
    
    return best_S

if __name__ == "__main__":
    s = time.time()
    res = construct()
    print(f"Time: {time.time()-s:.2f}s, Size: {len(res)}")
    if _is_valid_triples(res):
        print("Valid")
    else:
        print("Invalid")
