"""
Construct a 32-element subset of {1..512} maximizing the number of distinct
differences d > 0 for which a, a+d, a+2d are all in the set.

Strategy:
- Precompute all 3-APs (a, a+d, a+2d) with a,a+d,a+2d in 1..512, grouped by d.
- Use greedy set-cover style heuristic with random restarts to select 32 elements
  maximizing distinct d covered.
- Polish with local search (swap moves).
- Time guard: return best found before deadline.
"""

import time
import random
from collections import defaultdict

def _score_set(S):
    """Return number of distinct d > 0 such that exists a with a,a+d,a+2d in S."""
    S_set = set(S)
    found = set()
    for a in S_set:
        for b in S_set:
            if b == a:
                continue
            d = b - a
            if d > 0 and (a + 2 * d) in S_set:
                found.add(d)
    return len(found)

def _all_aps_by_d(max_val=512):
    """
    Return dict: d -> list of (a, a+d, a+2d) for all valid 3-APs in 1..max_val.
    """
    aps_by_d = defaultdict(list)
    for a in range(1, max_val + 1):
        for d in range(1, (max_val - a) // 2 + 1):
            a2 = a + d
            a3 = a + 2 * d
            if a3 <= max_val:
                aps_by_d[d].append((a, a2, a3))
    return aps_by_d

def _greedy_construct(aps_by_d, n_elems=32, max_val=512, seed=0, time_limit=100.0):
    """
    Greedy set-cover heuristic with random restarts.
    Returns best set found.
    """
    rng = random.Random(seed)
    start_time = time.time()
    best_score = 0
    best_set = None
    
    # Precompute for each element x, which d's it participates in
    # And for each d, the list of APs
    # We'll use a different approach: start with random set, then greedily add/swap
    
    # Universe of d's: 1 to 255 (since max d such that a+2d <= 512 for some a>=1 => 2d <= 511 => d <= 255)
    max_d = (max_val - 1) // 2  # 255
    
    # For efficiency, precompute for each d the set of elements that appear in some AP with difference d
    # Actually, let's think differently.
    
    # Approach: Random restarts. Each restart:
    # 1. Start with empty set
    # 2. Greedily add elements that maximize new d's covered
    # 3. Polish with swaps
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization: pick n_elems random elements
        S = set(rng.sample(range(1, max_val + 1), n_elems))
        current_score = _score_set(S)
        
        # Greedy improvement: try swapping elements
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            S_list = list(S)
            for i in range(n_elems):
                if time.time() - start_time > time_limit:
                    break
                for x in range(1, max_val + 1):
                    if x in S:
                        continue
                    new_S = set(S)
                    new_S.remove(S_list[i])
                    new_S.add(x)
                    new_score = _score_set(new_S)
                    if new_score > current_score:
                        S = new_S
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(S)
    
    return list(best_set) if best_set else list(range(1, n_elems + 1))

def _polish_swap(S, aps_by_d, max_val=512, time_limit=50.0):
    """
    Local search: try all single swaps to improve score.
    """
    start_time = time.time()
    S = set(S)
    n_elems = len(S)
    current_score = _score_set(S)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        S_list = list(S)
        for i in range(n_elems):
            if time.time() - start_time > time_limit:
                break
            for x in range(1, max_val + 1):
                if x in S:
                    continue
                new_S = set(S)
                new_S.discard(S_list[i])
                new_S.add(x)
                new_score = _score_set(new_S)
                if new_score > current_score:
                    S = new_S
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
    
    return list(S), current_score

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 110s for safety
    
    n_elems = 32
    max_val = 512
    
    # Precompute APs by d
    aps_by_d = _all_aps_by_d(max_val)
    
    # Strategy 1: Greedy construction with multiple random seeds
    best_set = None
    best_score = 0
    
    # Try several random seeds
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        S = _greedy_construct(aps_by_d, n_elems, max_val, seed=seed, time_limit=min(10, deadline - time.time() - 20))
        score = _score_set(S)
        if score > best_score:
            best_score = score
            best_set = S
    
    # Polish the best set found
    if best_set and time.time() < deadline - 5:
        polished, polished_score = _polish_swap(best_set, aps_by_d, max_val, time_limit=min(50, deadline - time.time() - 5))
        if polished_score > best_score:
            best_score = polished_score
            best_set = polished
    
    # If we still have time, do more random restarts with polish
    if best_set and time.time() < deadline - 10:
        rng = random.Random(12345)
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            S = set(rng.sample(range(1, max_val + 1), n_elems))
            polished, score = _polish_swap(S, aps_by_d, max_val, time_limit=min(10, deadline - time.time() - 10))
            if score > best_score:
                best_score = score
                best_set = polished
    
    if best_set is None:
        best_set = list(range(1, n_elems + 1))
    
    return best_set

# Quick self-test
if __name__ == "__main__":
    S = construct()
    assert len(S) == 32
    assert len(set(S)) == 32
    assert all(1 <= x <= 512 for x in S)
    print("Score:", _score_set(S))
    print("Set:", sorted(S))
