"""Rewritten program: bitset-based greedy + randomized restarts for max subset-sum-free set."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 512
MAX_TIME = 85.0  # safety margin under 90s

def _build_conflict_bitsets():
    """
    For each element a in 1..512, precompute a bitset of all elements b such that
    there exists a subset S of {1..512} \\ {a, b} with |S| >= 1 and
    sum(S) + b == (|S|+1) * a.
    
    Equivalently: a is the mean of {b} ∪ S, i.e., sum(S) + b = (|S|+1)*a.
    
    We use DP over sums to check for each a and potential b whether such S exists.
    Actually, a more direct approach: for each a, we want to know which b values
    create a conflict. A conflict exists if there is a subset T of OTHERS (not containing a)
    with |T| >= 2 and a is the mean of T. That is sum(T) = |T| * a.
    
    So for each a, we need: does there exist a subset T ⊆ [1..512] \\ {a}, |T| >= 2,
    such that sum(T) = |T| * a?
    
    For the greedy construction, when we have a partial set P and want to add x,
    we need to check: for every a in P ∪ {x}, is a the mean of some subset of
    (P ∪ {x}) \\ {a} of size >= 2?
    
    This is expensive to check naively. Let's use a different strategy:
    Precompute for each pair (a, b) whether a can be the mean of a subset containing b
    and at least one other element from [1..512] \\ {a, b}.
    
    Actually, let's think about it differently. The constraint is:
    No element a in the set is the mean of any subset of at least 2 OTHER elements in the set.
    
    So for a set S, for each a ∈ S, for each subset T ⊆ S \\ {a} with |T| >= 2,
    we need sum(T) != |T| * a.
    
    For the greedy approach with bitsets:
    When considering adding x to current set P, we need to verify that:
    1. x is not the mean of any subset of P of size >= 2.
    2. For each a ∈ P, a is not the mean of any subset of (P \\ {a}) ∪ {x} of size >= 2.
       This means: a is not the mean of any subset of (P \\ {a}) of size >= 2 (already checked),
       AND a is not the mean of any subset of (P \\ {a}) ∪ {x} of size >= 2 that includes x.
       The latter means: there is no subset U ⊆ P \\ {a} with |U| >= 1 such that
       sum(U) + x = (|U|+1) * a, i.e., sum(U) = (|U|+1)*a - x.
    
    This is still complex. Let me use a simpler but effective approach:
    For each candidate x and current set P, just do a DP check for each element in P ∪ {x}.
    
    But that's slow. Let's use the randomized greedy with a fast check.
    
    Alternative: Use the fact that if a is the mean of a subset T of size k, then
    sum(T) = k*a. For small k (2,3,4,...), we can check specific cases.
    
    For k=2: a is mean of {b,c} iff b+c=2a, i.e., b and c are symmetric around a.
    For k=3: a is mean of {b,c,d} iff b+c+d=3a.
    etc.
    
    The most common conflicts are for small k. Let's precompute for each a the set of
    "forbidden" pairs/triples/etc.
    
    Actually, let's just do a direct DP-based check for each candidate addition,
    but optimize it with numpy bitsets.
    """
    pass

def _is_valid_set(S):
    """Check if set S is valid: no element is the mean of any subset of >=2 other elements."""
    S_list = sorted(S)
    n = len(S_list)
    if n < 3:
        return True
    
    # For each element a in S, check if a is the mean of any subset of S\{a} of size >= 2
    for idx_a, a in enumerate(S_list):
        others = [x for i, x in enumerate(S_list) if i != idx_a]
        m = len(others)
        if m < 2:
            continue
        
        # DP: dp[s][k] = True if there's a subset of others of size k with sum s
        # We need to check if dp[k*a][k] is True for any k >= 2
        max_sum = sum(others)
        # Use a 2D boolean array: dp[k][s]
        # But max_sum can be large. Let's use a set-based approach or numpy.
        
        # For efficiency, use a list of sets: reachable[k] = set of sums achievable with k elements
        reachable = [set() for _ in range(m + 1)]
        reachable[0].add(0)
        
        for x in others:
            # Iterate in reverse to avoid using x multiple times
            for k in range(m - 1, 0, -1):
                if reachable[k - 1]:
                    for s in list(reachable[k - 1]):
                        new_s = s + x
                        if new_s <= max_sum:
                            reachable[k].add(new_s)
        
        # Check if any k >= 2 has k*a in reachable[k]
        for k in range(2, m + 1):
            if k * a in reachable[k]:
                return False
    
    return True

def _greedy_with_random_order(elements, rng):
    """Greedy construction using a random order of elements."""
    selected = []
    remaining = list(elements)
    rng.shuffle(remaining)
    
    for x in remaining:
        candidate = selected + [x]
        if _is_valid_set(candidate):
            selected.append(x)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Start with a base set
    best_set = []
    
    # Try multiple random orders
    rng = random.Random(42)
    elements = list(range(1, N + 1))
    
    # First, try a few deterministic heuristics
    # Heuristic 1: Powers of 2 (known to avoid 3-term arithmetic progressions, but we need stronger)
    # Actually, powers of 2: no element is the mean of any subset of others?
    # Let's just use random restarts.
    
    attempts = 0
    while time.time() < deadline - 5.0:  # Leave 5s safety margin
        attempts += 1
        s = _greedy_with_random_order(elements, rng)
        if len(s) > len(best_set):
            best_set = s
        
        # Also try a local improvement: try to add elements not in best_set
        if len(best_set) > 0:
            current = list(best_set)
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for x in elements:
                    if x not in current:
                        candidate = current + [x]
                        if _is_valid_set(candidate):
                            current = candidate
                            improved = True
                            break
                if len(current) > len(best_set):
                    best_set = current
    
    return best_set
# EVOLVE-BLOCK-END
