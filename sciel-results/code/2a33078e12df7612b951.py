"""
B2[2] set construction: maximize size of subset of {1..1000} where every sum a+b (a<=b)
has at most two representations.

Strategy:
1. Fast greedy to get a good baseline set.
2. Local search: try to add elements, and try swaps (remove one, add one or two) to improve.
3. Use efficient sum-count tracking with a dictionary.
"""
import time
import random
from collections import defaultdict

def _sum_counts_from_set(S):
    """Compute sum counts for a set S (list of distinct ints)."""
    counts = defaultdict(int)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i, n):
            b = S[j]
            counts[a + b] += 1
    return counts

def _can_add(S_sorted, counts, x):
    """Check if x can be added to S_sorted without violating B2[2]."""
    # S_sorted is sorted; find insertion point
    # New sums: x + s for s in S_sorted, and 2*x
    # We need each new sum to have count <= 2 in the updated counts
    # Current count of sum s is counts[s]
    # After adding x, count of (x+s) becomes counts[x+s] + 1, must be <= 2
    # And count of 2*x becomes counts[2*x] + 1, must be <= 2
    # But also, if x is already in S, we skip.
    for s in S_sorted:
        if counts[x + s] >= 2:
            return False
    if counts[2 * x] >= 2:
        return False
    return True

def _greedy_build():
    """Greedy: add x from 1..1000 if it doesn't violate B2[2]."""
    chosen = []
    counts = defaultdict(int)
    for x in range(1, 1001):
        # Check if x can be added
        ok = True
        for s in chosen:
            if counts[x + s] >= 2:
                ok = False
                break
        if ok and counts[2 * x] >= 2:
            ok = False
        if ok:
            chosen.append(x)
            for s in chosen:
                counts[x + s] += 1
            counts[2 * x] += 1
    return chosen, counts

def _local_search(S, counts, time_limit=100.0):
    """
    Local search to improve the set size.
    S is a list, counts is the sum-count dict.
    Try:
      1. Add any x not in S.
      2. Swap: remove one element, add one or two.
    """
    S_set = set(S)
    S_sorted = sorted(S)
    start_time = time.time()
    
    # Precompute which x are not in S
    candidates = [x for x in range(1, 1001) if x not in S_set]
    
    best_size = len(S)
    best_S = list(S)
    best_counts = dict(counts)
    
    # Phase 1: Try to add single elements
    improved = True
    while improved:
        improved = False
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            if _can_add(S_sorted, counts, x):
                # Add x
                S.append(x)
                S_set.add(x)
                S_sorted.append(x)
                S_sorted.sort()
                for s in S_sorted:
                    counts[x + s] += 1
                counts[2 * x] += 1
                candidates.remove(x)
                improved = True
                best_size = len(S)
                best_S = list(S)
                best_counts = dict(counts)
        if time.time() - start_time > time_limit:
            break
    
    # Phase 2: Randomized swaps
    random.seed(42)
    attempts = 0
    max_attempts = 50000
    while time.time() - start_time < time_limit and attempts < max_attempts:
        attempts += 1
        # Try to remove a random element and add a random candidate
        if not S:
            break
        idx = random.randint(0, len(S) - 1)
        remove_x = S[idx]
        
        # Try to add a random candidate
        if not candidates:
            # No candidates left, try to remove and see if we can add something
            # Actually if no candidates, we can't add, so skip
            continue
        
        add_x = random.choice(candidates)
        
        # Check if removing remove_x and adding add_x is valid
        # This is complex; let's just try a simple approach:
        # Temporarily remove remove_x, check if add_x can be added, then decide.
        
        # Save state
        old_counts = dict(counts)
        old_S = list(S)
        old_S_set = set(S_set)
        old_S_sorted = list(S_sorted)
        
        # Remove remove_x
        S.remove(remove_x)
        S_set.remove(remove_x)
        S_sorted.remove(remove_x)
        for s in S_sorted:
            counts[remove_x + s] -= 1
        counts[2 * remove_x] -= 1
        
        # Try to add add_x
        if _can_add(S_sorted, counts, add_x):
            S.append(add_x)
            S_set.add(add_x)
            S_sorted.append(add_x)
            S_sorted.sort()
            for s in S_sorted:
                counts[add_x + s] += 1
            counts[2 * add_x] += 1
            candidates.remove(add_x)
            # Success: size same, but we've changed the set
            # This doesn't improve size, but might enable future additions
        else:
            # Restore
            S = old_S
            S_set = old_S_set
            S_sorted = old_S_sorted
            counts = old_counts
    
    # Phase 3: Try to add again after swaps
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            if _can_add(S_sorted, counts, x):
                S.append(x)
                S_set.add(x)
                S_sorted.append(x)
                S_sorted.sort()
                for s in S_sorted:
                    counts[x + s] += 1
                counts[2 * x] += 1
                candidates.remove(x)
                improved = True
                best_size = len(S)
                best_S = list(S)
                best_counts = dict(counts)
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Phase 1: Greedy build
    S, counts = _greedy_build()
    best_S = list(S)
    best_size = len(S)
    
    # Phase 2: Local search
    if time.time() < deadline - 20:
        S2 = _local_search(S, counts, time_limit=deadline - time.time() - 5)
        if len(S2) > best_size:
            best_S = S2
            best_size = len(S2)
    
    # Verify and return
    return best_S
