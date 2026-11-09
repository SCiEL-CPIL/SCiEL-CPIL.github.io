import time
import random
from collections import defaultdict

START_TIME = time.time()
DEADLINE = 62.0  # seconds, safety margin before 90s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def compute_sums_by_cardinality(A):
    """Compute set of achievable sums for each cardinality.
    Returns list where sums_by_card[k] = set of sums achievable with exactly k elements."""
    n = len(A)
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    
    for a in A:
        # Update in reverse to avoid using element multiple times
        for k in range(n - 1, -1, -1):
            if sums_by_card[k]:
                for s in sums_by_card[k]:
                    sums_by_card[k + 1].add(s + a)
    
    return sums_by_card

def is_valid(A):
    """Check if A satisfies cardinality-separated subset sums."""
    n = len(A)
    if n == 0:
        return True
    
    sums_by_card = compute_sums_by_cardinality(A)
    
    # Check: no sum appears in two different cardinality sets
    sum_to_card = {}
    for k in range(n + 1):
        for s in sums_by_card[k]:
            if s in sum_to_card:
                if sum_to_card[s] != k:
                    return False
            else:
                sum_to_card[s] = k
    return True

def can_add(A, x):
    """Check if adding x to A preserves validity. A is sorted, x > A[-1]."""
    # Quick check: x must not be in A
    if x in A:
        return False
    
    n = len(A)
    # Compute sums for current A
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    for a in A:
        for k in range(n - 1, -1, -1):
            if sums_by_card[k]:
                for s in sums_by_card[k]:
                    sums_by_card[k + 1].add(s + a)
    
    # Now add x: new sums are old sums + x at cardinality k+1
    # Check for conflicts: for each k, new sums at k+1 must not conflict with existing sums at any other cardinality
    # Also new sums at k+1 must not conflict with existing sums at k+1 (but they're added to the set, so no conflict within same cardinality)
    
    # Existing sums at each cardinality
    # New sums at cardinality k+1: {s + x for s in sums_by_card[k]}
    
    # For each new sum at cardinality j = k+1, check it doesn't appear in any other cardinality's sums
    # Also check it doesn't appear in sums_by_card[j] already (but we're adding to it, so within same cardinality is fine)
    
    for k in range(n, -1, -1):
        if not sums_by_card[k]:
            continue
        new_card = k + 1
        for s in sums_by_card[k]:
            new_sum = s + x
            # Check if new_sum is in any cardinality set other than new_card
            for j in range(n + 1):
                if j == new_card:
                    continue
                if new_sum in sums_by_card[j]:
                    return False
    return True

def greedy_build(start_val=1, time_budget=5.0):
    """Greedily build a valid set by adding the smallest possible element."""
    A = [start_val]
    end_time = time.time() + time_budget
    
    while time.time() < end_time and not is_done():
        best_x = None
        # Try adding elements in increasing order
        for x in range(A[-1] + 1, 257):
            if can_add(A, x):
                best_x = x
                break
        
        if best_x is None:
            break
        
        A.append(best_x)
    
    return A

def local_search(A, time_budget=10.0):
    """Try to improve set A by swapping elements."""
    A = sorted(A)
    end_time = time.time() + time_budget
    best = A[:]
    
    while time.time() < end_time and not is_done():
        improved = False
        # Try removing one element and adding a different one
        for i in range(len(A)):
            if time.time() > end_time or is_done():
                break
            without_i = A[:i] + A[i+1:]
            # Try adding various elements
            for x in range(1, 257):
                if x in without_i:
                    continue
                candidate = sorted(without_i + [x])
                if len(candidate) > len(best) or (len(candidate) == len(best) and is_valid(candidate)):
                    if is_valid(candidate):
                        if len(candidate) > len(best):
                            best = candidate[:]
                            improved = True
                            break
                        elif candidate != best:
                            # Same size, try this one
                            if is_valid(candidate):
                                best = candidate[:]
                                improved = True
                                break
        
        if not improved:
            # Try random restarts
            break
    
    return best

def random_search(time_budget=15.0):
    """Random search for valid sets."""
    end_time = time.time() + time_budget
    best = []
    
    while time.time() < end_time and not is_done():
        # Try random sizes
        size = random.randint(5, 12)
        A = sorted(random.sample(range(1, 257), size))
        if is_valid(A) and len(A) > len(best):
            best = A[:]
    
    return best

def construct():
    """Return a list of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size."""
    
    # Start with known valid baseline: powers of 2
    baseline = []
    v = 1
    while v <= 256:
        baseline.append(v)
        v *= 2
    
    best = baseline[:]
    
    # Greedy build
    if not is_done():
        greedy_result = greedy_build(start_val=1, time_budget=5.0)
        if len(greedy_result) > len(best):
            best = greedy_result[:]
    
    # Local search from best
    if not is_done():
        local_result = local_search(best, time_budget=10.0)
        if len(local_result) > len(best):
            best = local_result[:]
    
    # Random search
    if not is_done():
        random_result = random_search(time_budget=10.0)
        if len(random_result) > len(best):
            best = random_result[:]
    
    # Final validation
    result = [int(x) for x in best]
    if result and not is_valid(result):
        result = [int(x) for x in baseline]
    
    return result
