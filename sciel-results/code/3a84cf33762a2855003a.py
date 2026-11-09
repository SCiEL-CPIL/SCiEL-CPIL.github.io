"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 256}
_DEADLINE = None

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    d = 5
    while d * d <= x:
        if x % d == 0 or x % (d + 2) == 0:
            return False
        d += 6
    return True

def _valid_set(S):
    """Check if set S is valid: for all distinct a,b,c in S, a ∤ b*c."""
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            for k in range(j + 1, m):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(S, x):
    """Check if x can be added to set S without violating the constraint."""
    S_list = list(S)
    m = len(S_list)
    # x ∤ a*b for all distinct a,b in S
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            if (a * b) % x == 0:
                return False
    # a ∤ x*b for all distinct a,b in S
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            if (x * b) % a == 0:
                return False
    return True

def _greedy_order(nums, order_func):
    """Greedy: add numbers in the order given by order_func."""
    S = set()
    for x in order_func(nums):
        if _can_add(S, x):
            S.add(x)
    return S

def _branch_and_bound(remaining, current_set, best, deadline):
    """
    Branch and bound for maximum independent set in 3-uniform hypergraph.
    remaining: list of candidate numbers not yet decided
    current_set: set of numbers chosen so far
    best: list holding the best solution found
    deadline: time limit
    """
    if time.time() > deadline:
        return
    
    if not remaining:
        if len(current_set) > len(best[0]):
            best[0] = set(current_set)
        return
    
    # Upper bound: current size + len(remaining)
    if len(current_set) + len(remaining) <= len(best[0]):
        return
    
    # Choose a pivot element to branch on
    # Use the element that appears in the most constraints (highest "degree")
    # For simplicity, just use the first element
    pivot = remaining[0]
    rest = remaining[1:]
    
    # Option 1: Include pivot
    if _can_add(current_set, pivot):
        new_set = set(current_set)
        new_set.add(pivot)
        # Filter remaining: remove elements that can't be added with new_set
        filtered = []
        for x in rest:
            if _can_add(new_set, x):
                filtered.append(x)
        _branch_and_bound(filtered, new_set, best, deadline)
    
    # Option 2: Exclude pivot
    _branch_and_bound(rest, current_set, best, deadline)

def _local_search(S, candidates, deadline):
    """Try to improve S by removing one element and adding others."""
    best = set(S)
    best_size = len(best)
    
    # Try all removals
    for rem in list(best):
        if time.time() > deadline:
            break
        S2 = set(best)
        S2.discard(rem)
        # Try to add candidates in various orders
        for x in candidates:
            if x in S2:
                continue
            if _can_add(S2, x):
                S2.add(x)
                # Try to add more
                for y in candidates:
                    if y in S2:
                        continue
                    if _can_add(S2, y):
                        S2.add(y)
                if len(S2) > best_size:
                    best = set(S2)
                    best_size = len(best)
    
    return best

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 55  # 55 second budget (leave 5s margin for 60s limit)
    
    n = P['n']
    nums = list(range(2, n + 1))
    
    # Get primes as initial solution
    primes = [x for x in nums if _is_prime(x)]
    best_set = set(primes)
    best_size = len(primes)
    
    # Try greedy with different orderings to get a better starting point
    orderings = [
        lambda nums: sorted(nums, key=lambda x: -x),  # descending
        lambda nums: sorted(nums, key=lambda x: x),    # ascending
        lambda nums: sorted(nums, key=lambda x: -sum(1 for d in range(2, int(x**0.5)+1) if x%d==0)),  # by number of divisors desc
        lambda nums: sorted(nums, key=lambda x: (x % 2, -x)),  # evens first then odds desc
    ]
    
    for order_func in orderings:
        if time.time() > _DEADLINE:
            break
        S_greedy = _greedy_order(nums, order_func)
        if len(S_greedy) > best_size:
            best_set = S_greedy
            best_size = len(S_greedy)
    
    # Use branch and bound from the best greedy solution
    # Filter candidates to those that can potentially be added
    candidates = [x for x in nums if x not in best_set]
    
    if time.time() < _DEADLINE:
        # Try branch and bound with a time-limited search
        best_wrapper = [best_set]
        # Only consider candidates that could improve the solution
        # Sort by some heuristic: numbers with fewer conflicts first
        def conflict_count(x):
            """Count how many pairs in best_set conflict with x."""
            count = 0
            S_list = list(best_set)
            m = len(S_list)
            for i in range(m):
                a = S_list[i]
                for j in range(i + 1, m):
                    b = S_list[j]
                    if (a * b) % x == 0 or (x * b) % a == 0:
                        count += 1
                        break
            return count
        
        candidates_sorted = sorted(candidates, key=conflict_count)
        
        # Limit the search space for B&B to keep it tractable
        # Only consider top candidates that have low conflict
        limited_candidates = [x for x in candidates_sorted if conflict_count(x) <= best_size]
        
        if limited_candidates and time.time() < _DEADLINE:
            _branch_and_bound(limited_candidates, best_set, best_wrapper, _DEADLINE)
            if len(best_wrapper[0]) > best_size:
                best_set = best_wrapper[0]
                best_size = len(best_set)
    
    # Local search polish
    if time.time() < _DEADLINE:
        candidates2 = [x for x in nums if x not in best_set]
        best_set = _local_search(best_set, candidates2, _DEADLINE)
        best_size = len(best_set)
    
    # Final verification
    S_list = sorted(best_set)
    if not _valid_set(S_list):
        # Fallback to primes
        S_list = sorted(primes)
    
    return list(S_list)
# EVOLVE-BLOCK-END
