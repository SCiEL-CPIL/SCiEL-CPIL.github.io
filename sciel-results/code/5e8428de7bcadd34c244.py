import time
from fractions import Fraction
import random
from itertools import combinations

_START_TIME = time.time()
DEADLINE = 85.0  # 5 seconds safety margin

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _can_form_target(target, items):
    """
    Check if target can be formed as a sum of a subset of items.
    items: list of Fraction values
    Returns True if possible, False otherwise.
    """
    if target == 0:
        return True
    if not items:
        return False
    
    # Filter items that are <= target (positive values only)
    valid_items = [v for v in items if 0 < v <= target]
    if not valid_items:
        return False
    
    # If any single item equals target
    for v in valid_items:
        if v == target:
            return True
    
    # Sort descending for better pruning
    valid_items.sort(reverse=True)
    
    # Quick check: if sum of all valid items < target, impossible
    if sum(valid_items) < target:
        return False
    
    n = len(valid_items)
    
    # Use meet-in-the-middle for moderate n
    if n <= 20:
        # Split into two halves
        mid = n // 2
        left = valid_items[:mid]
        right = valid_items[mid:]
        
        # Generate all subset sums for left half
        left_sums = set([0])
        for v in left:
            new_sums = set()
            for s in left_sums:
                new_sums.add(s + v)
            left_sums = left_sums | new_sums
        
        # For each sum in right, check if target - sum is in left_sums
        right_sums = set([0])
        for v in right:
            new_sums = set()
            for s in right_sums:
                ns = s + v
                if ns <= target:
                    new_sums.add(ns)
            right_sums = right_sums | new_sums
        
        for rs in right_sums:
            if (target - rs) in left_sums:
                return True
        
        return False
    else:
        # For larger sets, use greedy + DFS with pruning
        # Try a simple DFS with memoization on remaining sum
        # But this could be exponential, so add a time guard
        
        def dfs(idx, current_sum):
            if current_sum == target:
                return True
            if idx >= n:
                return False
            if current_sum > target:
                return False
            
            # Pruning: if current_sum + sum(valid_items[idx:]) < target
            # We precompute suffix sums
            if current_sum + suffix_sums[idx] < target:
                return False
            
            # Include
            if current_sum + valid_items[idx] <= target:
                if dfs(idx + 1, current_sum + valid_items[idx]):
                    return True
            
            # Exclude
            if dfs(idx + 1, current_sum):
                return True
            
            return False
        
        # Precompute suffix sums
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + valid_items[i]
        
        return dfs(0, Fraction(0))

def _check_violation(S, a):
    """
    Check if 1/a can be formed by a subset of {1/x for x in S, x != a}.
    Returns True if violation exists, False otherwise.
    """
    target = Fraction(1, a)
    # Items are 1/x for x in S, x != a, and 1/x < target (i.e., x > a)
    items = []
    for x in S:
        if x != a:
            val = Fraction(1, x)
            if val < target:
                items.append(val)
            elif val == target:
                return True
    
    if not items:
        return False
    
    return _can_form_target(target, items)

def _build_conflict_graph(S):
    """
    Build conflict graph: edge between i and j if {i,j} is not subsum-free.
    Actually, we need: for each element a in S, check if 1/a can be formed
    by other elements. But the conflict is pairwise in the sense that we
    want to find a subset where NO element's reciprocal is a subsum of others.
    
    Let me reconsider: We want a set S such that for every a in S, 1/a is NOT
    a sum of reciprocals of a subset of S\{a}.
    
    This is not a simple pairwise conflict. It's a global constraint.
    So maximum clique on a conflict graph won't directly work.
    
    Let me think differently: We can use a different approach.
    """
    pass

def _is_valid_set(S):
    """Check if S is subsum-free."""
    for a in S:
        if _check_violation(S, a):
            return False
    return True

def _try_add(S, x):
    """Try to add x to S, return True if valid."""
    new_S = S + [x]
    return _is_valid_set(new_S)

def _greedy_construct(order):
    """
    Greedily construct a valid set by trying elements in the given order.
    """
    S = []
    for x in order:
        if _try_add(S, x):
            S.append(x)
    return S

def _local_search(S):
    """
    Try to improve S by removing one element and trying to add others.
    """
    S = list(S)
    improved = True
    while improved and _time_left() > 2.0:
        improved = False
        for i, a in enumerate(S):
            # Try removing a
            S_without_a = S[:i] + S[i+1:]
            # Try adding elements not in S
            for x in range(1, 25):
                if x not in S_without_a and x != a:
                    new_S = S_without_a + [x]
                    if _is_valid_set(new_S) and len(new_S) > len(S):
                        S = new_S
                        improved = True
                        break
            if improved:
                break
    return S

def _random_order():
    """Generate a random order of elements."""
    order = list(range(1, 25))
    random.shuffle(order)
    return order

def _heuristic_order():
    """
    Heuristic: larger elements first (they're safer targets since 1/x is smaller).
    Actually, we want to include elements that are hard to form.
    Large x means 1/x is small, so harder to form as a sum of other reciprocals.
    So start with large elements.
    """
    return list(range(24, 0, -1))

def _priority_order():
    """
    Order by how 'safe' an element is.
    An element a is safe if it's hard to form 1/a from others.
    Rough heuristic: larger a is safer.
    """
    return list(range(24, 0, -1))

def construct():
    best_set = None
    best_size = 0
    
    # Try different construction orders
    orders = [
        _heuristic_order(),  # Large to small
        _priority_order(),   # Same as above
        _random_order(),
        _random_order(),
        _random_order(),
    ]
    
    for order in orders:
        if _time_left() < 3.0:
            break
        
        S = _greedy_construct(order)
        
        # Local search to improve
        S = _local_search(S)
        
        if S and _is_valid_set(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    # If we have time, try more random restarts
    restart_count = 0
    while _time_left() > 5.0 and restart_count < 50:
        order = _random_order()
        S = _greedy_construct(order)
        S = _local_search(S)
        
        if S and _is_valid_set(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
        
        restart_count += 1
    
    if best_set is None:
        # Fallback: just large elements
        best_set = list(range(13, 25))
    
    return best_set
