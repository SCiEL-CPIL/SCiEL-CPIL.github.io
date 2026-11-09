import random
import time
import sys
from itertools import combinations

L = 5354228880  # lcm(1..24)

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a subset of weights.
    Uses meet-in-the-middle.
    Returns True if found, False otherwise.
    """
    n = len(weights)
    if n == 0:
        return target == 0
    if target == 0:
        return True
    
    # Filter weights > target
    weights = [w for w in weights if w <= target]
    if not weights:
        return target == 0
        
    n = len(weights)
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    left_sums = set()
    left_sums.add(0)
    for w in left:
        new_sums = set()
        for s in left_sums:
            if s + w <= target:
                new_sums.add(s + w)
        left_sums.update(new_sums)
        
    right_sums = set()
    right_sums.add(0)
    for w in right:
        new_sums = set()
        for s in right_sums:
            if s + w <= target:
                new_sums.add(s + w)
        right_sums.update(new_sums)
        
    # Check intersection
    for s in left_sums:
        if (target - s) in right_sums:
            return True
    return False

def get_violation(S):
    """
    Returns (x, T) if x's reciprocal is sum of reciprocals of T, else None.
    S is a set of ints.
    """
    S_list = list(S)
    for x in S_list:
        target = L // x
        weights = []
        T = []
        for y in S_list:
            if y != x:
                w = L // y
                weights.append(w)
                T.append(y)
        
        if subset_sum_exists(weights, target):
            def find_subset(weights, indices, target):
                if target == 0:
                    return []
                if not weights:
                    return None
                if weights[0] <= target:
                    res = find_subset(weights[1:], indices[1:], target - weights[0])
                    if res is not None:
                        return [indices[0]] + res
                res = find_subset(weights[1:], indices[1:], target)
                if res is not None:
                    return res
                return None

            T_indices = find_subset(weights, T, target)
            if T_indices is not None:
                return x, T_indices
            else:
                return x, T
                
    return None

def is_valid(S):
    return get_violation(S) is None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    all_elements = list(range(1, 25))
    n = len(all_elements)
    
    # Precompute L//x for each element
    recip = [L // x for x in all_elements]
    
    # Precompute all subset sums of reciprocals for each possible "other" set
    # For element i, the forbidden target is recip[i]
    # We need to check if recip[i] can be formed by a subset of {recip[j] : j in S, j != i}
    
    # Strategy: Backtracking search with forward checking
    # Order elements to maximize pruning: try to include/exclude in a smart order
    
    # Sort elements by reciprocal value (ascending) - smaller reciprocals are "harder" to sum
    # Actually, let's think about which order gives best pruning
    
    # Key insight: If we include element x, then for any OTHER element y in our set,
    # we need recip[x] NOT to be a subset sum of {recip[z] : z in S, z != x}
    
    # Let's use a different approach: iterative deepening on size
    # Try to find a valid set of size k, starting from k=24 down
    
    best_set = None
    best_size = 0
    
    # First, get a quick baseline using the repair method
    S = set(all_elements)
    while True:
        if time.time() > deadline:
            break
        violation = get_violation(S)
        if violation is None:
            break
        x, T = violation
        # Remove x (the target element)
        S.remove(x)
        if not S:
            break
    
    if S:
        best_set = set(S)
        best_size = len(S)
    
    # Now try to find larger sets using backtracking with forward checking
    # We'll search for sets of size k from best_size+1 up to 24
    
    # Precompute: for each element i, and each subset of other elements,
    # whether recip[i] is a subset sum of that subset's reciprocals
    # This is expensive, so we'll compute on-the-fly with caching
    
    # Better approach: use the fact that we're looking for MAXIMUM size
    # Try greedy + local search with the CSP perspective
    
    # Precompute conflict graph: element i conflicts with element j if
    # including both creates a potential violation
    # But violations involve SUBSETS, not just pairs, so this is complex
    
    # Let's use a smarter backtracking:
    # Order elements by "difficulty" - elements that are more likely to cause violations first
    # An element x is "difficult" if recip[x] is easily formed by sums of other reciprocals
    
    # Compute difficulty score for each element
    difficulty = []
    for i in range(n):
        # Count how many subsets of other elements sum to recip[i]
        # This is expensive, so use a proxy: number of other elements that can contribute
        # (i.e., recip[j] <= recip[i])
        count = sum(1 for j in range(n) if j != i and recip[j] <= recip[i])
        difficulty.append(count)
    
    # Sort elements by difficulty (most difficult first for better pruning)
    order = sorted(range(n), key=lambda i: difficulty[i], reverse=True)
    
    # Backtracking search
    def backtrack(idx, current_set, current_recips):
        nonlocal best_set, best_size, deadline, start_time
        
        if time.time() > deadline:
            return False
        
        # If current set is already smaller than best, prune
        remaining = n - idx
        if len(current_set) + remaining <= best_size:
            return False
        
        if idx == n:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return False
        
        i = order[idx]
        
        # Option 1: Include element i
        # Check if including i violates any existing element's constraint
        can_include = True
        for j in current_set:
            # Check if recip[j] can be formed by a subset of (current_recips + [recip[i]])
            # i.e., is recip[j] in subset sums of current_recips ∪ {recip[i]}?
            # This is: is (recip[j] - recip[i]) in subset sums of current_recips, OR recip[j] in subset sums of current_recips?
            # If recip[j] was already formable without i, we wouldn't be in this state (we maintain validity)
            # So we only need to check: is (recip[j] - recip[i]) a subset sum of current_recips?
            # AND recip[i] <= recip[j] (otherwise can't subtract)
            if recip[i] <= recip[j]:
                target_diff = recip[j] - recip[i]
                # Check if target_diff is a subset sum of current_recips
                if target_diff == 0:
                    # This means recip[j] == recip[i], which can't happen for distinct elements
                    # But if it did, it would be a violation
                    can_include = False
                    break
                elif subset_sum_exists(current_recips, target_diff):
                    can_include = False
                    break
        
        # Also check: does recip[i] become formable by subset of current_recips?
        # If yes, then element i would be violated
        if can_include and subset_sum_exists(current_recips, recip[i]):
            can_include = False
        
        if can_include:
            new_set = current_set + [i]
            new_recips = current_recips + [recip[i]]
            if backtrack(idx + 1, new_set, new_recips):
                return True
        
        # Option 2: Exclude element i
        if backtrack(idx + 1, current_set, current_recips):
            return True
        
        return False
    
    # Run backtracking
    backtrack(0, [], [])
    
    if best_set is None:
        return [1]
        
    return list(best_set)
