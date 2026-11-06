import sys
from fractions import Fraction
from functools import lru_cache

def construct():
    """
    Returns a nonempty subset of [1, 24] that maximizes cardinality
    such that no reciprocal of a selected element equals the sum of
    reciprocals of a subset of the other selected elements.
    Uses Branch and Bound with integer arithmetic (LCM scaling).
    """
    
    # Precompute weights: LCM(1..24) / i
    # LCM(1..24) = 5354228880
    LCM = 5354228880
    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = LCM // i
        
    elements = list(range(1, 25))
    
    # Helper to check if target can be formed by subset of nums
    # nums are positive integers
    def can_form_target(nums, target):
        if target == 0:
            return True
        if target < 0:
            return False
        if not nums:
            return False
        
        # Use a set of reachable sums, pruning sums > target
        # This is often faster than DFS in Python for moderate K due to lower overhead
        reachable = {0}
        for n in nums:
            if n > target:
                continue
            # Create new sums
            new_sums = set()
            for s in reachable:
                ns = s + n
                if ns <= target:
                    new_sums.add(ns)
            if target in new_sums:
                return True
            reachable.update(new_sums)
            # Optional: if len(reachable) becomes too large, switch to DFS?
            # But for K<=23, 2^23 is ~8M, which is too many for a set in Python if it explodes.
            # However, pruning by target helps.
            # If the set grows too large, we might timeout.
            # Let's stick to this for now, it's a standard improvement.
            
        return target in reachable

    best_set = []
    
    def check_conflict(x, S):
        """
        Checks if adding x to S creates a violation.
        S is a list of selected elements.
        """
        wx = weights[x]
        S_weights = [weights[y] for y in S]
        
        # 1. Does x equal sum of subset of S?
        if can_form_target(S_weights, wx):
            return True
            
        # 2. Does any y in S equal sum of subset of (S \ {y}) U {x}?
        # This is equivalent to: w_y - w_x is a subset sum of S \ {y, x}
        # Note: x is not in S yet.
        for y in S:
            wy = weights[y]
            if wy < wx:
                # w_y - w_x < 0, cannot be a sum of positive weights
                continue
            # We need to check if w_y - w_x can be formed by S excluding y and x
            # Since x is not in S, we just exclude y from S.
            # We pass the list of weights excluding y.
            # To avoid creating a new list every time, we could pass indices, 
            # but the list comprehension is small.
            remaining_weights = [weights[z] for z in S if z != y]
            if can_form_target(remaining_weights, wy - wx):
                return True
                
        return False

    def backtrack(idx, current_set):
        nonlocal best_set
        
        # Pruning: if even adding all remaining can't beat best
        if len(current_set) + (24 - idx) <= len(best_set):
            return
            
        if idx == 24:
            if len(current_set) > len(best_set):
                best_set = current_set.copy()
            return
            
        x = elements[idx]
        
        # Option 1: Include x
        # Check if valid
        if not check_conflict(x, current_set):
            current_set.append(x)
            backtrack(idx + 1, current_set)
            current_set.pop()
            
        # Option 2: Exclude x
        backtrack(idx + 1, current_set)

    backtrack(0, [])
    
    return best_set

if __name__ == "__main__":
    result = construct()
    print(result)
