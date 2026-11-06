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
    Check if target can be formed by a subset of items (Fractions).
    Uses a pruned DFS. Returns True if found, False otherwise.
    """
    if not items:
        return target == Fraction(0)
    
    # Filter items: only those < target can be in a sum equal to target
    # (since all are positive). If any item == target, return True.
    filtered = []
    for v in items:
        if v == target:
            return True
        if v < target:
            filtered.append(v)
    
    if not filtered:
        return False
        
    # Sort descending to prune faster
    filtered.sort(key=lambda x: x, reverse=True)
    n = len(filtered)
    
    # Suffix sums
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + filtered[i]
        
    if suffix_sums[0] < target:
        return False

    # Memoization: (idx, current_sum) -> bool
    # Since sums are Fractions, we can use a set of tuples
    memo = set()
    
    def dfs(idx, current_sum):
        if current_sum == target:
            return True
        if idx >= n:
            return False
        if current_sum > target:
            return False
        if current_sum + suffix_sums[idx] < target:
            return False
            
        key = (idx, current_sum)
        if key in memo:
            return False
            
        # Option 1: Include
        if current_sum + filtered[idx] <= target:
            if dfs(idx + 1, current_sum + filtered[idx]):
                return True
                
        # Option 2: Exclude
        if dfs(idx + 1, current_sum):
            return True
            
        memo.add(key)
        return False
        
    return dfs(0, Fraction(0))

def _check_violation(S, a):
    """
    Check if 1/a can be formed by a subset of {1/x for x in S, x != a}.
    Returns True if violation found, else False.
    """
    target = Fraction(1, a)
    items = []
    for x in S:
        if x != a:
            items.append(Fraction(1, x))
            
    return _can_form_target(target, items)

def is_valid(S):
    for a in S:
        if _check_violation(S, a):
            return False
    return True

def greedy_add_strategy(order, best_size):
    """
    Grow a set by adding elements in `order`, keeping validity.
    """
    chosen = []
    recips = []
    for v in order:
        if v in chosen:
            continue
        
        # Check if 1/v is a subset sum of current recips
        if _can_form_target(Fraction(1, v), recips):
            continue
        
        # Check if adding v breaks validity of existing elements
        valid = True
        for u in chosen:
            # 1/u must not be a subset sum of (recips + [1/v])
            # This is equivalent to: 1/u is not a subset sum of recips (already true) 
            # AND 1/u is not a subset sum of (recips + [1/v]) that includes 1/v.
            # i.e., 1/u - 1/v is a subset sum of recips.
            # Note: 1/u > 1/v if u < v. If u > v, 1/u < 1/v, so 1/u - 1/v is negative, impossible.
            if u < v:
                target_diff = Fraction(1, u) - Fraction(1, v)
                if target_diff > 0 and _can_form_target(target_diff, recips):
                    valid = False
                    break
        
        if valid:
            chosen = trial = chosen + [v]
            recips.append(Fraction(1, v))
            
    return chosen

def construct():
    best_set = []
    best_size = 0
    
    # 1. Structural anchors
    # Seed: [13..24]
    S0 = list(range(13, 25))
    if is_valid(S0):
        best_set = S0
        best_size = len(S0)
        
    # Top halves
    for start in range(2, 25):
        if _time_left() < 2.0:
            break
        S = list(range(start, 25))
        if len(S) <= best_size:
            continue
        if is_valid(S):
            best_set = S
            best_size = len(S)
            
    # 2. Greedy Add
    orders = [
        list(range(1, 25)),
        list(range(24, 0, -1)),
        list(range(1, 25, 2)) + list(range(2, 25, 2)),
        list(range(24, 0, -2)) + list(range(23, 0, -2))
    ]
    
    for order in orders:
        if _time_left() < 2.0:
            break
        S = greedy_add_strategy(order, best_size)
        if len(S) > best_size and is_valid(S):
            best_set = S
            best_size = len(S)
            
    # 3. Randomized Greedy Add
    rng = random.Random(42)
    for _ in range(100):
        if _time_left() < 2.0:
            break
        order = list(range(1, 25))
        rng.shuffle(order)
        S = greedy_add_strategy(order, best_size)
        if len(S) > best_size and is_valid(S):
            best_set = S
            best_size = len(S)
            
    # 4. Local Search / Add
    # If best_set is valid, try to add elements not in it
    if best_set and _time_left() > 5.0:
        current = list(best_set)
        improved = True
        while improved and _time_left() > 1.0:
            improved = False
            for v in range(1, 25):
                if v in current:
                    continue
                trial = current + [v]
                if is_valid(trial):
                    current = trial
                    improved = True
                    break # Restart loop with new set
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
            
    # 5. Local Search / Swap
    # Try to improve best_set by swapping elements
    if best_set and _time_left() > 5.0:
        current = list(best_set)
        improved = True
        while improved and _time_left() > 1.0:
            improved = False
            for i in range(len(current)):
                if _time_left() < 1.0:
                    break
                for v in range(1, 25):
                    if v in current:
                        continue
                    trial = current[:i] + current[i+1:] + [v]
                    if is_valid(trial):
                        # Check if this swap allows further additions?
                        # For now, just accept if valid. 
                        # To avoid cycling, we could require it to be "better" in some way,
                        # but since size is same, we just want to find a better configuration for future adds.
                        # Let's do a simple hill climb: accept if valid.
                        current = trial
                        improved = True
                        break
                if improved:
                    break
                    
        # After swapping, try to add more
        if _time_left() > 2.0:
            improved = True
            while improved and _time_left() > 1.0:
                improved = False
                for v in range(1, 25):
                    if v in current:
                        continue
                    trial = current + [v]
                    if is_valid(trial):
                        current = trial
                        improved = True
                        break
            if len(current) > best_size:
                best_set = current
                best_size = len(current)

    if not best_set:
        best_set = list(range(13, 25))
        
    return best_set

if __name__ == "__main__":
    print(construct())
