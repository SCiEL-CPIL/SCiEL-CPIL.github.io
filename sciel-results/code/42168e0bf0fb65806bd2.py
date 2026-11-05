"""Rewritten program: maximum subset of [1,100] with no subset summing to 200."""
import time
import random
import itertools

def _check_valid(S, target=200):
    """Check if no subset of S sums to target using bitset DP."""
    dp = 1  # bit i set means sum i is achievable
    for x in S:
        dp |= (dp << x)
        if (dp >> target) & 1:
            return False
    return True

def _get_dp(S, target=200):
    """Compute bitset dp for set S, truncated to target bits."""
    dp = 1
    for x in S:
        dp |= (dp << x)
        dp &= (1 << (target + 1)) - 1
    return dp

def _can_add(x, dp_S, target=200):
    """Check if adding x to a set with bitset dp_S is valid."""
    if x >= target:
        return False
    return ((dp_S >> (target - x)) & 1) == 0

def construct():
    deadline = time.time() + 55  # Safety margin
    
    best_set = list(range(66, 101))
    best_size = len(best_set)
    
    def local_search(start_set, time_limit):
        """Local search to maximize size starting from start_set."""
        S = set(start_set)
        if not _check_valid(S, 200):
            return set(), 0
        
        size = len(S)
        dp_S = _get_dp(S)
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Try to add elements
            for x in range(1, 66):
                if time.time() > time_limit:
                    break
                if x in S:
                    continue
                
                if _can_add(x, dp_S):
                    S.add(x)
                    size += 1
                    dp_S = _get_dp(S)
                    improved = True
                    break  # Restart to pick up new opportunities
            
            # If no direct addition, try swaps
            if not improved and time.time() < time_limit:
                for x in range(1, 66):
                    if time.time() > time_limit:
                        break
                    if x in S:
                        continue
                    
                    # Try removing one element y to make room for x
                    for y in list(S):
                        if time.time() > time_limit:
                            break
                        new_S = S - {y}
                        new_S.add(x)
                        if _check_valid(new_S, 200):
                            S = new_S
                            dp_S = _get_dp(S)
                            improved = True
                            break
                    if improved:
                        break
        
        return S, size
    
    # Try multiple starting points
    start_sets = [
        list(range(66, 101)),  # Size 35
        list(range(70, 101)),  # Size 31
        list(range(60, 101)),  # Size 41, check validity
    ]
    
    # Add some random starting sets
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline:
            break
        rand_set = random.sample(range(1, 101), 40)
        if _check_valid(set(rand_set), 200):
            start_sets.append(rand_set)
    
    for start_set in start_sets:
        if time.time() > deadline - 5:
            break
        
        result_set, result_size = local_search(start_set, deadline - 5)
        
        if result_size > best_size:
            best_size = result_size
            best_set = list(result_set)
    
    # Final polish: try to add more elements to the best set found
    if time.time() < deadline - 2:
        S = set(best_set)
        dp_S = _get_dp(S)
        for x in range(1, 66):
            if x in S:
                continue
            if _can_add(x, dp_S):
                S.add(x)
                dp_S = _get_dp(S)
        if len(S) > best_size:
            best_set = list(S)
            best_size = len(S)
    
    return list(best_set)
