# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def construct():
    """
    Construct a large set of distinct positive integers <= 512 such that no element
    is the arithmetic mean of any subset of at least two other elements.
    
    Strategy:
    1. Use a randomized greedy approach with multiple restarts.
    2. Use a fast bitset DP for subset sum checks to validate candidates.
    3. Pre-compute some structural properties to speed up validation.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    n = 512
    max_size = 48
    
    def is_valid_set(S):
        """
        Check if set S is valid: no element is the mean of any subset of >= 2 other elements.
        Uses bitset DP for subset sums.
        """
        S_list = list(S)
        m = len(S_list)
        if m < 3:
            return True
        
        # For each element a in S, check if 2*a can be formed as sum of any subset of S\{a} with size >= 2
        # We need to check: does there exist a subset T of S\{a} with |T| >= 2 and sum(T) = 2*a?
        
        for i, a in enumerate(S_list):
            others = S_list[:i] + S_list[i+1:]
            target = 2 * a
            
            # We need subset of 'others' with sum = target and size >= 2
            # Use DP: dp[s][k] = True if we can form sum s with k elements
            # But this is expensive. Let's use a different approach.
            
            # Alternative: For each pair (x, y) in others, check if x + y = 2a (size 2)
            # For size 3: x + y + z = 2a
            # etc.
            
            # Let's use bitset DP for subset sums with count tracking
            # dp[j] = bitmask of counts achievable for sum j
            # Actually, let's do: dp[s] = set of possible counts, but that's slow.
            
            # Better: Use two arrays. 
            # reachable[s] = minimum number of elements to achieve sum s (or -1 if not reachable)
            # But we need to know if ANY count >= 2 works.
            
            # Let's do: dp[s] = True if sum s is achievable. Then separately track min/max counts?
            # No, we need to know if there exists a subset with sum=target AND size>=2.
            
            # Simplest fast check:
            # 1. Check all pairs: if any x+y == target, invalid
            # 2. For larger subsets, use DP.
            
            # First, quick pair check
            n_others = len(others)
            found_pair = False
            seen = set()
            for x in others:
                if (target - x) in seen and (target - x) != x:
                    found_pair = True
                    break
                seen.add(x)
            if found_pair:
                return False
            
            # Now check subsets of size >= 3
            # Use bitset DP: dp[s] = True if sum s is achievable using some subset of others
            # But we need to ensure the subset has size >= 3 (since size 2 already checked)
            
            # Actually, let's track: dp[s] = max subset size achievable for sum s
            # Initialize dp[0] = 0, others = -1
            max_sum = target
            dp = [-1] * (max_sum + 1)
            dp[0] = 0
            
            for x in others:
                # Update dp in reverse to avoid using x multiple times
                for s in range(max_sum - x, -1, -1):
                    if dp[s] >= 0:
                        if dp[s] + 1 > dp[s + x]:
                            dp[s + x] = dp[s] + 1
            
            if dp[target] >= 3:
                return False
        
        return True

    def is_valid_addition(current_set, candidate):
        """
        Check if adding 'candidate' to current_set keeps the set valid.
        Only need to check:
        1. Is 'candidate' the mean of any subset of current_set? (size >= 2)
        2. Is any element in current_set the mean of a subset involving 'candidate'?
        """
        S = current_set
        a = candidate
        
        # Check 1: Is a the mean of any subset of S with size >= 2?
        # i.e., does there exist subset T of S with |T| >= 2 and sum(T) = 2*a?
        target = 2 * a
        if target > 512 * 2:  # Max possible sum
            pass
        
        # Quick pair check
        seen = set()
        for x in S:
            if (target - x) in seen and (target - x) != x:
                return False
            seen.add(x)
        
        # Check larger subsets using DP
        # We need subset of S with sum = target and size >= 3
        max_sum = min(target, 512 * len(S))
        if target <= max_sum:
            dp = [-1] * (target + 1)
            dp[0] = 0
            for x in S:
                for s in range(target - x, -1, -1):
                    if dp[s] >= 0:
                        if dp[s] + 1 > dp[s + x]:
                            dp[s + x] = dp[s] + 1
            if dp[target] >= 3:
                return False
        
        # Check 2: For each y in S, is y the mean of a subset of (S \ {y}) U {a} with size >= 2?
        # The subset must include a, because if it doesn't, it was already valid in S.
        # So we need: subset T of (S \ {y}) U {a} such that a in T, |T| >= 2, sum(T) = 2*y
        # This means: subset T' of (S \ {y}) such that a + sum(T') = 2*y, and 1 + |T'| >= 2 => |T'| >= 1
        # So: sum(T') = 2*y - a, with T' subset of S\{y}, |T'| >= 1
        
        for y in S:
            target2 = 2 * y - a
            if target2 < 0:
                continue
            if target2 > 512 * len(S):
                continue
            
            # Need subset T' of S\{y} with sum = target2 and |T'| >= 1
            others2 = [x for x in S if x != y]
            
            # Quick check: is target2 in others2? (size 1)
            if target2 in others2:
                return False
            
            # Check size >= 2 using DP
            max_sum2 = min(target2, 512 * len(others2))
            if target2 <= max_sum2:
                dp2 = [-1] * (target2 + 1)
                dp2[0] = 0
                for x in others2:
                    for s in range(target2 - x, -1, -1):
                        if dp2[s] >= 0:
                            if dp2[s] + 1 > dp2[s + x]:
                                dp2[s + x] = dp2[s] + 1
                if dp2[target2] >= 2:
                    return False
        
        return True

    best_set = []
    
    # Try multiple random orderings
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random shuffle of candidates
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        current_set = []
        for x in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(current_set, x):
                current_set.append(x)
                if len(current_set) > len(best_set):
                    best_set = list(current_set)
                    if len(best_set) >= max_size:
                        return best_set
        
        if time.time() > deadline:
            break
    
    # Ensure we return at least something valid
    if not best_set:
        # Fallback: simple greedy from 1
        current_set = []
        for x in range(1, n + 1):
            if is_valid_addition(current_set, x):
                current_set.append(x)
            if len(current_set) >= max_size:
                break
        best_set = current_set
    
    return best_set

# EVOLVE-BLOCK-END
