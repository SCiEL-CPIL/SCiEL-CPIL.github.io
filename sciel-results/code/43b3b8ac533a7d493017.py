import time
import random
import itertools

def construct():
    deadline = time.time() + 50.0
    best_set = []
    target_size = 48
    
    def full_verify(S):
        """
        Rigorously checks if S is a valid set.
        Returns True if valid.
        """
        n = len(S)
        S_set = set(S)
        
        # Helper: DP to check if target sum is achievable with k elements from L
        def can_form_sum(L, k, target):
            if k == 0:
                return target == 0
            if k > len(L):
                return False
            # Simple bounds
            if target < sum(L[:k]) or target > sum(L[-k:]):
                return False
            
            # Bitset DP
            # dp[j] is a bitset where bit s is 1 if sum s is achievable with j elements
            dp = [0] * (k + 1)
            dp[0] = 1
            mask = (1 << (target + 1)) - 1
            
            for x in L:
                if x > target:
                    continue
                for j in range(min(k, len(L)), 0, -1):
                    dp[j] |= (dp[j-1] << x) & mask
            return (dp[k] >> target) & 1

        for i, x in enumerate(S):
            # 1. Is x the mean of a subset of OTHER elements?
            others = [y for y in S if y != x]
            for k in range(2, n):
                target_sum = k * x
                if can_form_sum(others, k, target_sum):
                    return False
            
            # 2. Is x the mean of a subset including x? 
            # Wait, the condition is: "No element may be the exact arithmetic mean of any subset of at least two OTHER elements."
            # So we only check if x is the mean of a subset of S \ {x}.
            # We do NOT check if x is the mean of a subset including x (trivially impossible unless 0).
            # And we do NOT check if x is the mean of a subset of S \ {x} of size 1.
            # So the check above is sufficient for each element x.
            # We don't need the complex Check 2 from the seed!
            # The seed's Check 2 was for when adding x, checking if EXISTING a becomes invalid.
            # But full_verify checks every element in the final set.
            # So for each x in S, we check if x is mean of subset of S\{x}.
            # This covers all cases.
            
        return True

    def fast_check(x, S, S_set):
        """
        Fast heuristic check to decide if x should be added.
        Checks k=2 rigorously.
        Checks k=3,4,5 with limited DP.
        """
        n = len(S)
        
        # 1. Check if x is mean of subset of S (size >= 2)
        # k=2
        for a in S:
            b = 2 * x - a
            if b != a and b in S_set:
                return False
        
        # k >= 3
        # Simple bounds
        for k in range(3, min(n, 6) + 1):
            target = k * x
            min_sum = sum(S[:k])
            max_sum = sum(S[-k:])
            if target < min_sum or target > max_sum:
                continue
            
            # Simple DP for small k
            # This is a simplified check. If it passes, we assume it's fine.
            # If it fails, we reject.
            # It might falsely reject, but that's okay (we just try another x).
            # It might falsely accept, but full_verify will catch it later.
            
            # We'll skip the DP here for speed in the greedy phase,
            # relying on the fact that k>=3 collisions are rarer.
            # Or do a very fast DP.
            pass # For now, just bounds check.

        # 2. Check if existing a becomes invalid?
        # We skip this in fast_check to save time.
        # The risk is that we add x, and later an a is invalid.
        # full_verify will catch this when we check the final set.
        # So we might add "bad" x's, but we will discard the set if it fails full_verify.
        # This might reduce the chance of finding a valid set quickly,
        # but randomized restarts help.
        
        return True

    while time.time() < deadline:
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        S = []
        S_set = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            if len(S) >= target_size:
                break
                
            if fast_check(x, S, S_set):
                S.append(x)
                S_set.add(x)
        
        if len(S) > len(best_set):
            best_set = S.copy()
            
            # Verify the best_set found so far
            if len(best_set) >= target_size:
                if full_verify(best_set):
                    return best_set
            elif len(best_set) > 0:
                # Verify smaller sets too? 
                # Only if we can't find a bigger one.
                if full_verify(best_set):
                    # Keep it as best valid
                    pass
                else:
                    best_set = [] # Discard invalid

        # If we have a valid set of size 48, we return it.
        # If we run out of time, return the best valid set found.
        # We need to track the best VALID set.
        
    # At the end, return the best set that passed full_verify
    # We need to keep track of the best valid set explicitly.
    # Let's refactor to track `best_valid`.
    
    return best_set # Might be invalid if we didn't verify. We should fix this.

# Refactor to track best_valid
def construct():
    deadline = time.time() + 50.0
    best_valid = []
    target_size = 48
    
    def full_verify(S):
        n = len(S)
        S_set = set(S)
        
        def can_form_sum(L, k, target):
            if k == 0:
                return target == 0
            if k > len(L):
                return False
            if target < sum(L[:k]) or target > sum(L[-k:]):
                return False
            
            dp = [0] * (k + 1)
            dp[0] = 1
            mask = (1 << (target + 1)) - 1
            
            for x in L:
                if x > target:
                    continue
                for j in range(min(k, len(L)), 0, -1):
                    dp[j] |= (dp[j-1] << x) & mask
            return (dp[k] >> target) & 1

        for i, x in enumerate(S):
            others = [y for y in S if y != x]
            for k in range(2, n):
                target_sum = k * x
                if can_form_sum(others, k, target_sum):
                    return False
        return True

    def fast_check(x, S, S_set):
        n = len(S)
        # k=2 check
        for a in S:
            b = 2 * x - a
            if b != a and b in S_set:
                return False
        # Bounds for k=3,4,5
        for k in range(3, min(n, 5) + 1):
            target = k * x
            min_sum = sum(S[:k])
            max_sum = sum(S[-k:])
            if min_sum <= target <= max_sum:
                # Do a quick check?
                # For speed, we just accept if bounds pass.
                pass
        return True

    while time.time() < deadline:
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        S = []
        S_set = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(S) >= target_size:
                break
            
            if fast_check(x, S, S_set):
                S.append(x)
                S_set.add(x)
        
        if len(S) > len(best_valid):
            if full_verify(S):
                best_valid = S.copy()
                if len(best_valid) >= target_size:
                    return best_valid

    return best_valid
