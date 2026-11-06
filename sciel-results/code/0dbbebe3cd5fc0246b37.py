import random
import time
import sys
from itertools import combinations

def construct():
    """
    Constructs a set of up to 48 distinct positive integers <= 512.
    No element is the mean of any subset of at least two other elements.
    """
    N = 512
    MAX_SIZE = 48
    DEPTH = 4  # Check up to subset size 4 in fast mode
    
    def is_fast_valid_subset_sum(target, elements, k):
        """
        Checks if 'target' can be formed by exactly 'k' elements from 'elements'.
        Only for small k (1, 2, 3).
        """
        if k == 1:
            return target in elements
        elif k == 2:
            # Check if any pair sums to target
            # elements is a set
            for e in elements:
                if target - e in elements and target - e != e:
                    return True
            return False
        elif k == 3:
            # Check if any triple sums to target
            # Optimization: iterate pairs? Or just combinations for small sets?
            # Set size is <= 48. Combinations(48, 3) is ~ 17000.
            # This is called many times.
            # Better: iterate e1, e2, check if target-e1-e2 in set.
            # O(m^2)
            for e1 in elements:
                for e2 in elements:
                    if e1 < e2:
                        if target - e1 - e2 in elements and target - e1 - e2 > e2:
                            return True
            return False
        else:
            return False

    def is_fast_valid_candidate(x, S):
        """
        Checks if adding x to S maintains validity for subset sizes up to DEPTH+1.
        S is a set.
        """
        # 1. Check if x is the mean of any subset of S of size k in [2, DEPTH]
        S_set = S
        for k in range(2, DEPTH + 1):
            target = k * x
            if is_fast_valid_subset_sum(target, S_set, k):
                return False
        
        # 2. Check if any a in S is the mean of a subset of S U {x} \ {a}
        #    The subset must include x (otherwise it was valid in S)
        #    Let T = T' U {x}, T' subset of S \ {a}
        #    |T| = k, |T'| = k-1
        #    Sum(T') = k*a - x
        #    We check k in [2, DEPTH+1] => k-1 in [1, DEPTH]
        for a in S:
            others = S_set - {a}
            for k in range(2, DEPTH + 2): # k up to DEPTH+1
                j = k - 1
                target = k * a - x
                if target < 0: continue
                if is_fast_valid_subset_sum(target, others, j):
                    return False
        return True

    def is_full_valid(S):
        """
        Verifies S is valid using DP.
        """
        S_list = list(S)
        m = len(S_list)
        if m == 0: return True
        
        max_sum = sum(S_list)
        # DP[k][s] = True if sum s can be formed by k elements
        # Max k is m-1 (since subset of OTHERS)
        # Actually we need to check for each a, if a is mean of subset of S\{a}
        
        # We can just check all subsets? No.
        # For each a in S:
        #   Check if k*a is formable by k elements from S\{a} for any k>=2
        
        for a in S_list:
            others = [x for x in S_list if x != a]
            m_others = len(others)
            if m_others < 2: continue
            
            # DP for others
            # dp[k] is a set of sums? Or bitset?
            # Using set of sums is easier for small m, but m=47.
            # Sums can be up to 47*512 ~ 24000.
            # Bitset is faster.
            # dp[k] as integer mask.
            dp = [0] * (m_others + 1)
            dp[0] = 1 # Sum 0 with 0 elements
            
            for x in others:
                # Update dp in reverse order of k
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            # Check if any k*a is in dp[k]
            valid = True
            for k in range(2, m_others + 1):
                target = k * a
                if target > max_sum: break
                if (dp[k] >> target) & 1:
                    valid = False
                    break
            if not valid:
                return False
        return True

    best_set = set()
    start_time = time.time()
    time_limit = 55.0 # Safety margin
    
    # Strategy: Random Restart Hill Climbing
    # Try to find a set of size 48.
    # If we find one, verify it.
    # If verification fails, try to fix or restart.
    
    candidates = list(range(1, N + 1))
    
    for restart in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Start with a random set of size MAX_SIZE
        # Or start small and grow?
        # Starting with MAX_SIZE and pruning is often faster to find a valid large set
        # because the space of invalid sets is smaller?
        # Actually, random sets of size 48 are likely invalid.
        # Let's try growing.
        
        S = set()
        random.shuffle(candidates)
        
        # Greedy growth
        for x in candidates:
            if len(S) >= MAX_SIZE:
                break
            if is_fast_valid_candidate(x, S):
                S.add(x)
        
        # If we reached MAX_SIZE, verify
        if len(S) == MAX_SIZE:
            if is_full_valid(S):
                return list(S)
        
        # If not, try to improve?
        # Or just record best found so far
        if len(S) > len(best_set):
            best_set = S.copy()
        
        # If we didn't reach max size, we can try a local search
        # But for now, restart is simpler.
        
        # To increase diversity, we can bias the random selection
        # e.g. prefer numbers that are not close to each other?
        # Or just use different random seeds.
        
    # If we didn't find a size 48 set, return the best we found
    # We must ensure the returned set is valid!
    if best_set:
        if is_full_valid(best_set):
            return list(best_set)
        else:
            # Fallback: prune best_set until valid
            # Convert to list to allow removal
            S_list = list(best_set)
            # Simple pruning: remove elements that cause violations
            # This might be slow, but best_set is likely close to valid
            # We can just return a small known valid set if this fails
            pass
    
    # Fallback: Return a known valid set (e.g. powers of 2)
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
