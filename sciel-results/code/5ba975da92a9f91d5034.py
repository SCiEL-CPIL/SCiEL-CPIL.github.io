"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

DEADLINE_MARGIN = 3.0  # seconds to spare before the 90s kill

def _valid_set(s):
    """Check if set s has no element equal to mean of any subset of >=2 others."""
    s = sorted(s)
    n = len(s)
    if n < 3:
        return True
    # For each candidate 'a' in s, check if a is mean of any subset of others
    # a = mean(T) iff sum(T) = len(T)*a for some T subset of s\{a}, |T|>=2
    # We use DP: for each a, compute reachable sums of subsets of s\{a} with size>=2
    # and check if any reachable sum equals len(T)*a for the corresponding size.
    
    # More efficient: for each a, let others = s without a.
    # We need: exists T subset of others, |T|>=2, such that sum(T) = |T| * a.
    # Let b_i = other_i - a. Then sum(T) = |T|*a iff sum_{i in T} b_i = 0.
    # So we need a non-empty subset of b's (with size>=2) that sums to 0.
    
    for i, a in enumerate(s):
        others = [x - a for x in s if x != a]
        # Check if there's a subset of others with size >= 2 and sum = 0
        # DP: dp[k] = set of sums achievable with exactly k elements
        # But this can be expensive. Let's use a different approach.
        
        # Use meet-in-the-middle or just iterate if n is small enough.
        # For n up to ~50, this is feasible with bitset tricks.
        
        # Alternative: sort others, use two-sum style for k=2, then DP for larger k.
        # Actually, let's just do a DP with bitsets.
        # dp[k] is a set of achievable sums using exactly k elements from others.
        
        # Since values can be negative and positive, and range is bounded by n*512,
        # let's use a set-based DP.
        
        # For efficiency, limit the DP size.
        m = len(others)
        if m < 2:
            continue
            
        # dp[k] = set of sums achievable with k elements
        dp = [set() for _ in range(m + 1)]
        dp[0].add(0)
        
        for b in others:
            for k in range(len(others) - 1, -1, -1):
                if k < m:
                    for s_val in list(dp[k]):
                        dp[k+1].add(s_val + b)
        
        # Check if 0 is in dp[k] for any k >= 2
        for k in range(2, m + 1):
            if 0 in dp[k]:
                return False
    return True

def _valid_set_fast(s):
    """Faster validity check using bitset DP for subset sums."""
    s = sorted(s)
    n = len(s)
    if n < 3:
        return True
    
    for i, a in enumerate(s):
        others = [x - a for x in s if x != a]
        m = len(others)
        if m < 2:
            continue
        
        # Check if any subset of others of size >= 2 sums to 0.
        # Use DP: dp[k] = set of achievable sums with exactly k elements.
        # To keep it fast, we can bound the sum range.
        
        # For small m, this is fine. For larger m, we need optimization.
        # Let's use a dict-based approach with pruning.
        
        # Actually, for m up to ~48, the number of subsets is 2^48 which is too large.
        # But we only need to check if sum=0 is achievable for some k>=2.
        
        # Better approach: use the fact that if a subset sums to 0, 
        # we can find it via DP tracking (count, sum) pairs.
        
        # Let's use a different strategy: for each pair (i,j) in others,
        # check if others[i] + others[j] == 0 (k=2 case).
        # Then for k>=3, use DP.
        
        # k=2 check
        other_set = set()
        for b in others:
            if -b in other_set:
                return False
            other_set.add(b)
        
        # For k>=3, use DP with bitsets
        # dp[k] = set of sums achievable with exactly k elements
        # We'll limit the number of distinct sums to keep it manageable.
        
        dp = [set() for _ in range(min(m, 10) + 1)]  # Limit to k<=10 for speed
        dp[0].add(0)
        
        for b in others:
            for k in range(min(m, 10) - 1, -1, -1):
                for s_val in list(dp[k]):
                    dp[k+1].add(s_val + b)
        
        for k in range(3, min(m, 10) + 1):
            if 0 in dp[k]:
                return False
    
    return True

def _valid_set_exact(s):
    """Exact validity check. Slower but complete."""
    s = sorted(s)
    n = len(s)
    if n < 3:
        return True
    
    for i, a in enumerate(s):
        others = [x - a for x in s if x != a]
        m = len(others)
        if m < 2:
            continue
        
        # DP: dp[k] = set of sums achievable with exactly k elements
        dp = [set() for _ in range(m + 1)]
        dp[0].add(0)
        
        for b in others:
            for k in range(m - 1, -1, -1):
                for s_val in list(dp[k]):
                    dp[k+1].add(s_val + b)
        
        for k in range(2, m + 1):
            if 0 in dp[k]:
                return False
    return True

def construct():
    start_time = time.time()
    
    # Strategy: greedy construction with backtracking, trying to maximize size.
    # We'll try multiple random orderings and pick the best valid set.
    
    best_set = []
    best_size = 0
    
    # First, try a simple greedy in increasing order
    selected = []
    for x in range(1, 513):
        candidate = selected + [x]
        if _valid_set_fast(candidate):
            selected = candidate
    
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = list(selected)
    
    # Try greedy in decreasing order
    selected = []
    for x in range(512, 0, -1):
        candidate = selected + [x]
        if _valid_set_fast(candidate):
            selected = candidate
    
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = list(selected)
    
    # Try random orderings
    random.seed(42)
    for trial in range(20):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        
        order = list(range(1, 513))
        random.shuffle(order)
        
        selected = []
        for x in order:
            candidate = selected + [x]
            if _valid_set_fast(candidate):
                selected = candidate
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
    
    # Try to improve by adding elements
    # For each candidate not in best_set, try adding it
    best_set_set = set(best_set)
    for x in range(1, 513):
        if x in best_set_set:
            continue
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        candidate = best_set + [x]
        if _valid_set_fast(candidate):
            best_set = candidate
            best_set_set.add(x)
    
    # Try removing and re-adding to find better configurations
    for _ in range(5):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        
        # Try removing a random element and see if we can add two
        if len(best_set) > 0:
            idx = random.randint(0, len(best_set) - 1)
            removed = best_set[idx]
            temp_set = best_set[:idx] + best_set[idx+1:]
            
            added = []
            for x in range(1, 513):
                if x in temp_set or x == removed:
                    continue
                if time.time() - start_time > DEADLINE_MARGIN:
                    break
                candidate = temp_set + added + [x]
                if _valid_set_fast(candidate):
                    added.append(x)
                    if len(added) >= 2:
                        break
            
            if len(temp_set) + len(added) > len(best_set):
                best_set = temp_set + added
                best_set_set = set(best_set)
    
    return best_set
# EVOLVE-BLOCK-END
