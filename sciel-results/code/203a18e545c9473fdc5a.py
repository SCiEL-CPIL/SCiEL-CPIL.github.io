import time
import random
from itertools import combinations

def get_subset_sums(S):
    """Return a set of all non-empty subset sums."""
    sums = set()
    for i in range(1, 1 << len(S)):
        s = 0
        for j in range(len(S)):
            if i & (1 << j):
                s += S[j]
        sums.add(s)
    return sums

def check_divisibility(sums):
    """Check if any two distinct sums in the set divide one another."""
    L = list(sums)
    L.sort()
    # Optimized check: only need to check if a smaller divides a larger.
    # Since L is sorted, for each i, check j > i if L[j] % L[i] == 0.
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            if L[j] % L[i] == 0:
                return False
    return True

def is_valid(S):
    """
    Check if set S is valid.
    1. All subset sums must be distinct (handled by using a set, but we need to ensure no two subsets give same sum).
       Actually, if we just use a set of sums, we lose the information if two subsets gave the same sum.
       The rule says: "Two different subsets with the same sum still make the set invalid".
       So we must check that all $2^k - 1$ sums are distinct.
    2. No two distinct resulting values may divide one another.
    """
    n = len(S)
    if n == 0:
        return True
    sums = []
    # Generate all subset sums
    for i in range(1, 1 << n):
        s = 0
        for j in range(n):
            if i & (1 << j):
                s += S[j]
        sums.append(s)
    
    # Check for duplicates
    if len(sums) != len(set(sums)):
        return False
        
    # Check divisibility
    return check_divisibility(set(sums))

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 55.0  # Keep 5s margin for Python overhead/return
    
    # We want to maximize size.
    # Try sizes from high to low.
    # Max possible size?
    # In [1, 256], subset sums max out at 256*k.
    # Let's try to find size 6, 7, 8.
    
    best_set = [31, 32, 34]  # Fallback to seed
    
    # Greedy randomized construction
    # To make it faster, we can use an incremental validity check.
    # But for simplicity and correctness within time, we can use a helper that checks validity of a candidate set.
    # However, full is_valid is $O(2^k \cdot k)$ to generate sums and $O(2^{2k})$ to check divisibility naively.
    # For k=6, 64 sums. $64^2 = 4096$ checks. Very fast.
    # For k=8, 256 sums. $256^2 = 65536$ checks. Still fast.
    # So we can just call is_valid on candidate sets.
    
    # Strategy:
    # 1. Generate random sets of size K.
    # 2. Check if valid.
    # 3. If valid, return it.
    # 4. Decrease K if not found in time.
    
    # To improve chances, bias towards larger numbers.
    # Numbers in [128, 256] are safer.
    
    def try_find(k, max_attempts=1000):
        nonlocal best_set
        # Use numbers from 100 to 256 to increase chance of non-divisibility
        # Smaller numbers create smaller sums which are more likely to divide larger sums.
        # Actually, if sums are large, divisibility is less likely unless they are multiples.
        
        for _ in range(max_attempts):
            if time.time() - start_time > time_limit:
                return None
            # Sample k distinct integers
            # Bias: prefer larger numbers
            # We can sample from [50, 256]
            candidates = random.sample(range(50, 257), k)
            if is_valid(candidates):
                return candidates
        return None

    # Try sizes 8, 7, 6, 5, 4
    # We prioritize larger sizes.
    
    # First, let's try a deterministic greedy approach which is often faster to find a valid set than pure random.
    # Greedy: add largest available number if it doesn't break validity.
    
    def greedy_construction(k_target):
        # Start with empty set
        S = []
        # Candidates sorted descending
        candidates = list(range(256, 0, -1))
        random.shuffle(candidates)
        
        # To be more effective, we can try multiple shuffles
        # But simple greedy on large numbers often works.
        
        # Let's refine: 
        # Iterate through a sorted list of candidates (e.g. descending).
        # If adding c keeps S valid, add it.
        # Stop when size is k_target.
        
        S = []
        for c in candidates:
            if len(S) == k_target:
                break
            new_S = S + [c]
            if is_valid(new_S):
                S = new_S
        return S if len(S) == k_target else None

    # Try target sizes from 8 down to 4
    for k in range(8, 3, -1):
        # Try a few times with different random seeds/shuffles
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                return best_set
            # Use a specific bias for random sampling in try_find, 
            # but greedy_construction is also good.
            # Let's mix them.
            
            # 1. Try Greedy with random candidate order
            res = greedy_construction(k)
            if res:
                return res
                
            # 2. Try Random Sampling
            res = try_find(k, max_attempts=50)
            if res:
                return res
        
    # If all else fails, return the known valid size 3 set
    return best_set

# EVOLVE-BLOCK-START
import time
import random

def get_subset_sums(S):
    """Return a set of all non-empty subset sums."""
    sums = set()
    for i in range(1, 1 << len(S)):
        s = 0
        for j in range(len(S)):
            if i & (1 << j):
                s += S[j]
        sums.add(s)
    return sums

def check_divisibility(sums):
    """Check if any two distinct sums in the set divide one another."""
    L = list(sums)
    L.sort()
    # Optimized check: only need to check if a smaller divides a larger.
    # Since L is sorted, for each i, check j > i if L[j] % L[i] == 0.
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            if L[j] % L[i] == 0:
                return False
    return True

def is_valid(S):
    """
    Check if set S is valid.
    1. All subset sums must be distinct.
    2. No two distinct resulting values may divide one another.
    """
    n = len(S)
    if n == 0:
        return True
    sums = []
    # Generate all subset sums
    for i in range(1, 1 << n):
        s = 0
        for j in range(n):
            if i & (1 << j):
                s += S[j]
        sums.append(s)
    
    # Check for duplicates
    if len(sums) != len(set(sums)):
        return False
        
    # Check divisibility
    return check_divisibility(set(sums))

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 55.0  # Keep 5s margin for Python overhead/return
    
    best_set = [31, 32, 34]  # Fallback to seed
    
    def try_find(k, max_attempts=1000):
        # Use numbers from 100 to 256 to increase chance of non-divisibility
        for _ in range(max_attempts):
            if time.time() - start_time > time_limit:
                return None
            # Sample k distinct integers
            # Bias: prefer larger numbers
            candidates = random.sample(range(50, 257), k)
            if is_valid(candidates):
                return candidates
        return None

    def greedy_construction(k_target):
        S = []
        candidates = list(range(256, 0, -1))
        random.shuffle(candidates)
        
        for c in candidates:
            if len(S) == k_target:
                break
            new_S = S + [c]
            if is_valid(new_S):
                S = new_S
        return S if len(S) == k_target else None

    # Try sizes 8, 7, 6, 5, 4
    for k in range(8, 3, -1):
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                return best_set
            
            # 1. Try Greedy with random candidate order
            res = greedy_construction(k)
            if res:
                return res
                
            # 2. Try Random Sampling
            res = try_find(k, max_attempts=50)
            if res:
                return res
        
    return best_set
# EVOLVE-BLOCK-END
