import random
import time
from itertools import combinations

def is_sidon(subset):
    """Check if a tuple/list of integers is a Sidon set (unique pair sums)."""
    sums = set()
    n = len(subset)
    # Optimization: sort subset? Not strictly needed for sum check, but good for consistency.
    # The set of sums {x+y | x<=y} is what matters.
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_subset(S, size):
    """Check if any subset of S of given size is Sidon."""
    # S is a list of distinct integers
    for comb in combinations(S, size):
        if is_sidon(comb):
            return True
    return False

def get_max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    # We know max size is at least 3 for any set of distinct integers > 2
    # that is not all 3-APs (which is impossible for size > 2).
    # We check down from 12.
    # But for optimization, we primarily care if it's < 4.
    # If we find a Sidon subset of size 4, we can stop and return 4 (or higher).
    # If we don't find any of size 4, we check size 3.
    
    # Check size 4 first. If exists, return 4 (conservative, actual might be higher, but 4 is "bad")
    if has_sidon_subset(S, 4):
        # It could be 5, 6, etc. But 4 is already worse than 3.
        # For the purpose of finding a k=3 set, we just need to know if k>=4.
        # However, if we want to compare two "bad" sets, we might want the exact size.
        # But we just need to find ONE good set.
        # So if has_sidon_subset(S, 4) is True, this set is "bad" (score >= 4).
        # If False, it is "good" (score = 3).
        return 4 # Placeholder, actual is >=4
        
    # If no Sidon subset of size 4, max size is <= 3.
    # Check if there is a Sidon subset of size 3.
    if has_sidon_subset(S, 3):
        return 3
    else:
        return 2 # Theoretical minimum, unlikely

def construct():
    start_time = time.time()
    time_limit = 50.0 # Leave 10s for safety
    
    best_set = None
    best_score = float('inf')
    
    # Initial random guesses
    # Bias towards smaller ranges to increase collisions
    ranges_to_try = [
        list(range(0, 30)),
        list(range(0, 50)),
        list(range(0, 80)),
        list(range(0, 129))
    ]
    
    while time.time() - start_time < time_limit:
        # Pick a range strategy
        # 50% chance small range, 30% medium, 20% full
        r = random.random()
        if r < 0.5:
            pool = list(range(0, 30))
        elif r < 0.8:
            pool = list(range(0, 60))
        else:
            pool = list(range(0, 129))
            
        # Ensure pool has at least 12 elements
        if len(pool) < 12:
            pool = list(range(0, 129))
            
        S = random.sample(pool, 12)
        
        # Evaluate
        # Fast check: Does it have a Sidon subset of size 4?
        if has_sidon_subset(S, 4):
            # This set has k >= 4. 
            # We can calculate exact k if we want to track "least bad", 
            # but finding a k=3 set is the goal.
            # Let's just record it as score 4 for now.
            # If we find a k=3 set, we return it immediately.
            # If we don't, we return the best k>=4 set.
            # To differentiate between k=4 and k=5, we'd need more checks.
            # But k=4 is the best "bad" score we can hope for if we fail to find k=3.
            # So we can just treat any k>=4 as "bad".
            continue
        
        # If we get here, S has NO Sidon subset of size 4.
        # This means max Sidon size is <= 3.
        # It is almost certainly 3.
        # We can return this immediately as it's the optimal target.
        return S

    # If loop finishes without finding a k=3 set, return a random one or the best tracked.
    # Since we didn't track specific "bad" sets in the loop above (we just continued),
    # we should pick a random one or a structured one.
    # Actually, let's pick a set from a very dense range, e.g., 0..15.
    # Max Sidon in 0..15 is likely 4.
    return random.sample(range(0, 20), 12)

# EVOLVE-BLOCK-START
# The code above needs to be inside the block or the construct function needs to be defined.
# The prompt says "Keep the function names...".
# I will place the helper functions and construct in the block.
# EVOLVE-BLOCK-END
