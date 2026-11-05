"""Search-based solver for the 'no element is mean of subset of others' problem."""
import time
import random

def _valid_mean_free(selected):
    """Returns True if the list 'selected' satisfies the condition that no element
    is the exact arithmetic mean of any subset of at least two OTHER elements."""
    n = len(selected)
    if n < 3:
        return True
    
    # Remove duplicates just in case, though we generate distinct ones
    selected = list(set(selected))
    
    for i, a in enumerate(selected):
        others = [x for x in selected if x != a]
        m = len(others)
        if m < 2:
            continue
            
        # DP: dp[k] is a bitset where bit s is set if sum s is achievable with exactly k elements from 'others'
        # We initialize dp[0] with bit 0 set.
        # Max possible sum is sum(others).
        # We only care about sums up to m*a (since target is k*a with k<=m).
        # Actually, we can just compute all sums.
        
        dp = [0] * (m + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for x in others:
            # Update in reverse order of k to avoid reusing x multiple times
            for k in range(m, 0, -1):
                # Shift the bitset of dp[k-1] by x and OR into dp[k]
                dp[k] |= (dp[k-1] << x)
                
        # Check if any k in [2, m] has a target sum k*a present in dp[k]
        for k in range(2, m + 1):
            target = k * a
            # Check if bit 'target' is set in dp[k]
            if dp[k] & (1 << target):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 60s limit
    best_set = []
    
    # Strategy 1: Greedy construction (fast to get a baseline)
    # Try to build a set greedily by picking numbers that don't violate the condition.
    # To speed up, we can just pick numbers in a specific order and check validity of the *growing* set.
    # But checking validity of the whole set each time is O(N^3) roughly.
    # With N=48, it might be okay if we do it a few times.
    
    # Let's try a few different greedy orders.
    orders = [
        list(range(1, 513)),
        list(range(512, 0, -1)),
        list(range(1, 513, 2)), # Odd
        list(range(2, 513, 2)), # Even
    ]
    
    # Add some random orders
    for _ in range(5):
        lst = list(range(1, 513))
        random.shuffle(lst)
        orders.append(lst)
        
    for order in orders:
        if time.time() > deadline:
            break
        current = []
        for x in order:
            if time.time() > deadline:
                break
            candidate = current + [x]
            if _valid_mean_free(candidate):
                current = candidate
                if len(current) > len(best_set):
                    best_set = current[:]
    
    # Strategy 2: Random search + Local Improvement
    # Generate random sets of size K and check validity.
    # If valid, try to add more elements.
    
    max_val = 512
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        # Start with a random small set
        size = random.randint(5, 15)
        current = random.sample(range(1, max_val + 1), size)
        
        # Check if this base set is valid
        if not _valid_mean_free(current):
            continue
            
        # Try to expand
        remaining = list(set(range(1, max_val + 1)) - set(current))
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline:
                break
            candidate = current + [x]
            if _valid_mean_free(candidate):
                current = candidate
        
        if len(current) > len(best_set):
            best_set = current[:]

    # Strategy 3: Hill Climbing / Local Search on the best set found so far
    # Try to swap elements or replace elements to increase size? 
    # Actually, we want to *maximize* size. If we have a valid set of size K, 
    # can we find a valid set of size K+1?
    # We can try to add any missing element.
    if best_set:
        best_set_set = set(best_set)
        missing = [x for x in range(1, 513) if x not in best_set_set]
        random.shuffle(missing)
        
        # Try to add one element
        improved = False
        for x in missing:
            if time.time() > deadline:
                break
            candidate = best_set + [x]
            if _valid_mean_free(candidate):
                best_set = candidate
                improved = True
                # After adding one, re-evaluate missing
                best_set_set = set(best_set)
                missing = [x for x in range(1, 513) if x not in best_set_set]
                random.shuffle(missing)
                # Continue trying to add more?
                # We can loop this addition process until no more can be added.
                while improved and time.time() < deadline:
                    improved = False
                    for x in missing:
                        if time.time() > deadline:
                            break
                        candidate = best_set + [x]
                        if _valid_mean_free(candidate):
                            best_set = candidate
                            best_set_set = set(best_set)
                            missing = [x for x in range(1, 513) if x not in best_set_set]
                            random.shuffle(missing)
                            improved = True
                            break
    
    return best_set

# Quick self-test if run directly
if __name__ == "__main__":
    import sys
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    # Verify
    assert _valid_mean_free(s), "Invalid set found!"
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"
