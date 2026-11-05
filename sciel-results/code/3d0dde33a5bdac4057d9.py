"""Optimized constructor for mean-free subset problem."""
import time
import numpy as np

def _is_valid_candidate(a, others):
    """Check if a is NOT the mean of any subset of others with size >= 2.
    Mean condition: sum(S) = len(S)*a for some S subset of others, |S|>=2.
    Equivalent: exists non-empty subset T of (x-a for x in others) with sum(T)=0 and |T|>=2.
    """
    n = len(others)
    if n < 2:
        return True
    diffs = np.array([x - a for x in others], dtype=np.int64)
    # We need subset of diffs with sum 0 and size >= 2
    # Track possible (sum, count) pairs
    # Use a set of (sum, min_count) - but that's expensive
    # Alternative: track for each sum the min and max count achievable
    # Actually, let's track: for each sum, the set of achievable counts (as bitset)
    # But sums range from -512*n to 512*n which is large
    
    # Better approach: track achievable sums for each count separately
    # For count k, track set of achievable sums
    # But we need count >= 2
    
    # Use DP: dp[k] = set of achievable sums using exactly k elements
    # This is O(n^2 * range) which might be too slow for large n
    
    # Alternative: use the fact that we just need ANY subset with sum 0 and size >= 2
    # This is equivalent to: the multiset of diffs has a zero-sum sub-multiset of size >= 2
    # Note: if any diff is 0, that's a single element with sum 0, but we need size >= 2
    # So we need either: two elements summing to 0, or three, etc.
    
    # Let's use a different approach: track achievable (sum, parity_of_count) or just count
    # Actually, let's just do: for each possible count from 2 to n, check if 0 is achievable
    
    # More efficient: use a dict mapping sum -> bitset of achievable counts
    # But this could be large. Let's bound it.
    
    # Since values are in [-511, 511] and n <= 47, sums are in [-24017, 24017]
    # That's about 48035 possible sums. With bitsets for counts (up to 47 bits), this is feasible.
    
    # dp[sum_offset] = bitset where bit k is set if sum is achievable with exactly k elements
    offset = 512 * (n + 1)  # safe offset
    size = 2 * offset + 1
    dp = np.zeros(size, dtype=np.uint64)
    dp[offset] = 1  # empty subset: sum 0, count 0
    
    for d in diffs:
        # Update dp: new_dp[s] |= dp[s] | (dp[s-d] << 1)
        # We need to be careful about the shift
        if d >= 0:
            # Shift right by d
            for s in range(offset + d, size - 1, -1):
                if dp[s - d] != 0:
                    dp[s] |= (dp[s - d] << 1)
        else:
            # Shift left by -d
            dd = -d
            for s in range(offset - dd, -1, -1):
                if dp[s + dd] != 0:
                    dp[s] |= (dp[s + dd] << 1)
    
    # Check if sum 0 is achievable with count >= 2
    # dp[offset] has bits set for achievable counts
    if dp[offset] & 0b110:  # bits 1 and above (count >= 1, but we need >= 2)
        # Actually bit 0 is count 0, bit 1 is count 1, bit 2 is count 2, etc.
        # We need count >= 2, so bits 2 and above
        if dp[offset] & ~0b11:  # mask out bits 0 and 1
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # Greedy approach: try to add elements in a way that maximizes count
    # Strategy: start with a good base set, then try to add more elements
    
    # First, let's try a simple greedy: add elements from 1 to 512 if valid
    # But this might not be optimal. Let's try multiple orderings.
    
    def try_greedy(order, time_limit=80):
        selected = []
        for x in order:
            if time.time() > time_limit:
                break
            # Check if x can be added
            valid = True
            # Check x against existing elements
            if selected:
                if not _is_valid_candidate(x, selected):
                    valid = False
            # Check existing elements against x
            if valid and selected:
                for a in selected:
                    others = [y for y in selected if y != a] + [x]
                    if not _is_valid_candidate(a, others):
                        valid = False
                        break
            if valid:
                selected.append(x)
        return selected
    
    # Try different orderings
    best = []
    
    # Ordering 1: natural order 1 to 512
    if time.time() < deadline:
        result = try_greedy(range(1, 513), deadline)
        if len(result) > len(best):
            best = result
    
    # Ordering 2: reverse order
    if time.time() < deadline:
        result = try_greedy(range(512, 0, -1), deadline)
        if len(result) > len(best):
            best = result
    
    # Ordering 3: powers of 2 first, then others
    if time.time() < deadline:
        powers = [2**i for i in range(1, 10) if 2**i <= 512]
        rest = [x for x in range(1, 513) if x not in set(powers)]
        order = powers + rest
        result = try_greedy(order, deadline)
        if len(result) > len(best):
            best = result
    
    # Ordering 4: large numbers first
    if time.time() < deadline:
        order = sorted(range(1, 513), key=lambda x: -x)
        result = try_greedy(order, deadline)
        if len(result) > len(best):
            best = result
    
    # If we haven't reached 48, try a more targeted approach
    # The condition is very restrictive. Let's try to build a set that avoids
    # creating arithmetic progressions or other mean structures.
    
    if len(best) < 48 and time.time() < deadline:
        # Try a randomized approach with multiple restarts
        import random
        random.seed(42)
        
        def try_random_greedy(time_limit):
            order = list(range(1, 513))
            random.shuffle(order)
            return try_greedy(order, time_limit)
        
        for i in range(10):
            if time.time() > deadline - 5:
                break
            remaining = deadline - time.time() - 5
            if remaining < 1:
                break
            result = try_random_greedy(remaining / 12)
            if len(result) > len(best):
                best = result
    
    return best[:48]
