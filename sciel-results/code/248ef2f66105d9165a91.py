"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums must differ.
    
    We maximize the set size. The constraint is:
    For all k != l, the set of sums of k-element subsets and l-element subsets are disjoint.
    
    Strategy:
    1. Start with a small set that is known to work (e.g., powers of 2 or similar).
    2. Try to add elements one by one, checking if the constraint is maintained.
    3. Use dynamic programming to track achievable sums for each cardinality.
    
    The key insight: if we use elements that are "far apart" in magnitude, subsets of 
    different sizes will tend to have different sums. Specifically, if each element is 
    larger than the sum of all previous elements, then the sum of any k-element subset 
    is strictly increasing with k (in a strong sense).
    
    However, we need distinct integers in 1..256, so we can't use powers of 2 beyond 2^7=128.
    
    Let's think about this differently. We want sets S_k (sums of k-element subsets) to be
    pairwise disjoint for different k.
    
    A simple construction: use elements that are all in a range where the minimum sum of 
    (k+1) elements is greater than the maximum sum of k elements. This is achieved if 
    each element is large enough.
    
    Specifically, if we have elements a_1 < a_2 < ... < a_n, and we require:
    a_{k+1} > sum of the k largest previous elements... no, that's too strong.
    
    Actually, the condition is: min sum of (k+1) elements > max sum of k elements.
    min sum of (k+1) elements = sum of the (k+1) smallest elements.
    max sum of k elements = sum of the k largest elements.
    
    So we need: sum of (k+1) smallest > sum of k largest, for all k from 0 to n-1.
    
    This is a very strong condition. Let's see what sets satisfy this.
    
    For k=0: min sum of 1 element > max sum of 0 elements = 0. So min element > 0. Always true.
    For k=1: min sum of 2 elements > max sum of 1 element. So a_1 + a_2 > a_n.
    For k=2: a_1 + a_2 + a_3 > a_{n-1} + a_n.
    ...
    
    This is hard to satisfy for large n with elements in 1..256.
    
    Alternative approach: use a greedy algorithm with backtracking, or use a randomized 
    search with the DP check.
    
    Let's implement a greedy approach:
    1. Start with an empty set.
    2. Try adding elements from 1 to 256 in some order (or randomized order).
    3. For each candidate element, check if adding it maintains the disjointness constraint.
    4. Use DP to track sums: sums[k] is a bitset (or set) of achievable sums with k elements.
    
    Since we have at most 60 elements and sums up to 60*256=15360, we can use bitsets.
    
    Actually, let's think about the maximum possible size. If we use elements that are 
    "spread out", we might get a decent size.
    
    Let's try a different construction: use elements from a geometric progression or 
    similar. But we're limited to 1..256.
    
    Let's just implement the greedy with DP check and see how far we get.
    """
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    def check_validity(sums, n):
        """Check if sums[k] bitsets are pairwise disjoint for k=0..n."""
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums[i] & sums[j]:
                    return False
        return True
    
    def add_element(sums, n, x):
        """
        Try to add element x to a set of size n.
        sums[k] is a bitset where bit s is 1 if sum s is achievable with k elements.
        Returns new sums and n+1 if valid, else None.
        """
        max_sum = 60 * 256  # Upper bound on sums
        new_sums = [0] * (n + 2)
        for k in range(n + 1):
            new_sums[k] = sums[k]
            new_sums[k + 1] |= (sums[k] << x)
        
        # Mask to keep only relevant bits
        mask = (1 << (max_sum + 1)) - 1
        for k in range(n + 2):
            new_sums[k] &= mask
        
        if check_validity(new_sums, n + 1):
            return new_sums, n + 1
        return None
    
    # Initialize with empty set
    max_sum = 60 * 256
    sums = [0] * 61
    sums[0] = 1  # Empty subset has sum 0
    n = 0
    used = set()
    
    # Try adding elements in increasing order first
    for x in range(1, 257):
        if time.time() - start_time > deadline:
            break
        result = add_element(sums, n, x)
        if result is not None:
            sums, n = result
            used.add(x)
    
    # If we haven't reached a good size, try randomized search
    if n < 15:
        random.seed(42)
        for _ in range(100):
            if time.time() - start_time > deadline:
                break
            # Reset
            sums = [0] * 61
            sums[0] = 1
            n = 0
            used = set()
            
            # Random order
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() - start_time > deadline:
                    break
                result = add_element(sums, n, x)
                if result is not None:
                    sums, n = result
                    used.add(x)
            
            if n > 0:
                break
    
    # Convert to list of ints
    result = sorted(used)
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
