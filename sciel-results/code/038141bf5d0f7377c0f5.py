import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    def subset_sums_bitset(elements):
        """
        Returns an integer bitset where bit i is 1 if i is a subset sum of elements.
        """
        sums = 1  # Bit 0 is 1 (empty sum)
        for x in elements:
            sums |= (sums << x)
        return sums

    def is_valid_set(A):
        """
        Checks if no element in A is the mean of any subset of at least 2 OTHER elements.
        """
        n = len(A)
        if n < 3:
            return True
        
        # Precompute sums for A
        # We need to check for each a in A:
        # Is k*a a subset sum of A \ {a} for any k in 2..n-1?
        
        # Optimization: 
        # We can compute total_sum.
        # But subset sums of A \ {a} are not easily derived from A's bitset 
        # without division (which isn't possible for bitsets).
        # So we recompute for each a.
        
        for i, a in enumerate(A):
            # Get others
            others = A[:i] + A[i+1:]
            sums_others = subset_sums_bitset(others)
            
            # Check if k*a is in sums_others for k=2..n-1
            # Max sum is roughly n * 512
            # We only need to check up to max(others) * len(others)
            # But k*a is the target.
            # a is at most 512. k is at most 47. k*a <= 24064.
            
            # Check bits
            valid = True
            for k in range(2, n):
                target = k * a
                # Check if bit 'target' is set in sums_others
                if (sums_others >> target) & 1:
                    valid = False
                    break
            if not valid:
                return False
        
        return True

    def greedy_construct(order):
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            candidate = selected + [x]
            if is_valid_set(candidate):
                selected = candidate
        return selected

    # Generate candidate orderings
    candidates = list(range(1, 513))
    
    best_set = []
    
    # Strategy 1: 1 to 512
    s1 = greedy_construct(candidates)
    if len(s1) > len(best_set):
        best_set = s1

    # Strategy 2: 512 to 1
    s2 = greedy_construct(candidates[::-1])
    if len(s2) > len(best_set):
        best_set = s2

    # Strategy 3: Sorted by popcount, then value
    def popcount(x):
        return bin(x).count('1')
        
    s3_order = sorted(candidates, key=lambda x: (popcount(x), x))
    s3 = greedy_construct(s3_order)
    if len(s3) > len(best_set):
        best_set = s3
        
    # Strategy 4: Sorted by popcount desc, then value
    s4_order = sorted(candidates, key=lambda x: (-popcount(x), x))
    s4 = greedy_construct(s4_order)
    if len(s4) > len(best_set):
        best_set = s4

    # Strategy 5: Randomized restarts if time permits
    # We can try a few random permutations
    # But 512! is huge. Just shuffle.
    # If current best is small, try harder.
    
    # Let's try to improve by local search?
    # Or just more random shuffles.
    
    # The seed score was 1.0. 
    # If seed size is ~10, we want > 10.
    # Powers of 2 give 9.
    # If we can get 20+, that's a big jump.
    
    # Let's run a few random shuffles
    for _ in range(5):
        if time.time() > deadline:
            break
        shuffled = candidates[:]
        random.shuffle(shuffled)
        s_rand = greedy_construct(shuffled)
        if len(s_rand) > len(best_set):
            best_set = s_rand
            
    return best_set

# Ensure no global state issues
random.seed(42)
