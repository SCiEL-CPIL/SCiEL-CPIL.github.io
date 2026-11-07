import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    def try_greedy_build(order):
        S = []
        n = 0
        # dp[k] is a bitset of sums
        # Max sum estimate: 256 * 60 = 15360. 
        # Let's allocate enough bits. 
        # Python ints are arbitrary precision, so no limit.
        dp = [0] * 61  # Bitsets for cardinalities 0..60
        dp[0] = 1  # Sum 0 exists
        
        # all_sums is a bitset of all possible sums so far
        all_sums = 1  # Sum 0
        
        for x in order:
            if x in S:
                continue
            
            # Check if x can be added
            valid = True
            # We need to check for k in 1..n+1
            # new_sums_k = dp[k] | (dp[k-1] << x)
            # We need to ensure new_sums_k does not intersect with all_sums \ dp[k]
            # AND new_sums_k does not intersect with new_sums_j for j != k?
            # Wait, the condition is S'_k cap S'_j = empty.
            # S'_k = dp[k] U (dp[k-1] << x)
            # S'_j = dp[j] U (dp[j-1] << x)
            
            # Let's just compute the new dp and check all pairs?
            # That is O(n^2) bit operations.
            # Or check incrementally.
            
            # If we add x, the new sums are formed.
            # Let's compute the full new_dp first to check validity.
            new_dp = [0] * 61
            for k in range(n + 1):
                # Old sums
                new_dp[k] = dp[k]
                # New sums from k-1
                if k > 0:
                    new_dp[k] |= (dp[k-1] << x)
            
            # Now check if new_dp[k] and new_dp[j] are disjoint for k != j
            # We can check this by iterating k and j?
            # Or just check if any new bit is set in multiple new_dp[k]?
            # No, we need to check intersection.
            
            # Efficient check:
            # Union of all new_dp[k] should have the same popcount as sum of popcounts of new_dp[k].
            # i.e., no bit is set in more than one new_dp[k].
            
            union_bits = 0
            total_popcount = 0
            for k in range(n + 2):
                bits = new_dp[k]
                if bits:
                    if union_bits & bits:
                        valid = False
                        break
                    union_bits |= bits
                    total_popcount += bits.bit_count()
            
            # Also check if total_popcount == union_bits.bit_count()
            # This is equivalent to no overlap.
            
            if valid and union_bits.bit_count() == total_popcount:
                # Commit
                S.append(x)
                n += 1
                dp = new_dp
                all_sums = union_bits # Not strictly needed if we use dp for checks, but good for reference
                
        return S

    best_set = []
    
    # Generate diverse orders
    # 1. Sorted
    orders = []
    orders.append(list(range(1, 257)))
    orders.append(list(range(256, 0, -1)))
    
    # 2. Random
    # Run many random orders
    # Each order takes ~256 steps.
    # Bit operations are very fast.
    # We can run thousands of orders.
    
    num_random = 500
    for i in range(num_random):
        if time.time() > deadline:
            break
        o = list(range(1, 257))
        random.shuffle(o)
        orders.append(o)
        
    # 3. Biased orders
    # Small first
    o = list(range(1, 257))
    random.shuffle(o)
    o.sort(key=lambda x: x) # Just sorted again
    orders.append(o)
    
    # Mix of small and large
    for i in range(50):
        if time.time() > deadline:
            break
        o = list(range(1, 257))
        random.shuffle(o)
        # Try to interleave small and large?
        orders.append(o)
        
    for o in orders:
        if time.time() > deadline:
            break
        S = try_greedy_build(o)
        if len(S) > len(best_set):
            best_set = S
            # Update best
            if len(best_set) >= 40:
                # Early exit if we find a very good one?
                # Or continue to find even better?
                pass

    return [int(x) for x in best_set]
