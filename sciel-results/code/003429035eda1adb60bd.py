import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 58

    def try_greedy_build(order):
        S = []
        n = 0
        # dp[k] is a bitset (Python int) where bit s is set iff sum s is achievable
        # by a subset of cardinality k from S.
        # Max possible sum: 60 * 256 = 15360. We use arbitrary precision ints.
        dp = [0] * 61
        dp[0] = 1  # bit 0 set: sum 0 with cardinality 0
        
        # all_sums is a bitset of all sums achievable by any subset of S.
        all_sums = 1  # bit 0 set
        
        for x in order:
            if x in S:
                continue
            
            # Check if x can be added
            # New dp': dp'[k] = dp[k] | (dp[k-1] << x)
            # Condition: for all k != j, dp'[k] & dp'[j] == 0
            # Since dp[k] & dp[j] == 0 for k != j (invariant), we need:
            # (dp[k] | (dp[k-1] << x)) & (dp[j] | (dp[j-1] << x)) == 0 for k != j
            # This expands to:
            # dp[k] & dp[j] == 0 (true)
            # dp[k] & (dp[j-1] << x) == 0
            # (dp[k-1] << x) & dp[j] == 0
            # (dp[k-1] << x) & (dp[j-1] << x) == 0  => dp[k-1] & dp[j-1] == 0 (true if k-1 != j-1)
            
            # So we need to check for all k, j with k != j:
            # 1. dp[k] & (dp[j-1] << x) == 0
            # 2. (dp[k-1] << x) & dp[j] == 0
            
            # A simpler sufficient and necessary condition derived from the invariant:
            # Let new_sums_k = dp[k-1] << x.
            # We need new_sums_k to be disjoint from (all_sums \ dp[k]).
            # Because dp[k] already contains valid sums for cardinality k.
            # If new_sums_k has any bit set that is in all_sums but not in dp[k], it's a conflict.
            
            valid = True
            # We also need to ensure that new_sums_k for different k don't collide with each other?
            # No, the condition is: sums of DIFFERENT cardinalities must differ.
            # So dp'[k] and dp'[j] must be disjoint.
            # dp'[k] = dp[k] | new_sums_k.
            # We know dp[k] and dp[j] are disjoint.
            # We need:
            # 1. dp[k] & new_sums_j == 0 for k != j
            # 2. new_sums_k & dp[j] == 0 for k != j
            # 3. new_sums_k & new_sums_j == 0 for k != j
            
            # Let's check these.
            # new_sums_k = dp[k-1] << x.
            
            # Precompute new_sums for all k
            new_sums = [0] * (n + 2)
            for k in range(1, n + 2):
                if k - 1 < len(dp):
                    new_sums[k] = dp[k-1] << x
            
            # Check 3: new_sums_k & new_sums_j == 0 for k != j
            # new_sums_k & new_sums_j = (dp[k-1] << x) & (dp[j-1] << x) = (dp[k-1] & dp[j-1]) << x
            # This is 0 if dp[k-1] & dp[j-1] == 0.
            # Since k != j, k-1 != j-1.
            # By invariant, dp[a] & dp[b] == 0 for a != b.
            # So Check 3 is automatically satisfied.
            
            # Check 1: dp[k] & new_sums_j == 0 for k != j
            # Check 2: new_sums_k & dp[j] == 0 for k != j
            # Note that Check 1 for pair (k, j) is equivalent to Check 2 for pair (j, k).
            # So we just need to check: for all k, j with k != j, dp[k] & (dp[j-1] << x) == 0.
            
            # Efficient check:
            # For each k, let A_k = dp[k].
            # Let B = union of (dp[j-1] << x) for all j != k? No, that's not right.
            # We need to check if A_k intersects with any (dp[j-1] << x) where j != k.
            
            # Alternative:
            # Compute total_new = OR of all new_sums[k].
            # This doesn't help directly with the disjointness from dp[k].
            
            # Let's just iterate k and j? n is small (up to ~20).
            # But we can be smarter.
            # Condition: For all k, (dp[k] & (OR_{j != k} (dp[j-1] << x))) == 0.
            
            # Let's compute shifted_dp[i] = dp[i] << x for i in 0..n.
            # Then new_sums[k] = shifted_dp[k-1].
            # We need dp[k] & (OR_{j != k} shifted_dp[j-1]) == 0.
            
            # Let's compute OR_all_shifted_except_k.
            # This is a bit expensive if done naively for each k.
            
            # Simpler:
            # For each k, check if dp[k] & (OR of all shifted_dp[i] for i != k-1) is 0.
            
            # Actually, let's just check the original condition:
            # For each k from 1 to n+1:
            #   conflict = new_sums[k] & (all_sums \ dp[k])
            #   if conflict != 0: valid = False
            # Wait, is this sufficient?
            # If new_sums[k] has a bit in all_sums that is not in dp[k], it means that sum is achieved by some other cardinality j.
            # So yes, this ensures dp'[k] is disjoint from dp'[j] for all j != k.
            
            # Let's verify:
            # dp'[k] = dp[k] | new_sums[k].
            # If new_sums[k] & (all_sums \ dp[k]) == 0, then new_sums[k] is subset of dp[k] OR (complement of all_sums).
            # Since new_sums[k] is subset of all_sums' (the new all_sums), and we are checking against old all_sums...
            # Actually, new_sums[k] might contain sums not in old all_sums. Those are fine.
            # The conflict is if new_sums[k] contains a sum that is in old all_sums but NOT in dp[k].
            # Because if it is in dp[k], it's already there (duplicate sum within same cardinality is allowed? 
            # "Repeated sums within the SAME cardinality are allowed."
            # So dp[k] can have multiple subsets with same sum. The bitset just tracks if sum is achievable.
            # So if new_sums[k] has a bit that is in dp[k], it's fine.
            # If it has a bit in all_sums but not dp[k], it conflicts with some dp[j], j!=k.
            
            # So the check is:
            # for k in 1..n+1:
            #    if (new_sums[k] & (all_sums & ~dp[k])) != 0:
            #        valid = False
            #        break
            
            # Note: ~dp[k] in Python is infinite bits 1. We need to mask with all_sums.
            # all_sums & ~dp[k] gives bits in all_sums that are NOT in dp[k].
            
            for k in range(1, n + 2):
                if k - 1 >= len(dp):
                    continue
                ns = dp[k-1] << x
                # Check conflict
                # forbidden = all_sums & ~dp[k]
                # if ns & forbidden:
                #    valid = False
                #    break
                if ns & (all_sums & ~dp[k]):
                    valid = False
                    break
            
            if valid:
                # Commit x
                S.append(x)
                n += 1
                
                # Update dp
                # dp[k] = dp[k] | (dp[k-1] << x)
                # Must do in reverse order to avoid using updated dp[k-1]
                # Or use a copy.
                # Since n is small, copy is fine.
                old_dp = dp[:]
                for k in range(n, 0, -1):
                    dp[k] = old_dp[k] | (old_dp[k-1] << x)
                # dp[0] remains 1
                
                # Update all_sums
                # all_sums = OR of all dp[k]
                # Or all_sums |= (old_all_sums << x)? No, that's not right.
                # all_sums' = all_sums | (OR of new_sums)
                # new_sums = (OR of old_dp) << x = all_sums << x
                # So all_sums |= (all_sums << x)
                all_sums |= (all_sums << x)
                
                # Optimization: Mask all_sums to reasonable range?
                # Max sum is 256 * n. 
                # We can mask to (1 << (256 * (n+1))) - 1 to keep ints small?
                # Python ints are efficient, but let's see.
                # 256 * 60 = 15360 bits. That's ~2KB. Very small.
                # No need to mask unless we go very large.

        return S

    best_set = []
    
    # Generate diverse orders
    orders = []
    
    # 1. Sorted ascending and descending
    orders.append(list(range(1, 257)))
    orders.append(list(range(256, 0, -1)))
    
    # 2. Alternating small/large
    small = list(range(1, 129))
    large = list(range(256, 127, -1))
    alt = []
    for i in range(128):
        alt.append(small[i])
        alt.append(large[i])
    orders.append(alt)
    
    # 3. Random with different seeds
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > deadline - 2:
            break
        o = list(range(1, 257))
        rng.shuffle(o)
        orders.append(o)
    
    # 4. Biased random: weight towards certain ranges
    for trial in range(200):
        if time.time() > deadline - 2:
            break
        o = list(range(1, 257))
        # Create a biased order: pick from ranges with different probabilities
        biased = []
        for _ in range(256):
            r = rng.random()
            if r < 0.3:
                biased.append(rng.randint(1, 64))
            elif r < 0.6:
                biased.append(rng.randint(65, 128))
            elif r < 0.9:
                biased.append(rng.randint(129, 192))
            else:
                biased.append(rng.randint(193, 256))
        # Remove duplicates, keep unique
        seen = set()
        unique_biased = []
        for v in biased:
            if v not in seen:
                seen.add(v)
                unique_biased.append(v)
        # Add any missing values
        for v in range(1, 257):
            if v not in seen:
                unique_biased.append(v)
        orders.append(unique_biased)
        
    # 5. Spaced orders (e.g., multiples of 4, then 3, etc.)
    # This might help with sum separation.
    for step in [2, 3, 4, 5, 8]:
        o = []
        for i in range(1, 257, step):
            o.append(i)
        # Fill in the rest
        for v in range(1, 257):
            if v not in o:
                o.append(v)
        orders.append(o)
        
    # 6. Descending spaced
    for step in [2, 3, 4, 5, 8]:
        o = []
        for i in range(256, 0, -step):
            o.append(i)
        for v in range(256, 0, -1):
            if v not in o:
                o.append(v)
        orders.append(o)

    for o in orders:
        if time.time() > deadline:
            break
        S = try_greedy_build(o)
        if len(S) > len(best_set):
            best_set = S
            # If we found a very good set, we can tighten the deadline check
            if len(S) >= 15:
                deadline = start_time + 58
    
    return [int(x) for x in best_set]
