import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, leave 5s margin
    
    best_n = 0
    best_set = []
    
    # --- Strategy 1: Gap-based structured construction ---
    # If we choose numbers such that the minimum sum of (k+1) elements is strictly greater
    # than the maximum sum of k elements, then all cardinality classes are disjoint.
    # Let the sorted set be a_1 < a_2 < ... < a_n.
    # Min sum of k+1 elements: S_min(k+1) = sum_{i=1}^{k+1} a_i
    # Max sum of k elements: S_max(k) = sum_{i=n-k+1}^{n} a_i
    # Condition: S_min(k+1) > S_max(k) for all k in 0..n-1
    # For k=0: S_min(1) = a_1 > 0 (always true for positive ints)
    # For k=n-1: S_min(n) > S_max(n-1) => sum(a) > sum(a) - a_1 => a_1 > 0 (always true)
    # The tightest constraints are in the middle.
    # A simple sufficient condition: a_{k+1} > a_{n-k} for all k.
    # This ensures the "left" side grows faster than the "right" side shrinks.
    # Actually, let's just greedily build a set satisfying S_min(k+1) > S_max(k).
    
    # We want to maximize n.
    # Let's try to build a set from the high end downwards or specific structures.
    # If we pick a_i = c * 2^i, sums are unique even within cardinality (superincreasing).
    # But we are limited to 1..256.
    # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128. Next is 256.
    # Set: {1, 2, 4, 8, 16, 32, 64, 128, 256}. Size 9.
    # Can we do better with non-powers?
    
    # Let's use the greedy gap check.
    # We maintain a sorted list. When adding a new element x, we insert it in sorted order.
    # Then we check if the gap condition holds for the new set.
    # To maximize size, we should try to add small numbers first? Or large?
    # If we add small numbers, S_min grows slowly, S_max stays low (if we haven't added large ones yet).
    # If we add large numbers, S_max grows fast, S_min stays low.
    # The condition S_min(k+1) > S_max(k) is hard to satisfy if both sides are large.
    # It is easiest if the set is "skewed".
    
    # Let's try a specific family: Geometric progression with base r.
    # Or just use the general greedy with the gap check.
    
    # Optimization: Pre-compute prefix sums and suffix sums for the current sorted set.
    # But since we are inserting, we need to update.
    # Given the small size (max 60), we can just recompute or update efficiently.
    
    # Let's try a few strategies for the gap-based approach:
    # 1. Greedy from 1 upwards: Try to add the smallest available integer that maintains the gap condition.
    # 2. Greedy from 256 downwards: Try to add the largest available integer.
    # 3. Mix.
    
    def check_gap_condition(sorted_set):
        n = len(sorted_set)
        if n == 0:
            return True
        # S_min(k+1) = sum of first k+1 elements
        # S_max(k) = sum of last k elements
        # For k=0: S_min(1) > 0
        # For k=n-1: S_min(n) > S_max(n-1)
        
        # Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + sorted_set[i]
        
        # Compute suffix sums
        suffix = [0] * (n + 1)
        for i in range(n-1, -1, -1):
            suffix[i] = suffix[i+1] + sorted_set[i]
        
        # Check condition for k = 0 to n-1
        for k in range(n):
            # S_min(k+1) is prefix[k+1]
            # S_max(k) is sum of last k elements = suffix[n-k]
            # Note: if k=0, S_max(0) = 0. suffix[n] = 0.
            if prefix[k+1] <= suffix[n-k]:
                return False
        return True

    # Strategy 1a: Greedy from 1 to 256
    current_set = []
    for x in range(1, 257):
        if time.time() > deadline:
            break
        # Try adding x
        # Insert x into sorted_set
        # Since we are iterating 1..256, x is always larger than current elements.
        # So sorted_set + [x] is the new sorted set.
        candidate_set = current_set + [x]
        if check_gap_condition(candidate_set):
            current_set = candidate_set
            if len(current_set) > best_n:
                best_n = len(current_set)
                best_set = current_set[:]
            if best_n >= 60:
                break

    # Strategy 1b: Greedy from 256 to 1
    current_set = []
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        # x is always smaller than current elements (since we go down).
        # So [x] + current_set is the new sorted set.
        candidate_set = [x] + current_set
        if check_gap_condition(candidate_set):
            current_set = candidate_set
            if len(current_set) > best_n:
                best_n = len(current_set)
                best_set = current_set[:]
            if best_n >= 60:
                break

    # Strategy 1c: Try to fill gaps. Start with powers of 2, then try to insert others.
    # This might allow more elements if the gap condition is loose.
    # But the gap condition is quite strict.
    # Let's stick to the two greedy passes for now as they are fast.
    
    # --- Strategy 2: Bitset collision check (Original approach, optimized) ---
    # Use this if gap-based didn't reach 60, or to potentially beat it?
    # Gap-based guarantees validity. Bitset allows more flexibility.
    # The current score is 3.875, which corresponds to size 31 (31/8? No, score = value/seed. Seed value? 
    # If seed is 8, 31/8 = 3.875. So current best is size 31.
    # Can we beat 31 with gap-based?
    # Let's see how large the gap-based set can be.
    # If a_i = 2^i, size is 9 (1,2,4,8,16,32,64,128,256).
    # Can we add more?
    # If we have 1, 2, 4, 8, 16, 32, 64, 128.
    # S_min(5) = 1+2+4+8+16 = 31. S_max(4) = 32+64+128+? 
    # Wait, the set is {1,2,4,8,16,32,64,128}.
    # S_min(5) = 31. S_max(4) = 32+64+128 = 224. 31 < 224. Condition fails.
    # So powers of 2 do NOT satisfy the gap condition S_min(k+1) > S_max(k).
    # The gap condition is MUCH stricter than just disjointness.
    # Disjointness only requires that the *ranges* don't overlap? No, the *sets* of sums.
    # If S_min(k+1) > S_max(k), then the sets are disjoint.
    # But sets can be disjoint even if ranges overlap, as long as specific sums don't match.
    # The bitset approach checks exact sums. The gap approach checks a sufficient condition.
    # The gap condition is likely too strict to get size 31.
    # Example: {1, 2, 4, 8, 16, 32, 64, 128} has size 8.
    # To get size 31, we need a much denser set.
    # Therefore, the gap-based strategy will likely yield a smaller set than the bitset strategy.
    # We should KEEP the bitset strategy as the primary one, and maybe use gap-based as a quick check or fallback?
    # Or maybe the "inspiration" about gap-based was for a different variant?
    # "Constructs a set of 31 integers in [226, 256]... gap between max sum of k and min sum of k+1 is positive".
    # If the set is in [226, 256], the elements are large.
    # Min sum of 1: 226. Max sum of 0: 0.
    # Min sum of 2: 226+227 = 453. Max sum of 1: 256. 453 > 256.
    # Min sum of 3: 226+227+228 = 681. Max sum of 2: 256+255 = 511. 681 > 511.
    # It seems that if all elements are large, the gap condition holds easily.
    # Let's test this hypothesis: Pick a set of large numbers.
    # If we pick numbers from 200 to 256.
    # Min sum of k+1 vs Max sum of k.
    # Let a_i be the sorted set.
    # S_min(k+1) >= (k+1) * min_val
    # S_max(k) <= k * max_val
    # Condition: (k+1) * min_val > k * max_val
    # min_val / max_val > k / (k+1)
    # If min_val = 200, max_val = 256. Ratio = 200/256 = 0.78.
    # For k=1: 1/2 = 0.5. 0.78 > 0.5. OK.
    # For k=10: 10/11 = 0.909. 0.78 < 0.909. Fails.
    # So we can't just take all large numbers. We need to be careful.
    # However, if we pick numbers very close to each other, e.g., 250, 251, ..., 256.
    # Min sum of k+1: sum of smallest k+1.
    # Max sum of k: sum of largest k.
    # If the numbers are consecutive, the gap is small.
    # Let's try the bitset approach first, as it is more general.
    
    # Optimize the bitset approach:
    # Use a single `all_bits` mask to check collisions against ALL other cardinalities at once?
    # No, because collision is only allowed WITHIN the same cardinality.
    # So we must check `new_sums & bits[j]` for j != k+1.
    # We can pre-compute `all_bits_except_k_plus_1`?
    # Or just iterate. The iteration is over k (up to 60) and j (up to 60). 3600 ops.
    # With bitwise AND on Python ints (which are fast), this is very fast.
    
    # Let's rewrite the bitset greedy to be as fast as possible.
    
    if time.time() < deadline:
        # Try multiple orders
        orders = [
            list(range(1, 257)),
            list(range(256, 0, -1)),
            list(range(2, 257, 2)) + list(range(1, 257, 2)),
            list(range(1, 257, 2)) + list(range(2, 257, 2)),
        ]
        
        # Add some random shuffles
        for _ in range(10):
            o = list(range(1, 257))
            random.shuffle(o)
            orders.append(o)
        
        for fam in orders:
            if time.time() > deadline:
                break
            
            current_set = []
            bits = [0] * 61
            bits[0] = 1
            used = set()
            
            for x in fam:
                if time.time() > deadline:
                    break
                if x in used:
                    continue
                
                valid = True
                # Check collision
                # For each k, new_sums = bits[k] << x
                # Must not intersect with bits[j] for j != k+1
                for k in range(len(current_set), -1, -1):
                    if bits[k] == 0:
                        continue
                    new_sums = bits[k] << x
                    # Check against all j != k+1
                    # We can check against a combined mask of all j != k+1
                    # But building that mask is expensive.
                    # Just loop j.
                    for j in range(len(current_set) + 2):
                        if j == k + 1:
                            continue
                        if j < 61 and (bits[j] & new_sums):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Update bits
                    for k in range(len(current_set), -1, -1):
                        if bits[k]:
                            bits[k + 1] |= bits[k] << x
                    current_set.append(x)
                    used.add(x)
                    
                    if len(current_set) > best_n:
                        best_n = len(current_set)
                        best_set = current_set[:]
                    
                    if best_n >= 60:
                        break
        
        # Random restarts to try to beat the greedy result
        if best_n < 60 and time.time() < deadline - 5:
            for _ in range(1000):
                if time.time() > deadline:
                    break
                
                target_n = best_n + 1
                if target_n > 60:
                    break
                
                candidates = random.sample(range(1, 257), target_n)
                bits = [0] * (target_n + 1)
                bits[0] = 1
                for x in candidates:
                    for k in range(target_n - 1, -1, -1):
                        if bits[k]:
                            bits[k + 1] |= bits[k] << x
                
                seen_bits = 0
                valid = True
                for k in range(target_n + 1):
                    if bits[k] & seen_bits:
                        valid = False
                        break
                    seen_bits |= bits[k]
                
                if valid:
                    best_n = target_n
                    best_set = candidates

    # Ensure we return a valid list of ints
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 60}

# The construct function is defined above.
# We need to make sure it's accessible as construct().
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program had:
# def construct():
#    return ...
# So we just expose construct.

# EVOLVE-BLOCK-END
