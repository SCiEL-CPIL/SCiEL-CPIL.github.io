import time
import sys

def construct():
    start = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    # We want to find the largest set S of distinct integers in [1, 256]
    # such that for any two subsets A, B with |A| != |B|, sum(A) != sum(B).
    # This is equivalent to: The set of all subset sums of cardinality c,
    # denoted S_c, must be disjoint from S_d for all c != d.
    
    # Strategy:
    # 1. Start with empty set S = [].
    # 2. Maintain DP[c] = set of all possible sums of subsets of S with cardinality c.
    # 3. To add a new number x:
    #    a. Check if x is in [1, 256] and not in S.
    #    b. Compute new DP' where DP'[c] = DP[c] U {s + x | s in DP[c-1]}.
    #    c. Check if any DP'[c] intersects with any DP'[d] for c != d.
    #    d. If valid, add x to S and update DP.
    # 4. Try to add numbers in increasing order? Or search for the best next number?
    #    A simple greedy approach: Try adding 1, 2, 3, ...
    #    However, adding small numbers early might restrict future options.
    #    But since we want to MAXIMIZE size, and the range is small (1..256),
    #    a greedy approach that picks the smallest valid number is a good heuristic.
    #    Alternatively, we can try to find a valid set of size k by backtracking,
    #    but greedy is much faster.
    
    # Let's use a slightly smarter greedy:
    # Maintain current set S and current DP.
    # For each step, try candidate x from current_min+1 to 256.
    # If adding x keeps the DP disjoint, add it.
    # To allow for more flexibility, we might want to try smaller numbers first
    # because they leave more "room" in the sum space? Or larger numbers?
    # Actually, the constraint is on the *separation* of sums.
    # Small numbers create dense sums. Large numbers create sparse sums.
    # The "Powers of 2" strategy works because sums are unique.
    # A greedy that picks the smallest valid number might get stuck if it creates
    # too much overlap potential.
    
    # Let's try a greedy that picks the smallest valid number.
    # If it gets stuck, we might not find the max.
    # But the ILP approach scored 2.25 (likely 18 elements? 2.25 * 8 = 18).
    # Wait, the seed is likely 8 (powers of 2). 2.25 * 8 = 18.
    # So the current best is 18.
    # Can we get more than 18?
    # The maximum possible size is bounded by the fact that we have 256 numbers.
    # With 18 numbers, there are 2^18 subsets. The sums must be separated by cardinality.
    # There are 19 cardinalities (0 to 18).
    # The total number of distinct sums is at most 256*18/2 + ... roughly 2300.
    # 2^18 is 262144. So we have plenty of room for sums.
    # The bottleneck is the overlap between different cardinalities.
    
    # Let's implement the greedy with a twist:
    # We will try to build the set. If we can't add the next smallest number,
    # we might try a larger one? No, just pick the smallest valid one.
    # This is deterministic and fast.
    
    S = []
    # DP[c] is a set of sums for cardinality c
    # Max cardinality will be len(S)
    # We can use a list of sets.
    
    # To make it faster, we can use a single set of tuples (cardinality, sum)
    # or just check intersections.
    
    # Let's use a list of sets: dp[c]
    dp = [set() for _ in range(257)] # Max cardinality 256
    dp[0].add(0)
    current_max_c = 0
    
    # We will try to add numbers from 1 to 256.
    # But maybe we should skip some?
    # Let's just iterate through candidates 1..256 and add if valid.
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
            
        # Check if x is already in S
        if x in S:
            continue
            
        # Simulate adding x
        # New sums for cardinality c: old sums for c, plus (old sums for c-1) + x
        # We need to check if any new sum conflicts with existing sums of DIFFERENT cardinality.
        
        # Let's create a temporary set of all (c, sum) pairs that would exist.
        # Actually, we just need to check intersections.
        
        # Potential new sums:
        # For each c from current_max_c down to 0:
        #   new_sums = {s + x for s in dp[c]}
        #   These will become part of dp[c+1]
        
        # We need to ensure:
        # 1. For any c, (dp[c] U new_sums_from_c-1) does not intersect with dp[d] for d != c+1?
        #    No, the condition is:
        #    For any c != d, the final set of sums for c is disjoint from final set of sums for d.
        
        # Let's compute the potential new dp sets.
        # new_dp[c] = dp[c] U {s+x for s in dp[c-1]}
        
        # We only need to check conflicts involving the new sums.
        # A new sum s' = s + x (where s in dp[c-1]) is added to dp[c].
        # It must not be in any dp[d] for d != c.
        
        # Optimization:
        # We can maintain a global set of all (sum, cardinality) pairs?
        # Or just check:
        # For each c from 0 to current_max_c:
        #   for each s in dp[c]:
        #      new_sum = s + x
        #      # This new_sum will be in cardinality c+1
        #      # Check if new_sum is in dp[d] for any d != c+1
        #      # AND check if new_sum is in the "new" dp[c+1] from other s?
        #      # Actually, if two different subsets of cardinality c produce the same new_sum,
        #      # that's fine (repeated sums within same cardinality allowed).
        #      # But if new_sum is in dp[d] for d != c+1, it's invalid.
        
        # Also, we must ensure that the new sums don't conflict with EACH OTHER in a way that
        # violates the rule? No, conflicts within the same new cardinality are allowed.
        
        # So the check is:
        # For all c in 0..current_max_c:
        #   For all s in dp[c]:
        #     ns = s + x
        #     If ns is in dp[d] for any d != c+1:
        #        Invalid.
        
        # Wait, what if ns is in dp[c+1] already? That's fine.
        # What if ns is in dp[c]? That would mean a sum of cardinality c+1 equals a sum of cardinality c. Invalid.
        # What if ns is in dp[d] for d > c+1? Invalid.
        
        # So we need to check if ns is in the union of all dp[d] for d != c+1.
        
        # To make this fast, let's maintain a set `all_sums` of all sums seen so far,
        # but we need to know their cardinalities to exclude the correct one.
        
        # Alternative:
        # Maintain `sums_by_cardinality` which is `dp`.
        # For a candidate x, we compute the set of new sums `new_sums` (which will go into cardinality c+1).
        # We need to check if `new_sums` intersects with `dp[d]` for any d != c+1.
        
        # Let's pre-compute the union of all dp[d] for d != c+1?
        # That's expensive to do for each c.
        
        # Better:
        # Let `all_sums` be the set of all sums in all dp[c].
        # If `ns` is not in `all_sums`, it's definitely safe regarding other cardinalities.
        # If `ns` is in `all_sums`, we need to check if it ONLY exists in dp[c+1].
        
        # So:
        # 1. Compute all potential new sums `new_sums_set`.
        # 2. For each `ns` in `new_sums_set`:
        #    If `ns` is in `all_sums`:
        #       Check if `ns` is in `dp[d]` for any `d` such that `d != c+1` (where c is the source cardinality).
        #       Wait, `ns` comes from `dp[c]`. So it targets `dp[c+1]`.
        #       We need to ensure `ns` is not in `dp[d]` for any `d != c+1`.
        
        # This check is O(|new_sums| * K) where K is number of cardinalities.
        # |new_sums| can be large.
        
        # Let's try a simpler heuristic first:
        # Just try to add 1, 2, 3... and see how far we get.
        
        # Efficient Check:
        # We maintain `dp` as list of sets.
        # We also maintain `sum_to_cards` dict: sum -> set of cardinalities that have this sum.
        
        # When checking x:
        # For each c in 0..current_max_c:
        #   For each s in dp[c]:
        #     ns = s + x
        #     # ns will be added to card c+1
        #     # Check if ns is in sum_to_cards
        #     if ns in sum_to_cards:
        #        cards = sum_to_cards[ns]
        #        # If any card in cards is not c+1, then conflict!
        #        # Because ns would be in card d (d != c+1) and we are adding it to c+1.
        #        # Wait, if ns is already in dp[c+1], that's fine.
        #        # So we just need to check if there exists d in cards such that d != c+1.
        #        if any(d != c+1 for d in cards):
        #            valid = False
        #            break
        #     else:
        #        # ns is new, no conflict with existing sums.
        #        pass
        #   if not valid: break
        
        # If valid, we update dp and sum_to_cards.
        
        # Let's implement this.
        
        valid = True
        # We need to check all potential new sums.
        # Note: Multiple s in dp[c] can produce the same ns. We only need to check once per unique ns?
        # Yes, but we can just check each s. If one fails, x is invalid.
        
        # To speed up, we can collect unique new sums first?
        # Or just check on the fly.
        
        # Let's check on the fly.
        
        # Optimization: If x is large, the sums are large and less likely to conflict.
        # If x is small, sums are dense.
        
        # Let's try candidates in increasing order.
        
        # We need to iterate through c and s.
        # To avoid modifying dp while checking, we just check against the CURRENT state.
        
        for c in range(current_max_c + 1):
            if not valid:
                break
            for s in dp[c]:
                ns = s + x
                # ns will be in cardinality c+1
                if ns in sum_to_cards:
                    cards = sum_to_cards[ns]
                    # Check if any card is NOT c+1
                    # If cards contains only c+1, it's fine.
                    # If it contains anything else, it's a conflict.
                    # We can check: if cards - {c+1} is not empty.
                    if len(cards) > 1 or (len(cards) == 1 and next(iter(cards)) != c+1):
                        valid = False
                        break
                # If ns not in sum_to_cards, it's new, so no conflict.
        
        if valid:
            # Add x to S
            S.append(x)
            
            # Update dp
            # We must update in reverse order to avoid using updated values for next c
            # But we are adding to c+1 from c.
            # So we should iterate c from current_max_c down to 0?
            # No, we are adding to dp[c+1] based on dp[c].
            # If we iterate c from 0 up, we might use dp[1] that was just updated?
            # No, we are reading dp[c] and writing to dp[c+1].
            # If we update dp[1] from dp[0], then when we process c=1, we read dp[1].
            # But the new sums from c=0 are in dp[1].
            # Do we want to use them to generate sums for c=2?
            # No! The new element x can only be used ONCE.
            # So the new sums for cardinality c+1 are generated ONLY from the OLD dp[c].
            # Therefore, we must iterate c from current_max_c down to 0.
            # Or store the old dp.
            
            # Let's iterate c from current_max_c down to 0.
            new_max_c = current_max_c + 1
            for c in range(current_max_c, -1, -1):
                if dp[c]:
                    # Generate new sums
                    new_sums = {s + x for s in dp[c]}
                    # Add to dp[c+1]
                    dp[c+1].update(new_sums)
                    # Update sum_to_cards
                    for ns in new_sums:
                        if ns in sum_to_cards:
                            sum_to_cards[ns].add(c+1)
                        else:
                            sum_to_cards[ns] = {c+1}
            
            current_max_c = new_max_c
            
            # Check time
            if time.time() > deadline:
                break
        
        # If we finished the loop, we tried all numbers 1..256.
        # But we might have skipped some because they were invalid at that time.
        # Could a number that was invalid earlier become valid later?
        # No, adding more numbers only adds more sums, so it only creates MORE potential conflicts.
        # If x was invalid because it created a conflict with existing sums,
        # adding more numbers will keep that conflict (the existing sums are still there).
        # So if x is invalid now, it will be invalid forever.
        # Thus, a single pass from 1 to 256 is sufficient.
        
    # Return the list of ints
    # Ensure types are Python int
    return [int(x) for x in S]

# We need to initialize sum_to_cards inside construct or globally?
# It should be local to construct.
# I missed the initialization of sum_to_cards in the loop above.
# Let's refactor.

def construct():
    start = time.time()
    deadline = 85.0
    
    S = []
    dp = [set() for _ in range(257)]
    dp[0].add(0)
    current_max_c = 0
    
    # sum_to_cards: dict sum -> set of cardinalities
    sum_to_cards = {0: {0}}
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
            
        valid = True
        
        # Check validity
        for c in range(current_max_c + 1):
            if not valid:
                break
            for s in dp[c]:
                ns = s + x
                if ns in sum_to_cards:
                    cards = sum_to_cards[ns]
                    # If ns exists in any cardinality other than c+1, it's a conflict
                    # because we are adding ns to cardinality c+1.
                    # If ns is already in c+1, it's fine.
                    # So conflict if there is any d in cards such that d != c+1.
                    if any(d != c+1 for d in cards):
                        valid = False
                        break
        
        if valid:
            S.append(x)
            
            # Update dp and sum_to_cards
            # Iterate c from current_max_c down to 0
            for c in range(current_max_c, -1, -1):
                if dp[c]:
                    new_sums = {s + x for s in dp[c]}
                    dp[c+1].update(new_sums)
                    for ns in new_sums:
                        if ns in sum_to_cards:
                            sum_to_cards[ns].add(c+1)
                        else:
                            sum_to_cards[ns] = {c+1}
            
            current_max_c += 1
            
            if time.time() > deadline:
                break
                
    return [int(x) for x in S]

