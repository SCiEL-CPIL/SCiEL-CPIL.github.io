import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 60s limit
    
    # Target: 1..30 distinct integers in 1..256
    # Maximize size.
    
    # Initial seed set
    # We want numbers that are pairwise incomparable in divisibility for their subset sums.
    # A simple heuristic: start with large numbers to reduce the number of subset sums?
    # Or start with small numbers to allow more combinations?
    # Actually, if we pick numbers close to 256, the subset sums are large and spread out, 
    # making divisibility less likely. But we need 30 numbers.
    # The range is 1..256. 
    # If we pick 30 numbers, the average is around 128.
    
    # Let's try a greedy approach.
    # Start with an empty set.
    # Add numbers one by one.
    # To check if adding a new number x is valid:
    # 1. Generate all subset sums of current set S.
    # 2. For each sum s in S_sums, the new sums are s and s+x.
    # 3. We need to ensure that no two distinct sums in the new set divide each other.
    #    This includes:
    #    - s1 divides s2 (already checked for S)
    #    - s1 divides s2 + x
    #    - s1 + x divides s2
    #    - s1 + x divides s2 + x
    #    - s1 divides x (if s1=0? no, non-empty sums. x is a sum itself)
    #    - x divides s1
    #    - x divides s1 + x (always true if x>0? No, x divides x+s1 iff x divides s1. 
    #      Wait, if x divides s1, then x divides s1+x. So we must check x | s1 for all s1.
    #      If x | s1, then sum x and sum s1+x exist. x | (s1+x). Invalid.
    #      So x must NOT divide any existing subset sum s1.
    #    - s1 divides x. If s1 | x, then s1 | (s1+x)? No, s1 | x implies x = k*s1.
    #      s1+x = (k+1)s1. s1 divides s1+x. So s1 must NOT divide x.
    #      So x must be incomparable with all existing subset sums s1.
    #    - s1+x divides s2+x.
    #    - s2+x divides s1+x.
    #    - s1+x divides s2 (if s1+x < s2)
    #    - s2 divides s1+x (if s2 < s1+x)
    
    # This incremental check is O(|S_sums|^2) roughly, or O(|S_sums| * log(max_sum)) if optimized.
    # |S_sums| can be up to 2^30 which is huge.
    # But we stop at 30 elements. 2^30 is too many to generate explicitly in Python quickly if we do it naively.
    # However, the sums are bounded by 30*256 = 7680.
    # We can represent the set of sums as a boolean array of size 7681.
    
    max_val = 256
    max_size = 30
    max_sum = max_size * max_val + 1  # 7681
    
    # Candidate numbers to add.
    # Which numbers are good candidates?
    # Large numbers are often better because they are less likely to divide smaller sums.
    # But small numbers create many small sums.
    # Let's try adding numbers in descending order from 256 down to 1.
    # Or maybe random?
    # Greedy: try to add the largest possible number that fits.
    
    current_set = []
    # Use a bitset or boolean array for subset sums
    # sums_present[0] is always True (empty set), but we only care about non-empty.
    # Actually, the condition is on NONEMPTY subset sums.
    # Let's maintain a set of non-empty sums.
    
    # For efficiency, use a boolean array `sums` where sums[s] is True if s is a subset sum.
    # Initially, only 0 is a sum (empty set).
    sums = np.zeros(max_sum, dtype=bool)
    sums[0] = True
    
    # To check validity of adding x:
    # 1. New sums will be old_sums union (old_sums + x).
    # 2. We need to check divisibility between:
    #    a. Old sums among themselves (already valid)
    #    b. Old sums and x (x is a new sum)
    #    c. x and new sums (s + x)
    #    d. New sums (s + x) among themselves
    #    e. Old sums and new sums (s1 and s2+x)
    
    # Let's precompute the list of existing non-empty sums.
    # Since max_sum is small (~7680), we can iterate over the range.
    
    def check_validity(x, sums, current_sums_list):
        # current_sums_list is a list of non-zero sums currently present.
        # x is the new number.
        
        # 1. Check x vs existing sums
        # x must not divide any existing sum s
        # and no existing sum s must divide x.
        # i.e., for all s in current_sums_list:
        #   if x % s == 0: return False
        #   if s % x == 0: return False
        
        # Optimization: 
        # If s > x, s % x == 0 is possible. x % s == 0 is impossible (unless s=0, which is excluded).
        # If s < x, x % s == 0 is possible. s % x == 0 is impossible.
        
        for s in current_sums_list:
            if s > x:
                if s % x == 0:
                    return False
            else: # s < x (s cannot be x as x is new and distinct from set, but s is a sum, x is an element. 
                   # Could s == x? If x is already a sum of other elements, then we have two subsets with sum x.
                   # The problem says: "Two different subsets with the same sum still make the set invalid".
                   # So if x is already in current_sums_list, it's invalid.
                   # But x is an integer not in current_set. It could be a sum of current_set.
                   if s == x:
                        return False
            if s < x:
                if x % s == 0:
                    return False
        
        # 2. Check x vs new sums (s + x)
        # New sums are s + x for s in {0} U current_sums_list.
        # Actually, the new sums generated are {x + s | s is a subset sum of current_set}.
        # This includes x (when s=0).
        # We already checked x vs current sums.
        # Now we need to check x vs (s+x) for s in current_sums_list (non-zero).
        # x divides (s+x) iff x divides s.
        # We already checked that no s divides x and x doesn't divide s? 
        # Wait. "x divides s" is checked in step 1 (x % s == 0 is s|x? No. x % s == 0 means s divides x).
        # We need to check if x divides (s+x). This is equivalent to x divides s.
        # In step 1, we checked `if s < x: if x % s == 0`. That checks if s divides x.
        # We did NOT check if x divides s. 
        # If x divides s, then x divides s+x. So we must check `if s % x == 0`.
        # In step 1:
        # if s > x: if s % x == 0: return False. (This covers x divides s for s > x)
        # if s < x: x cannot divide s.
        # So step 1 covers "x divides s".
        # Thus, x vs (s+x) is covered by "x divides s".
        
        # 3. Check new sums (s1 + x) and (s2 + x) for s1, s2 in current_sums_list (including 0?)
        # Let A = s1 + x, B = s2 + x.
        # If A == B, invalid (distinct subsets same sum).
        # This happens if s1 == s2. Since s1, s2 are distinct subset sums?
        # No, s1 and s2 are elements from the set of sums. If s1 != s2, then A != B.
        # But if there are two different subsets with the same sum s, that was already invalid for current_set.
        # So current_sums_list contains unique values.
        # So A and B are distinct if s1 != s2.
        
        # Divisibility: A divides B or B divides A.
        # (s1+x) divides (s2+x).
        # We need to check all pairs of new sums.
        # Number of new sums is |current_sums_list| + 1 (for s=0).
        # This is O(N^2) where N is number of current sums.
        # N can be up to 7680. 7680^2 is ~60M. Too slow for Python loops in 45s if we do it many times.
        
        # We can optimize:
        # We only need to check if any new sum divides another.
        # Let new_sums = [x + s for s in [0] + current_sums_list]
        
        # 4. Check old sums s and new sums (t + x).
        # s divides (t+x) or (t+x) divides s.
        
        # This is getting complex. Let's just generate the new set of sums and check all pairs.
        # But 2^30 is too big.
        # However, the number of *distinct* sums is bounded by 7680.
        # So the list of sums is at most 7680 long.
        # Checking all pairs in a list of size 7680 is 7680^2 / 2 ~ 30M operations.
        # In Python, 30M operations takes ~1-3 seconds.
        # We do this for each candidate x.
        # We have 256 candidates. 256 * 3s = 768s. Too slow.
        
        # We need a faster check.
        
        # Alternative:
        # Maintain the set of sums as a sorted list.
        # When adding x:
        # 1. Compute new_sums = {s + x for s in sums} U {x}.
        # 2. Merge with old sums.
        # 3. Check divisibility.
        
        # How to check divisibility fast?
        # For each sum a in the new set:
        #   For each multiple m of a:
        #     If m is in the new set and m != a, invalid.
        
        # This is O(Sum * log(Sum)) roughly.
        # Sum is 7680. 7680 * log(7680) ~ 7680 * 13 ~ 100k operations.
        # This is very fast.
        
        # So the algorithm:
        # 1. Have a boolean array `sums` of size MaxSum+1.
        # 2. To add x:
        #    a. Create a copy of `sums` (or track changes).
        #    b. For each s where sums[s] is True:
        #       if s + x <= MaxSum:
        #          new_sums[s+x] = True
        #    c. Also new_sums[x] = True (from s=0).
        #    d. Merge new_sums into a temporary `temp_sums`.
        #    e. Check validity of `temp_sums`.
        #       - Iterate i from 1 to MaxSum.
        #       - If temp_sums[i] is True:
        #           - For j = 2*i, 3*i, ... <= MaxSum:
        #               - If temp_sums[j] is True:
        #                   - Return False (i divides j)
        #    f. If valid, update `sums` to `temp_sums` and add x to current_set.
        
        # Complexity:
        # Step b: O(MaxSum) = 7680.
        # Step e: O(MaxSum log MaxSum) = ~100k.
        # Total per candidate: ~100k operations.
        # 256 candidates * 100k = 25.6M operations.
        # This should run in < 1 second in NumPy/Python if optimized.
        
        # Let's implement this.
        
        pass

    # Initialize
    current_set = []
    sums = np.zeros(max_sum + 1, dtype=bool)
    sums[0] = True
    
    # Candidates: 1 to 256.
    # Order: Try larger numbers first? 
    # Larger numbers have fewer multiples in the range, so fewer conflicts?
    # Or smaller numbers? 
    # Let's try descending order 256..1.
    
    candidates = list(range(256, 0, -1))
    
    # We want to maximize size.
    # Greedy: add if valid.
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        if len(current_set) >= max_size:
            break
            
        # Check if x can be added
        # 1. Generate potential new sums
        # new_sums will be sums OR (sums shifted by x)
        # We need to check the union.
        
        # Create temp_sums
        temp_sums = sums.copy()
        
        # Shift and OR
        # sums[0..MaxSum-x] shifted to x..MaxSum
        # temp_sums[x:] |= sums[:MaxSum-x+1]
        
        # Note: sums[0] is True. So temp_sums[x] will become True.
        # This handles the sum x.
        
        # Efficient NumPy operation:
        # temp_sums[x:] = temp_sums[x:] | sums[:max_sum + 1 - x]
        
        if x <= max_sum:
             temp_sums[x:] = temp_sums[x:] | sums[:max_sum + 1 - x]
        
        # 2. Check validity of temp_sums
        # Iterate i from 1 to max_sum
        # If temp_sums[i]:
        #    Check multiples 2i, 3i...
        
        # We can vectorize this check?
        # Or just loop. 7680 iterations is small.
        
        valid = True
        # Get indices where temp_sums is True
        # Instead of getting all indices, just loop i.
        # To speed up, we can precompute the list of active sums?
        # But the set changes.
        
        # Simple loop:
        for i in range(1, max_sum + 1):
            if temp_sums[i]:
                # Check multiples
                # range(2*i, max_sum+1, i)
                # If any of these are in temp_sums, invalid.
                # We can check: 
                # multiples = range(2*i, max_sum+1, i)
                # if np.any(temp_sums[multiples]): valid = False; break
                # But creating the range and indexing might be slow.
                # Alternative:
                # for j in range(2*i, max_sum+1, i):
                #    if temp_sums[j]: ...
                
                # Let's try the np.any approach with slicing if possible?
                # Slicing with step is not a contiguous view, so it copies?
                # For small arrays, it's fine.
                
                # Optimization: 
                # If i > max_sum // 2, no multiples exist.
                # So loop only to max_sum // 2.
                if i > max_sum // 2:
                    continue
                    
                # Check multiples
                # We can check if any element in temp_sums[2*i : max_sum+1 : i] is True
                # This creates a view/copy.
                if np.any(temp_sums[2*i : max_sum + 1 : i]):
                    valid = False
                    break
                    
        if valid:
            current_set.append(x)
            sums = temp_sums
            
    # Return the set
    # Must be list of ints.
    return current_set

# Test locally? No, just output.
# One issue: "Two different subsets with the same sum still make the set invalid".
# My validity check `if np.any(temp_sums[2*i...])` checks if i divides j.
# It does NOT check if two different subsets have the same sum.
# However, the problem states: "Equal sum values are counted once. Two different subsets with the same sum still make the set invalid: removing their common elements leaves two disjoint subsets with the same sum s, and their union has sum 2s, which s divides."
# This implies that if there are two subsets with sum s, then s and 2s are both subset sums (s from one, 2s from union).
# Since s divides 2s, the divisibility check `s | 2s` will catch this!
# So checking divisibility among distinct sums is sufficient to enforce uniqueness of subset sums (for non-zero sums).
# Wait, what if the two subsets are disjoint? Then sum is s and s. Union is 2s. s | 2s. Caught.
# What if they overlap? Remove common, get disjoint A, B with sum s. Union A U B has sum 2s. s | 2s. Caught.
# So yes, checking divisibility is sufficient.

# What about the sum 0?
# "Form every NONEMPTY subset sum".
# So 0 is not considered.
# My `sums` array has `sums[0]=True` to facilitate the shift.
# The validity check loops `i` from 1. So 0 is ignored.
# This is correct.

# One edge case: 
# If x is added, and x is a multiple of some existing sum s?
# Then s | x. 
# In the check:
# i = s. 
# Multiples of s include x.
# If temp_sums[x] is True, then np.any(temp_sums[2*s...]) will detect it if x >= 2s.
# What if x < 2s? i.e., s < x < 2s.
# Then x is not a multiple of s.
# But we need to check if x divides s? No, s > x.
# We need to check if s divides x? No, x < 2s, so s cannot divide x (unless x=s, but x is new).
# Wait. If s > x, s cannot divide x.
# If x > s, x can divide s? No, x > s.
# So if s and x are distinct and neither divides the other, we are good for that pair.
# The check `i` divides `j` covers all pairs where one is a multiple of the other.
# Since divisibility is symmetric in the sense that if a|b or b|a, one of them will be the smaller `i` in the loop.
# So the check is complete.

# Final check on types:
# "Types are checked exactly: integers must be Python int... sequences must be list or tuple"
# `current_set` is a list of Python ints (from range).
# `return current_set` is a list.

# Deadline guard is in place.

# One potential issue: 
# `temp_sums[x:] = temp_sums[x:] | sums[:max_sum + 1 - x]`
# If x is large, `max_sum + 1 - x` is small.
# `sums[:k]` is a view.
# `temp_sums[x:]` is a view.
# `|` creates a new array.
# Assignment copies it back.
# This is O(N).

# The loop `for i in range(1, max_sum + 1)` is ~7680 iterations.
# Inside, `np.any` on a slice.
# This should be fast enough.

# Let's double check the "distinct integers" constraint.
# We iterate candidates 256..1.
# We only add x if it's valid.
# Since we iterate each x once, and only add if not already in set (implied by iteration), 
# and we check validity which implies x is not a sum of existing (which would cause collision),
# wait, does validity check ensure x is not in current_set?
# Yes, we iterate unique candidates.
# Does it ensure x is not a sum of current_set?
# If x is a sum of current_set, then `sums[x]` is True.
# When we add x, `temp_sums[x]` becomes True (it was already True).
# But we also generate new sums `s+x`.
# If x is a sum, say x = sum(A).
# Then we have subset A with sum x.
# And subset {x} with sum x.
# These are two distinct subsets with same sum x.
# This leads to x and 2x being sums?
# Union of A and {x} is A U {x}. Sum is 2x.
# So x and 2x are in temp_sums.
# x divides 2x.
# The check `i=x` will look for multiples. 2x is a multiple.
# If `temp_sums[2x]` is True, it will be caught.
# Is `temp_sums[2x]` guaranteed to be True?
# 2x = x + x.
# x is a sum (from A).
# So x+x is a sum (from A U {x}).
# So yes, 2x will be in temp_sums.
# So the check will catch it.

# Therefore, the algorithm is correct.
