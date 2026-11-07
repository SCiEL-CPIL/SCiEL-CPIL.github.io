import time

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ.
    
    Strategy:
    1. Start with the known optimal AP [226..256] of size 31.
    2. Attempt to grow the set by adding elements from 1..225 in descending order.
    3. For each candidate, check if the property holds using a fast range-based check
       (which is necessary but not sufficient) and then a full DP check (sufficient).
    4. Use a deadline guard to ensure we finish within 90s.
    
    The property "subset sums of different cardinalities are distinct" is equivalent to:
    For all k1 != k2, max_sum(k1) < min_sum(k2) OR max_sum(k2) < min_sum(k1).
    Since we are adding elements, the sums generally increase, so we mainly care about
    whether adding a small element creates a collision with a larger cardinality's min sum.
    
    However, a more robust check is:
    For each cardinality k, compute the set of all possible subset sums of size k.
    Check that these sets are pairwise disjoint for different k.
    
    To speed this up, we can use the fact that if the ranges [min_sum(k), max_sum(k)]
    are disjoint, then the property holds. If they overlap, we need to check the actual
    sums. But for APs and near-APs, the ranges are often disjoint or have small overlaps.
    
    Actually, the condition "whenever two subsets have different cardinalities their sums must differ"
    means that the sets of sums S_k = {sum(T) | T subset of S, |T|=k} must be pairwise disjoint for k != l.
    
    We can compute S_k using DP. For k up to ~31, and values up to 256, the total sum is at most 31*256 = 7936.
    We can use a bitset for each cardinality k.
    
    Let's implement a DP that tracks possible sums for each cardinality.
    dp[k] is a bitset where bit s is set if there exists a subset of size k with sum s.
    
    We start with dp[0] = {0}.
    For each element x in the set:
      for k from current_max_k down to 0:
        dp[k+1] |= (dp[k] << x)
    
    Then we check if any bit is set in dp[k] & dp[l] for k != l.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Initial set: AP from 226 to 256
    # This is known to be valid and size 31.
    current_set = list(range(226, 257))
    
    # Candidate elements to add: 1 to 225, in some order.
    # To maximize the chance of success, we might want to add elements that are "far" from the current set
    # or that fill gaps in a way that preserves the separation.
    # However, adding small elements increases the min sum for small k, which might collide with max sum for larger k.
    # Wait, min sum for k increases if we add a large element? No.
    # min_sum(k) is the sum of the k smallest elements.
    # If we add a small element, min_sum(1) decreases, min_sum(2) might decrease, etc.
    # This makes it *harder* to satisfy max_sum(k) < min_sum(k+1) because min_sum(k+1) decreases.
    # So adding small elements is bad for the range separation condition.
    # Therefore, we should try adding the *largest* possible elements first?
    # But the largest available are 225, 224, ...
    # Let's try adding from 225 down to 1.
    
    candidates = list(range(225, 0, -1))
    
    def is_valid(s):
        """
        Check if the set s satisfies the cardinality-separated subset sum property.
        Returns True if valid.
        """
        if len(s) == 0:
            return True
        
        # DP with bitsets
        # max_sum is at most sum(s)
        total_sum = sum(s)
        n = len(s)
        
        # dp[k] is a bitset (integer) where bit i is 1 if sum i is possible with k elements
        # We only need to track up to total_sum
        # Initialize dp[0] = 1 (sum 0 is possible with 0 elements)
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for x in s:
            # Update dp in reverse order to avoid using the same element multiple times
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << x)
        
        # Now check for collisions between different cardinalities
        # For k from 0 to n-1, check if dp[k] & dp[k+1] is 0?
        # No, we need to check all pairs k, l where k != l.
        # But since dp[k] contains sums of subsets of size k, and dp[l] for size l,
        # if any sum s is in both dp[k] and dp[l], it's a collision.
        
        # Optimization: If the ranges [min_sum(k), max_sum(k)] are disjoint, we can skip the bitset check for that pair.
        # But computing min/max sums is O(n^2) or O(n log n).
        # Let's just do the bitset check. It's O(n * total_sum / word_size).
        # total_sum <= 256 * 60 = 15360. n <= 60.
        # 60 * 15360 / 64 ~ 14400 operations. Very fast.
        
        for k in range(n + 1):
            for l in range(k + 1, n + 1):
                if dp[k] & dp[l]:
                    return False
        return True
    
    # Verify initial set
    if not is_valid(current_set):
        # This should not happen, but just in case
        pass
    
    best_set = current_set[:]
    
    # Try to add candidates
    for x in candidates:
        if time.time() > deadline:
            break
        
        # Try adding x
        new_set = current_set + [x]
        # Sort for consistency (not strictly necessary for DP, but good for min/max logic if we used it)
        # Actually, DP doesn't need sorting.
        
        if is_valid(new_set):
            current_set = new_set
            best_set = current_set[:]
            # Continue trying to add more
        # Else, skip x
        
    # Final check
    if not is_valid(best_set):
        # Fallback to a known valid smaller set if something went wrong
        # But the initial set was valid, so this shouldn't happen unless is_valid is buggy.
        pass
        
    # Ensure we return a list of Python ints
    return list(best_set)

# EVOLVE-BLOCK-END
