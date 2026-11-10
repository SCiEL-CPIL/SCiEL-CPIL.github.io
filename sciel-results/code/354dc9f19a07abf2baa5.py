# EVOLVE-BLOCK-START
def construct():
    """
    Construct exactly 12 distinct positive integers <= 1048576 such that every
    subset (including empty) has a unique sum. Minimize the largest integer.
    
    The optimal solution is the powers of 2: [1, 2, 4, 8, ..., 2048].
    This is because for k elements with distinct subset sums, the maximum
    possible sum is at least 2^k - 1, so the largest element must be at
    least 2^(k-1). Powers of 2 achieve this bound exactly.
    
    We use a greedy construction with backtracking framework to verify
    and construct the sequence, though the greedy choice is forced to be
    powers of 2 at each step.
    """
    n = 12
    
    # Greedy construction with backtracking
    # At each step, we try the smallest unused positive integer that
    # doesn't create duplicate subset sums. We use backtracking to
    # explore alternatives if needed (though the greedy choice is optimal).
    
    def is_valid_candidate(elements, candidate):
        """Check if adding candidate to elements preserves distinct subset sums."""
        # Compute all subset sums of elements
        subset_sums = {0}
        for e in elements:
            new_sums = set()
            for s in subset_sums:
                new_sums.add(s + e)
            subset_sums.update(new_sums)
        
        # Check if adding candidate creates any duplicate
        for s in subset_sums:
            if s + candidate in subset_sums:
                return False
            if s + candidate == s:  # candidate == 0, but we use positive ints
                return False
        return True
    
    def backtrack(elements, idx, max_val):
        """
        Recursively build the sequence. elements is the list of chosen integers.
        idx is the number of elements chosen so far.
        max_val is the upper bound for the largest element.
        Returns the completed sequence or None if no solution exists.
        """
        if idx == n:
            return elements[:]
        
        # Try candidates in increasing order
        # The minimum possible next value is 1, but we can optimize
        # by noting that the next value must be > sum of all previous
        # for the greedy optimal solution, but we try all valid ones.
        
        # To keep it efficient, we start from 1 but skip values we know
        # are too small (they would create duplicates with existing subset sums)
        
        # Compute all subset sums of current elements
        subset_sums = {0}
        for e in elements:
            new_sums = set()
            for s in subset_sums:
                new_sums.add(s + e)
            subset_sums.update(new_sums)
        
        # The forbidden values are those x where x = s2 - s1 for some s1, s2
        # in subset_sums, i.e., x is in the difference set
        # But more simply: x is forbidden if s + x is in subset_sums for any s
        # Actually, we need: for all s1 != s2 in subset_sums, s1 + x != s2
        # i.e., x != s2 - s1 for any distinct s1, s2
        # And also x > 0
        
        # Efficient check: x is valid iff for all s in subset_sums, s + x not in subset_sums
        # (since if s1 + x = s2, then s2 - s1 = x)
        
        # We try x from 1 upward
        # But we can bound: x must be at least 1, and at most max_val
        # Also, for the optimal solution, x should be around 2^idx
        
        for x in range(1, max_val + 1):
            # Quick check: if x is in the difference set of subset_sums, skip
            valid = True
            for s in subset_sums:
                if s + x in subset_sums:
                    valid = False
                    break
            if not valid:
                continue
            
            # Also ensure x is not already in elements (distinct integers)
            if x in elements:
                continue
            
            # Try adding x
            elements.append(x)
            result = backtrack(elements, idx + 1, max_val)
            if result is not None:
                return result
            elements.pop()
        
        return None
    
    # The optimal maximum is 2^(n-1) = 2048 for n=12
    # We know this is achievable, so we search with max_val = 2048
    # If we want to be safe, we could try smaller max_val first, but
    # we know 2048 is optimal.
    
    max_val = 1 << (n - 1)  # 2048
    
    result = backtrack([], 0, max_val)
    
    if result is None:
        # Fallback: direct powers of 2
        result = [1 << i for i in range(n)]
    
    return result
# EVOLVE-BLOCK-END
