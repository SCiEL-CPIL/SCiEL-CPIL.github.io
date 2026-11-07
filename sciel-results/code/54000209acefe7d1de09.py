"""Program for cardinality-separated subset sums using reverse greedy construction."""
# EVOLVE-BLOCK-START

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    
    Uses reverse greedy: start with the largest available integers, and for each
    candidate element (from largest to smallest), check whether adding it to the
    current set maintains the cardinality separation constraint. If adding would
    violate any constraint, skip that element and try the next smaller one.
    
    The key insight: we build the set in descending order. When we add a new
    smallest element, it can only affect constraints where it participates in
    the min k-sum. We check all constraints after each addition.
    """
    
    def check_constraints(s):
        """
        Check if sorted list s satisfies: for all k >= 2,
        sum of k smallest > sum of (k-1) largest.
        Returns True if valid, False otherwise.
        """
        n = len(s)
        if n <= 1:
            return True
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + s[i]
        total = prefix[n]
        for k in range(2, n + 1):
            min_k_sum = prefix[k]
            max_km1_sum = total - prefix[n - k + 1]
            if min_k_sum <= max_km1_sum:
                return False
        return True
    
    def try_add(s, candidate):
        """
        Try adding candidate to sorted list s (candidate must be smaller than all
        elements in s). Returns the new sorted list if constraints are maintained,
        or None if adding would violate constraints.
        """
        if not s or candidate >= s[0]:
            # If s is empty, just add
            if not s:
                return [candidate]
            # candidate should be smaller than s[0] for our descending build
            return None
        
        # New set: [candidate] + s, which is already sorted
        new_s = [candidate] + s
        if check_constraints(new_s):
            return new_s
        return None
    
    # Reverse greedy: start with the largest element, then try to add
    # smaller elements one by one.
    # We process candidates from 256 down to 1.
    
    # Start with the largest element
    result = [256]
    
    # Try to add each smaller element
    for candidate in range(255, 0, -1):
        new_result = try_add(result, candidate)
        if new_result is not None:
            result = new_result
    
    return result

# EVOLVE-BLOCK-END
