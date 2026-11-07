import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ.
    
    The optimal size is 31. We use the arithmetic progression [226, 227, ..., 256].
    This works because for any k, the minimum sum of (k+1) elements is strictly greater
    than the maximum sum of (k-1) elements, ensuring no sum collisions across cardinalities.
    
    Specifically:
    - Max sum of (k-1) smallest elements: sum of (k-1) terms of AP
    - Min sum of (k+1) largest elements: sum of (k+1) terms of AP
    - For AP starting at a with common difference d=1:
      max_sum(k-1) = (k-1)*a + (k-1)(k-2)/2
      min_sum(k+1) = (k+1)*a + (k+1)k/2
    - Difference = 2a + k > 0, so no collisions.
    
    We verify this computationally to be safe.
    """
    start_time = time.time()
    
    # The optimal set: arithmetic progression from 226 to 256 (31 elements)
    k = 31
    x_1 = 226
    result = list(range(x_1, x_1 + k))
    
    # Validate the result meets all constraints
    # 1. All integers are in 1..256
    assert all(1 <= x <= 256 for x in result)
    
    # 2. All integers are distinct
    assert len(set(result)) == len(result)
    
    # 3. Check cardinality-separated subset sum property
    # For each cardinality, compute the range of possible sums
    # and verify ranges for different cardinalities don't overlap
    
    from itertools import combinations
    
    # Compute min and max sum for each cardinality
    min_sums = {}
    max_sums = {}
    
    for c in range(k + 1):
        if c == 0:
            min_sums[c] = 0
            max_sums[c] = 0
        else:
            # Min sum is sum of c smallest elements
            min_sums[c] = sum(result[:c])
            # Max sum is sum of c largest elements
            max_sums[c] = sum(result[-c:])
    
    # Verify no overlap between sums of different cardinalities
    # If max_sum[c1] < min_sum[c2] or max_sum[c2] < min_sum[c1] for c1 != c2, no collision
    # Actually, we need: for any c1 != c2, max_sum[c1] < min_sum[c2] OR max_sum[c2] < min_sum[c1]
    # This is satisfied if the ranges [min_sum[c], max_sum[c]] for different c are disjoint
    
    # More directly: check that for any c1 < c2, max_sum[c1] < min_sum[c2]
    for c1 in range(k + 1):
        for c2 in range(c1 + 1, k + 1):
            if max_sums[c1] >= min_sums[c2]:
                raise AssertionError(
                    f"Collision: max sum of {c1} elements ({max_sums[c1]}) >= "
                    f"min sum of {c2} elements ({min_sums[c2]})"
                )
    
    # Time guard: if we've exceeded safe time, return immediately
    elapsed = time.time() - start_time
    if elapsed > 85:  # 5-second safety margin
        return result
    
    return result
# EVOLVE-BLOCK-END
