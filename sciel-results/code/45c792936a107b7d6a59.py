import time

def construct():
    """
    Return a nonempty subset of [1, 100] with no subset summing to 200.
    Maximize cardinality.
    
    The set {66, 67, ..., 100} has cardinality 35.
    - Minimum sum of any 3 distinct elements is 66 + 67 + 68 = 201 > 200.
    - Maximum sum of any 2 distinct elements is 99 + 100 = 199 < 200.
    - Maximum element is 100 < 200.
    Thus, no subset sums to 200.
    
    This construction is optimal: any subset of [1, 100] with cardinality >= 36
    must contain a subset summing to 200.
    """
    # Deadline guard for safety, though execution is O(1)
    start_time = time.time()
    deadline = start_time + 5.0  # 5 seconds is plenty for O(1)
    
    # The optimal set: all integers from 66 to 100 inclusive
    result = list(range(66, 101))
    
    # Check if we are near deadline (not really, but following directive)
    if time.time() > deadline:
        return result
        
    return result
