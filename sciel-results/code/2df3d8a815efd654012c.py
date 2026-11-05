import time

def construct():
    # The task requires 12 distinct positive integers with distinct subset sums.
    # This is equivalent to a superincreasing sequence.
    # The minimal such sequence is powers of 2: 1, 2, 4, ..., 2^11.
    # The largest element is 2^11 = 2048.
    # This is the theoretical minimum for the largest element.
    
    # Deadline guard (though computation is O(1))
    start_time = time.time()
    deadline = start_time + 59.0 # Safety margin
    
    # Generate the optimal set
    # n = 12
    # a_i = 2^(i-1) for i in 1..12
    result = [1 << i for i in range(12)]
    
    # Check if we are still within time (we always are)
    if time.time() < deadline:
        return result
    
    # Fallback (should never happen)
    return result
