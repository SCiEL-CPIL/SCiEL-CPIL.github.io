import time

_START_TIME = time.time()

def priority(vector, n):
    # Calculate zero count
    c0 = 0
    total_sum = 0
    for v in vector:
        if v == 0:
            c0 += 1
        else:
            total_sum += v
            
    # Strong penalty for zeros to ensure core {1,2}^n is added first
    # This guarantees a cap set of size 2^n
    # Then we add zero-vectors.
    
    # For zero-vectors, we want a specific order.
    # Let's try prioritizing vectors with HIGHER sum (fewer zeros / larger values)
    # This means vectors with 1 zero are added before vectors with 2 zeros?
    # No, c0 is in the penalty.
    
    # If we use c0 as primary key, all c0=0 are first.
    # Then c0=1.
    # Within c0=1, sort by sum?
    
    # Let's use a large multiplier for c0 to dominate.
    # And use total_sum as secondary.
    
    # To favor higher sum (denser values), we add total_sum.
    # To favor lower c0, we subtract large * c0.
    
    return 1000.0 * total_sum - 100000.0 * c0
