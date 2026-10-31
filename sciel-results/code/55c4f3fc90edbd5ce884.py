import zlib

def priority(vector, n):
    # Count the number of 1s in the vector
    # Vectors with fewer 1s are prioritized
    # This avoids the "center" of the space which is dense with 3-term APs
    count_ones = 0
    for x in vector:
        if x == 1:
            count_ones += 1
    
    # Base priority is negative count of ones
    # Higher priority (less negative) is better
    # We want 0 ones to be highest
    base_priority = -count_ones * 1000.0
    
    # Add a small deterministic hash for tie-breaking
    # This ensures a consistent order for vectors with same count
    # Use a simple linear hash for speed
    h = 0
    for x in vector:
        h = (h * 31 + x) & 0xFFFFFFFF
    
    return base_priority + (h / 0xFFFFFFFF)
