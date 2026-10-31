import zlib

def priority(vector, n):
    # Calculate L2 distance to the center (1, 1, ..., 1)
    dist = 0
    for v in vector:
        diff = v - 1
        dist += diff * diff
        
    # Calculate a pseudo-random tie-breaker
    # We want this to be in [0, 1)
    b = bytes(vector)
    h = zlib.crc32(b)
    rand_part = (h % 100000) / 100000.0
    
    # Priority: Higher is better.
    # We want to pick low distance vectors first.
    # So we negate the distance.
    # The random part breaks ties.
    # To ensure distance dominates, we scale the random part small?
    # No, if we just add rand_part, and dist is integer, 
    # vectors with dist 0 have priority ~ [0, 1).
    # Vectors with dist 1 have priority ~ [-1, 0).
    # So all dist 0 vectors are picked before any dist 1 vectors.
    # This is good.
    
    return -float(dist) + rand_part
