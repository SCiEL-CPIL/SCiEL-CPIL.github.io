import time
import math
import random
import zlib
from itertools import product

_START_TIME = time.time()
TIME_LIMIT = 280

def priority(vector, n):
    """
    Refined time-varying randomized hash-based priority.
    Uses zlib.crc32 for better hash distribution and a smooth time-varying 
    weight to explore different regions of the vector space.
    """
    elapsed = time.time() - _START_TIME
    
    # Smooth time variation for weights
    t = elapsed * 0.5
    
    # High-quality hash using zlib.crc32
    # Convert vector to bytes for hashing
    vec_bytes = bytes(vector)
    
    # Use multiple CRCs with different initial values to decorrelate
    h1 = zlib.crc32(vec_bytes) & 0xFFFFFFFF
    h2 = zlib.crc32(vec_bytes, int(elapsed)) & 0xFFFFFFFF
    h3 = zlib.crc32(bytes([int(elapsed * 10) % 256]) + vec_bytes) & 0xFFFFFFFF
    
    # Combine hashes with smooth time-varying weights
    w1 = 1.0 + 0.5 * math.sin(t)
    w2 = 1.0 + 0.5 * math.cos(t * 1.1)
    w3 = 1.0 + 0.5 * math.sin(t * 0.9)
    
    total_weight = w1 + w2 + w3
    combined_hash = (w1 * h1 + w2 * h2 + w3 * h3) / total_weight
    
    # Balance component
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    balance_penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Score: Hash dominates, balance is a soft preference
    score = (combined_hash / 4294967295.0) - 0.02 * balance_penalty
    
    return score
