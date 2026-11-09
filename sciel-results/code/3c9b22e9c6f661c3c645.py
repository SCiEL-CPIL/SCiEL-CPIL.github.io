"""Cap set greedy construction with improved static priority heuristic."""
import time
import math
import random
import hashlib
import struct
from itertools import product

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 280  # seconds, with 20s safety margin before 300s limit

def priority(vector, n):
    """
    Static priority function for greedy cap set construction.
    
    Prioritize vectors that are balanced in digit distribution,
    with tie-breakers that encourage diversity and spatial spread.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    
    # Primary: minimize deviation from balanced distribution
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Secondary: sum of squared digits (encourages spread)
    sq_sum = sum(v * v for v in vector)
    
    # Tertiary: spatial spread of 1-positions (breaks ties, reduces local clustering)
    ones_positions = [i for i, v in enumerate(vector) if v == 1]
    if len(ones_positions) >= 2:
        # Sum of pairwise distances between 1-positions
        spatial_spread = 0.0
        for i in range(len(ones_positions)):
            for j in range(i+1, len(ones_positions)):
                spatial_spread += abs(ones_positions[i] - ones_positions[j])
    else:
        spatial_spread = 0.0
    
    # Quaternary: Hash-based tie-breaker (LSH-like effect)
    # Using a cryptographic hash to generate a pseudo-random float.
    # This spreads out vectors that are "close" in Hamming distance
    # across the priority list, helping to break symmetries and
    # potentially avoid local clustering of conflicting vectors.
    data = b''.join(bytes([v]) for v in vector)
    h = hashlib.md5(data).digest()
    hash_val = struct.unpack('<Q', h[:8])[0] / (2**64)
    
    return -penalty * 10000.0 + sq_sum * 100.0 + spatial_spread * 10.0 + hash_val
