import numpy as np
import hashlib

def priority(vector, n):
    # Convert to numpy array for fast operations
    v = np.array(vector, dtype=np.int8)
    
    # Heuristic: Prioritize vectors with weight close to n/2
    # This tends to create more "balanced" sets which often have higher independence numbers
    # in symmetric hypergraphs.
    weight = np.count_nonzero(v)
    target_weight = n / 2.0
    dist = abs(weight - target_weight)
    
    # Base priority is negative distance (higher is better)
    base_priority = -dist
    
    # Tie-breaking: Use a stable hash to ensure deterministic but non-lexicographic order
    # among vectors with the same weight distance.
    v_bytes = bytes(v)
    h = int(hashlib.md5(v_bytes).hexdigest(), 16)
    tie_break = (h % 10000) / 10000.0
    
    return base_priority + tie_break * 0.01
