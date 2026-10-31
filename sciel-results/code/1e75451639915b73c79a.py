import numpy as np
import hashlib

def priority(vector, n):
    # Convert to numpy array for fast operations
    v = np.array(vector, dtype=np.int8)
    
    # Heuristic: Minimize the number of 1s in the vector.
    # In the cap set hypergraph (where edges are arithmetic progressions),
    # coordinates equal to 1 allow for more potential partners (0,2) and (2,0)
    # compared to coordinates 0 or 2 which only pair with themselves.
    # Thus, vectors with fewer 1s are structurally "safer" and less likely
    # to cause conflicts with future additions.
    num_ones = np.sum(v == 1)
    
    # Base priority is negative count of ones (higher is better, so we minimize ones)
    base_priority = -float(num_ones)
    
    # Tie-breaking: Use a stable hash to ensure deterministic but non-lexicographic order
    # among vectors with the same number of ones.
    v_bytes = bytes(v)
    h = int(hashlib.md5(v_bytes).hexdigest(), 16)
    tie_break = (h % 10000) / 10000.0
    
    # Small weight for tie-breaking to not override the main heuristic
    return base_priority + tie_break * 0.01
