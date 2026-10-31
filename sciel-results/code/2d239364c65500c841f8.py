"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
import random

D = 11

# EVOLVE-BLOCK-START

def generate_candidates():
    """Generate weight-2 integer vectors."""
    candidates = []
    
    # Weight 2: Two non-zero entries, each +/- 1
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(np.array(v, dtype=np.int64))
                
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    candidates = generate_candidates()
    
    # For weight-2 vectors, all distinct vectors are compatible.
    # Proof: Norm^2 = 2. Min dist^2 for u != v is 2 (when dot=1).
    # Condition: dist >= max_norm => sqrt(2) >= sqrt(2). Holds.
    
    # We can just return all of them.
    # To be safe against any subtle floating point or evaluator quirks, 
    # we can verify a random sample, but 220 is small enough to verify all pairs if needed.
    # But 220*220/2 = 24k checks, very fast.
    
    n = len(candidates)
    # Optional: Verify validity to be 100% sure
    # But we are confident. 
    # Let's just return the array.
    
    return np.array(candidates, dtype=np.int64)

# EVOLVE-BLOCK-END
