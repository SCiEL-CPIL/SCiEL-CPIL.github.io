import numpy as np
import itertools
import time

D = 11

def construct():
    # Generate all vectors in D11 with squared norm 2
    # These are of the form +/- e_i +/- e_j for i != j
    vecs = []
    for i in range(D):
        for j in range(i + 1, D):
            # 4 sign combinations: (+, +), (+, -), (-, +), (-, -)
            # v[i] = s1, v[j] = s2
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    vecs.append(v)
                    
    # Convert to numpy array
    V = np.array(vecs, dtype=np.int64)
    
    # Verify constraints just in case
    # Norm squared is 2 for all
    # Min distance squared between any two:
    # If they share no indices: dist^2 = 2 + 2 - 0 = 4 >= 2 (max norm squared)
    # If they share one index: e.g. e1-e2 and e1-e3. dot=1. dist^2 = 2+2-2=2 >= 2.
    # If they are same: not considered as pairwise distinct.
    # So min dist squared is 2. Max norm squared is 2.
    # Condition: min_dist >= max_norm => sqrt(2) >= sqrt(2). Holds.
    
    return V
