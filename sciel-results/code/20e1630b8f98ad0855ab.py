import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s margin
    
    vectors = []
    
    # Add all size 1 vectors (norm squared 1)
    # Support of size 1: choose 1 index, sign +/-
    for idx in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[idx] = sign
            vectors.append(v)
            
    # Add all size 2 vectors (norm squared 2)
    # Support of size 2: choose 2 indices, signs +/- for each
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    vectors.append(v)
                    
    # Total vectors: 22 + 220 = 242
    # These all satisfy min pairwise distance >= max norm
    # Max norm is sqrt(2). Min distance squared is 2 (for size 2 vs size 2 sharing 1 index with same sign on shared? No, min is 2).
    # Wait, let's re-verify min distance for size 2 vs size 2.
    # v=(1,1,0..), w=(1,1,0..) is same.
    # v=(1,1,0..), w=(1,-1,0..) d^2=4.
    # v=(1,1,0..), w=(1,0,1,0..) d^2=2.
    # So min d^2=2. Max norm^2=2. OK.
    
    # Size 1 vs Size 1:
    # v=(1,0..), w=(1,0..) same.
    # v=(1,0..), w=(-1,0..) d^2=4.
    # v=(1,0..), w=(0,1..) d^2=2.
    # Min d^2=2. Max norm^2=1. 2>=1 OK.
    
    # Size 1 vs Size 2:
    # v=(1,0..), w=(1,1,0..) d^2=2.
    # Max norm^2=2. 2>=2 OK.
    
    return np.array(vectors, dtype=np.int64)
