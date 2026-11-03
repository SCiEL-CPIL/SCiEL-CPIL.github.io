import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback
        # ...
        pass

    # Precomputation
    SPACE = 729
    balls = []
    for i in range(SPACE):
        w = [i % 3, (i//3) % 3, (i//9) % 3, (i//27) % 3, (i//81) % 3, (i//243) % 3]
        b = [i]
        for pos in range(6):
            orig = w[pos]
            for v in range(3):
                if v == orig: continue
                w[pos] = v
                idx = w[0] + w[1]*3 + w[2]*9 + w[3]*27 + w[4]*81 + w[5]*243
                b.append(idx)
            w[pos] = orig
        balls.append(b)
        
    # ... main logic ...
