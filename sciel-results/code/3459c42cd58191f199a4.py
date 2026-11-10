import numpy as np
import time
import random
from math import gcd, isqrt

def construct():
        start_time = time.time()
        limit = 512
        # Precompute compatibility
        # compat[i, j] is True if lcm(i+1, j+1) <= 512
        
        # Use a list of lists or numpy array
        # N=512 is small enough for O(N^2) precomputation
        
        # 1. Precomputation
        compat = np.zeros((limit, limit), dtype=bool)
        for i in range(limit):
            a = i + 1
            for j in range(i, limit):
                b = j + 1
                l = (a * b) // gcd(a, b)
                if l <= limit:
                    compat[i, j] = True
                    compat[j, i] = True
        
        # 2. Search
        best_set = set(range(1, 23)) # Seed
        
        # Greedy expansion
        def expand(S):
            S = set(S)
            changed = True
            while changed:
                changed = False
                for v in range(1, limit + 1):
                    if v not in S:
                        # Check compatibility
                        ok = True
                        for u in S:
                            if not compat[u-1, v-1]:
                                ok = False
                                break
                        if ok:
                            S.add(v)
                            changed = True
            return S
        
        best_set = expand(best_set)
        
        # Local Search / Random Restarts
        # Try different seeds
        seeds = [
            list(range(1, 23)),
            list(range(1, 20)),
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], # Powers of 2
            # Divisors of 480
            [d for d in range(1, 481) if 480 % d == 0],
            # Divisors of 360
            [d for d in range(1, 361) if 360 % d == 0],
            # Random sets
        ]
        
        for _ in range(50): # 50 random restarts
            # Create random seed
            size = random.randint(5, 15)
            seed_set = set(random.sample(range(1, limit + 1), size))
            # Filter to valid set first? No, just expand from scratch is safer
            # Actually, expand from empty is slow if we iterate 1..512.
            # Better: Randomly pick valid pairs?
            
            # Just use greedy from a random subset
            s = expand(seed_set)
            if len(s) > len(best_set):
                best_set = s
        
        return list(best_set)
