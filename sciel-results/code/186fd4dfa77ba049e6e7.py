import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85.0
    best_set = []
    N = 4096
    MAX_SUM = 3 * N
    
    def try_add(x, S, used_sums):
        s = 3 * x
        if used_sums[s]:
            return False
        
        # Check 2*x + y and x + 2*y
        # Use local reference for speed
        us = used_sums
        for y in S:
            s1 = 2 * x + y
            if us[s1]:
                return False
            s2 = x + 2 * y
            if us[s2]:
                return False
        
        n = len(S)
        for i in range(n):
            y = S[i]
            for j in range(i + 1, n):
                z = S[j]
                s3 = x + y + z
                if us[s3]:
                    return False
                
        us[s] = True
        for y in S:
            us[2 * x + y] = True
            us[x + 2 * y] = True
        
        for i in range(n):
            y = S[i]
            for j in range(i + 1, n):
                z = S[j]
                us[x + y + z] = True
                
        S.append(x)
        return True

    used_sums = np.zeros(MAX_SUM + 1, dtype=bool)
    
    # 3000 seeds
    for seed in range(3000):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        S = []
        used_sums[:] = False
        
        mode = seed % 3
        
        if mode == 0:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        elif mode == 1:
            base = random.choice([2, 3, 4, 5, 6, 7])
            geo = []
            val = 1
            while val <= N:
                geo.append(val)
                val *= base
            for g in sorted(geo):
                try_add(g, S, used_sums)
            
            rest = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(rest)
            candidates = rest
        else:
            large_candidates = random.sample(range(N//2, N + 1), 5)
            large_candidates.sort()
            for x in large_candidates:
                try_add(x, S, used_sums)
            
            rest = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(rest)
            candidates = rest

        for x in candidates:
            try_add(x, S, used_sums)
            
            if len(S) > len(best_set):
                best_set = list(S)

    return best_set
