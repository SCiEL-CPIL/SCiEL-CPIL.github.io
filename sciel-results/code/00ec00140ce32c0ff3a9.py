import random
from itertools import combinations

def construct():
    # Build candidate pool
    pool = set()
    
    # Small integers
    for i in range(1, 101):
        pool.add(i)
    
    # Powers
    for base in [2, 3, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 48, 54, 64, 72, 80, 96, 108, 128, 144, 160, 192, 216, 256, 288, 320, 384, 432, 512, 576, 640, 768, 864, 1024, 1152, 1280, 1536, 1728, 2048, 2304, 2560, 3072, 3456, 4096, 4608, 5120, 6144, 6912, 8192, 9216, 10240, 12288, 13824, 16384, 18432, 20480, 24576, 27648, 32768, 36864, 40960, 49152, 55296, 65536, 73728, 81920, 98304, 110592, 131072, 147456, 163840, 196608, 221184, 262144, 294912, 327680, 393216, 442368, 524288, 589824, 655360, 786432, 884736, 1048576]:
        p = base
        while p <= 10000:
            pool.add(p)
            p *= base
            
    # Smooth numbers
    for i in range(15):
        for j in range(10):
            v = (2 ** i) * (3 ** j)
            if v <= 10000:
                pool.add(v)
                
    for i in range(15):
        for j in range(10):
            for k in range(8):
                v = (2 ** i) * (3 ** j) * (5 ** k)
                if v <= 10000:
                    pool.add(v)
                    
    for i in range(14):
        for j in range(9):
            for k in range(7):
                for l in range(5):
                    v = (2 ** i) * (3 ** j) * (5 ** k) * (7 ** l)
                    if v <= 10000:
                        pool.add(v)
                        
    pool = sorted([x for x in pool if 1 <= x <= 10000])
    pool_set = set(pool)
    
    def score(A):
        sums = set()
        prods = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                sums.add(ai + A[j])
                prods.add(ai * A[j])
        return max(len(sums), len(prods))
        
    # Heuristic Initialization
    candidates_init = [
        pool[:20],
        [2**i for i in range(20) if 2**i <= 10000][:20],
        [3**i for i in range(20) if 3**i <= 10000][:20] + pool[:20-len([3**i for i in range(20) if 3**i <= 10000])],
    ]
    
    best_set = None
    best_score = float('inf')
    
    for c in candidates_init:
        c = list(set(c))
        if len(c) < 20:
            for p in pool:
                if p not in c:
                    c.append(p)
                if len(c) == 20:
                    break
        c = c[:20]
        c.sort()
        s = score(c)
        if s < best_score:
            best_score = s
            best_set = c
            
    # Local Search
    random.seed(42)
    max_iters = 1000
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        indices = list(range(20))
        random.shuffle(indices)
        
        for idx in indices:
            for _ in range(10):
                cand = random.choice(pool)
                if cand in best_set:
                    continue
                
                new_set = list(best_set)
                new_set[idx] = cand
                
                s = score(new_set)
                if s < best_score:
                    best_set = new_set
                    best_score = s
                    improved = True
                    break
            
    return best_set
