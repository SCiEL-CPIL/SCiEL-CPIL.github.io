import random
import time

def construct():
    best_set = []
    best_size = 0
    start_time = time.time()
    time_limit = 110  # leave some buffer
    
    # Try different orderings to escape local optima of the greedy choice
    # 0: Random, 1: Descending, 2: Ascending, 3: Random (different seed)
    
    for seed in range(500):
        if time.time() - start_time > time_limit:
            break
        
        strategy = seed % 4
        random.seed(seed)
        candidates = list(range(1, 1001))
        
        if strategy == 0:
            random.shuffle(candidates)
        elif strategy == 1:
            candidates.sort(reverse=True)
        elif strategy == 2:
            candidates.sort()
        else:
            random.shuffle(candidates)
            
        chosen = []
        counts = [0] * 2001 # indices 0..2000
        
        for x in candidates:
            ok = True
            c = counts
            ch = chosen
            
            for y in ch:
                s = x + y
                if c[s] >= 2:
                    ok = False
                    break
            
            if ok:
                s = 2 * x
                if c[s] >= 2:
                    ok = False
            
            if ok:
                ch.append(x)
                # Update counts for pairs (x, y) where y was in chosen before x
                # ch now has x at the end, so ch[:-1] are the previous elements
                # Using index loop to avoid slicing overhead
                for i in range(len(ch) - 1):
                    c[x + ch[i]] += 1
                c[2 * x] += 1
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
    
    return best_set
