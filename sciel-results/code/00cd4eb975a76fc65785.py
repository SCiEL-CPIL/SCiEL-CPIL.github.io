import time
import sys

N = 9
MAX_VAL = 1 << N

def construct():
    start_time = time.time()
    TIME_LIMIT = 115.0
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_VAL)]
    for v in range(MAX_VAL):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    best_path = []
    best_len = 0
    
    sys.setrecursionlimit(10000)
    
    def backtrack(path, used, forbidden, current_time):
        nonlocal best_path, best_len
        
        if current_time > TIME_LIMIT:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        cur = path[-1]
        
        candidates = []
        for nb in neighbors[cur]:
            if nb in used or nb in forbidden:
                continue
            avail_count = 0
            for x in neighbors[nb]:
                if x not in used and x not in forbidden:
                    avail_count += 1
            candidates.append((avail_count, nb))
            
        if not candidates:
            return
            
        candidates.sort(key=lambda x: x[0])
        
        for _, nb in candidates:
            if time.time() > TIME_LIMIT:
                return
                
            used.add(nb)
            path.append(nb)
            
            new_forbidden = set(forbidden)
            if len(path) >= 2:
                prev = path[-2]
                for nb2 in neighbors[prev]:
                    new_forbidden.add(nb2)
            
            backtrack(path, used, new_forbidden, time.time())
            
            path.pop()
            used.remove(nb)
            
    path = [0]
    used = {0}
    forbidden = set()
    
    backtrack(path, used, forbidden, start_time)
    
    return best_path
