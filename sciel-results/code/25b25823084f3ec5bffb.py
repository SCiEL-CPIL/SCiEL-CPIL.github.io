import time
import random

N = 9
MAX_TIME = 115.0

# Precompute neighbors
NEIGHBORS = [[] for _ in range(1 << N)]
for v in range(1 << N):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def _greedy_snake(seed=0):
    random.seed(seed)
    path = [0]
    used = 1
    forbidden = 0  # Mask of vertices that are neighbors of any vertex in path[:-1]
    
    while True:
        current = path[-1]
        candidates = []
        for n in NEIGHBORS[current]:
            if (used >> n) & 1:
                continue
            if (forbidden >> n) & 1:
                continue
            candidates.append(n)
            
        if not candidates:
            break
            
        if len(candidates) == 1:
            c = candidates[0]
        else:
            # Heuristic: Pick candidate with most valid next-neighbors
            best_c = None
            best_score = -1
            # Precompute forbidden for the next step (neighbors of current path)
            forb_next_base = forbidden
            for n in NEIGHBORS[current]:
                forb_next_base |= (1 << n)
                
            for c in candidates:
                score = 0
                for n_next in NEIGHBORS[c]:
                    if (used >> n_next) & 1:
                        continue
                    # n_next cannot be c, and c is not in used yet, so used check is sufficient for 'used'
                    # We don't need to check if n_next is c because neighbors are distinct.
                    if (forb_next_base >> n_next) & 1:
                        continue
                    score += 1
                    
                if score > best_score:
                    best_score = score
                    best_c = c
            c = best_c
            
        path.append(c)
        used |= (1 << c)
        # Update forbidden: add neighbors of the vertex we just left (current)
        for n in NEIGHBORS[current]:
            forbidden |= (1 << n)
            
    return path

def construct():
    start_time = time.time()
    best_path = []
    best_len = 0
    
    # Initial run to ensure we have a valid path
    p = _greedy_snake(seed=42)
    best_path = p
    best_len = len(p)
    
    seed = 100
    while time.time() - start_time < MAX_TIME:
        p = _greedy_snake(seed=seed)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        seed += 1
        
    return best_path
