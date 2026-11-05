import time
import random
import sys

# Increase recursion limit for deep backtracking
sys.setrecursionlimit(10000)

N = 9
SIZE = 1 << N
MAX_TIME = 145.0  # Safety margin

# Precompute neighbor masks
# neighbor_mask[v] is a bitmask where bit u is set if v and u are adjacent
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = mask

# Precompute list of neighbors for iteration
NEIGHBORS = []
for v in range(SIZE):
    nbrs = []
    for b in range(N):
        nbrs.append(v ^ (1 << b))
    NEIGHBORS.append(nbrs)

# Global state for search
best_path = [0]
best_len = 1
stop_flag = False
start_time = 0.0

def construct():
    global best_path, best_len, stop_flag, start_time
    
    start_time = time.time()
    stop_flag = False
    best_len = 1
    best_path = [0]
    
    # Use a fixed seed for reproducibility in case of ties, 
    # but we will vary the seed for each "attempt" if we restart.
    # However, a single long run with random tie-breaking is usually best.
    
    # We will run a single DFS with random tie-breaking.
    # If it finishes quickly, we can start another one with a different seed.
    
    # Define the DFS function inside construct to access local variables if needed,
    # but using globals is faster in Python.
    
    def dfs(current, used, path, depth):
        global best_path, best_len, stop_flag
        
        if stop_flag:
            return
        
        if depth + 1 > best_len:
            best_len = depth + 1
            best_path = path[:]
            
        if used == (1 << SIZE) - 1:
            return
            
        candidates = []
        for v in NEIGHBORS[current]:
            if used & (1 << v):
                continue
            # Check if v is adjacent to any other vertex in used (besides current)
            if (NEIGHBOR_MASK[v] & used) == (1 << current):
                candidates.append(v)
                
        if not candidates:
            return
            
        scored_candidates = []
        for v in candidates:
            next_used = used | (1 << v)
            count = 0
            for u in NEIGHBORS[v]:
                if next_used & (1 << u):
                    continue
                if (NEIGHBOR_MASK[u] & next_used) == (1 << v):
                    count += 1
            # Random noise to break ties
            noise = random.random() * 0.1
            scored_candidates.append((count + noise, v))
            
        scored_candidates.sort(key=lambda x: x[0])
        
        for score, v in scored_candidates:
            if stop_flag:
                return
                
            path.append(v)
            dfs(v, used | (1 << v), path, depth + 1)
            path.pop()
            
            if stop_flag:
                return

    # Run DFS
    # We can try multiple starting vertices if we have time, but 0 is sufficient.
    # We can also try to restart if the first run finishes early.
    
    while time.time() - start_time < MAX_TIME:
        if time.time() - start_time > MAX_TIME:
            break
            
        # Reset for a new attempt?
        # If the previous DFS ran out of candidates (dead end) and backtracked all the way,
        # it might have finished. 
        # But usually it will hit the time limit.
        
        # To diversify, we can shuffle the global NEIGHBORS? No, that's expensive.
        # The random.random() inside dfs provides the diversity.
        
        path = [0]
        used = 1
        
        dfs(0, used, path, 0)
        
        # If dfs returned quickly (before time limit), we can try again with a new seed
        # or just break if we are close to the limit.
        if time.time() - start_time > MAX_TIME - 5:
            break

    return best_path

# Ensure the function signature matches
# def construct():
#     ...
#     return list
