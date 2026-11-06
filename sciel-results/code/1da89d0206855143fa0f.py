import time
import random

def construct():
    # Try sizes from 35 down to 17
    for N in [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        start_time = time.time()
        time_limit = 100 if N > 30 else 50
        
        # Run local search
        adj, valid = local_search(N, time_limit)
        if valid:
            # Convert adj to edge list
            edges = []
            for i in range(N):
                for j in range(i + 1, N):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': N, 'edges': edges}
            
    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def local_search(N, time_limit):
    # ... implementation ...
    pass
