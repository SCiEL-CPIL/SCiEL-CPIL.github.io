import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def get_edges(adj):
        edges = []
        n = adj.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    edges.append([i, j])
        return edges

    def is_regular_subset(subset, adj_rows):
        k = len(subset)
        if k < 2:
            return True
        # Compute degrees within the subset
        degs = [0] * k
        for i, u in enumerate(subset):
            d = 0
            # Sum adjacency to other vertices in subset
            for j, v in enumerate(subset):
                if i != j:
                    d += adj_rows[u][v]
            degs[i] = d
        
        first_deg = degs[0]
        for d in degs[1:]:
            if d != first_deg:
                return False
        return True

    def find_bad_subset(adj, n):
        vertices = list(range(n))
        adj_rows = [adj[i] for i in range(n)]
        
        # Check size 6 first
        for subset in combinations(vertices, 6):
            if is_regular_subset(subset, adj_rows):
                return subset
                
        # Check size 7
        for subset in combinations(vertices, 7):
            if is_regular_subset(subset, adj_rows):
                return subset
                
        # Check size 8 to n
        for k in range(8, n + 1):
            for subset in combinations(vertices, k):
                if is_regular_subset(subset, adj_rows):
                    return subset
                    
        return None

    def solve_for_n(n, time_limit):
        # Multiple restarts
        while time.time() < start_time + time_limit:
            adj = np.random.randint(0, 2, (n, n))
            adj = (adj + adj.T) % 2
            np.fill_diagonal(adj, 0)
            
            # Local search
            while time.time() < start_time + time_limit:
                bad_subset = find_bad_subset(adj, n)
                if bad_subset is None:
                    return n, adj
                
                # Flip a random edge in the bad subset
                u, v = random.sample(bad_subset, 2)
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = adj[u, v]
                
        return None, None

    # Strategy:
    # 1. Try to find n=20.
    # 2. If found and time remains, try n=21.
    # 3. If n=20 not found, try n=19 as fallback.
    
    # Allocate 80% of time to n=20
    time_for_20 = (deadline - time.time()) * 0.8
    if time_for_20 > 10:
        n_found, adj_found = solve_for_n(20, time_for_20)
        if n_found is not None:
            # Check time for n=21
            time_left = deadline - time.time()
            if time_left > 20:
                n_found_21, adj_found_21 = solve_for_n(21, time_left - 5.0)
                if n_found_21 is not None:
                    return {'n': 21, 'edges': get_edges(adj_found_21)}
            return {'n': 20, 'edges': get_edges(adj_found)}
            
    # Fallback: Try n=19
    time_left = deadline - time.time()
    if time_left > 10:
        n_found, adj_found = solve_for_n(19, time_left - 5.0)
        if n_found is not None:
            return {'n': 19, 'edges': get_edges(adj_found)}
            
    # Trivial fallback
    return {'n': 5, 'edges': []}
