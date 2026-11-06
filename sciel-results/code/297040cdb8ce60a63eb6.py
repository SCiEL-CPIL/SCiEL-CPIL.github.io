import random
import time
from itertools import combinations

def construct():
    def generate_random_regular(n, d, seed):
        """
        Generate a random d-regular graph on n vertices using the configuration model.
        Returns adjacency list as bitmasks or None if failed.
        """
        random.seed(seed)
        if (n * d) % 2 != 0:
            return None
        
        stubs = []
        for i in range(n):
            stubs.extend([i] * d)
        
        random.shuffle(stubs)
        
        # Pair up stubs
        adj = [0] * n
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None # Self-loop
            if adj[u] & (1 << v):
                return None # Multi-edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        return adj

    def is_valid(n, adj):
        # Check all subsets of size >= 6
        # Optimization: Early break on non-regular
        for mask in range(1, 1 << n):
            k = mask.bit_count()
            if k < 6:
                continue
            
            # Get degree of first vertex in mask
            lsb = mask & -mask
            v0 = lsb.bit_length() - 1
            d0 = (adj[v0] & mask).bit_count()
            
            # Check if all other vertices in mask have same degree
            m = mask ^ lsb
            is_reg = True
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m -= lsb
                if (adj[v] & mask).bit_count() != d0:
                    is_reg = False
                    break
            
            if is_reg:
                return False
        
        return True

    def edges_from_adj(n, adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    start_time = time.time()
    max_time = 110.0
    
    # Try n=21
    # Degrees to try: 9, 10, 11 (since n=21, max degree 20)
    # Random regular graphs are good candidates for "no regular induced subgraph"
    # because they are "locally random".
    
    seed = 0
    found_21 = False
    while time.time() - start_time < max_time and not found_21:
        for d in [9, 10, 11, 8, 12]:
            # Generate a few graphs for this degree
            for attempt in range(5):
                if time.time() - start_time > max_time:
                    break
                adj = generate_random_regular(21, d, seed)
                seed += 1
                if adj is None:
                    continue
                if is_valid(21, adj):
                    edges = edges_from_adj(21, adj)
                    return {'n': 21, 'edges': edges}
        
        # If we've tried a lot, maybe increase range or change strategy
        if seed % 100 == 0 and time.time() - start_time > 30:
             # Try some non-regular random graphs as well?
             # But regular is a strong candidate.
             pass

    # Fallback to n=20
    seed = 10000
    while time.time() - start_time < max_time:
        for d in [9, 10, 11]:
            for attempt in range(5):
                if time.time() - start_time > max_time:
                    break
                adj = generate_random_regular(20, d, seed)
                seed += 1
                if adj is None:
                    continue
                if is_valid(20, adj):
                    edges = edges_from_adj(20, adj)
                    return {'n': 20, 'edges': edges}

    # Fallback to known good graph or simple construction
    # A path or cycle is not regular induced? 
    # A path on 20 vertices: induced subgraph on 6 vertices is a path, not regular.
    # But we want to maximize n.
    # Just return a small valid graph if all else fails
    return {'n': 5, 'edges': []}
