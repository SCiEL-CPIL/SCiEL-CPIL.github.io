import random
import math
import time

def construct():
    n = 16
    target_edges = 65
    
    # Helper to get score (max triangle degree sum)
    # adj is a list of n integers, where adj[i] is a bitmask of neighbors of i
    def get_score(adj):
        max_sum = 0
        # Precompute degrees
        degs = [bin(adj[i]).count('1') for i in range(n)]
        
        # Iterate over all potential triangles
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] >> j) & 1:
                    continue
                for k in range(j+1, n):
                    # Check if (i,k) and (j,k) exist
                    if (adj[i] >> k) & 1 and (adj[j] >> k) & 1:
                        s = degs[i] + degs[j] + degs[k]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # Initialize with the seed graph: K_{8,8} + {0,1}
    # Vertices 0-7 and 8-15.
    # Edges between 0-7 and 8-15.
    # Plus edge (0,1).
    adj = [0] * n
    for u in range(8):
        for v in range(8, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0)
    
    current_score = get_score(adj)
    best_score = current_score
    best_adj = adj[:]
    
    # Simulated Annealing
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    # Initial temperature
    T = 5.0
    T_min = 0.1
    alpha = 0.9995
    
    # We will try to find a graph with score <= 23
    # If we find 23, we can stop or continue to see if 22 is possible (unlikely)
    
    # Helper to pick a random edge
    def pick_random_edge(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        if not edges:
            return None
        return random.choice(edges)

    # Helper to pick a random non-edge
    def pick_random_non_edge(adj):
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    non_edges.append((i, j))
        if not non_edges:
            return None
        return random.choice(non_edges)

    # Optimization: Precompute list of all potential edges and non-edges?
    # Since n=16, total edges = 120.
    # It is faster to iterate through all pairs and check.
    
    # To speed up, we can maintain a list of current edges and non-edges?
    # Or just random sampling. With 120 pairs, sampling is fast.
    
    # Let's optimize the move generation.
    # Instead of picking from lists, we can pick random i, j.
    
    iterations = 0
    while time.time() - start_time < timeout:
        if T < T_min:
            break
            
        # 1. Pick a random edge to remove
        # We can just pick two distinct vertices and check if it's an edge.
        # Or maintain a list. Let's maintain a list for efficiency.
        
        # For simplicity and robustness, we'll use the list method but update it carefully.
        # Actually, with n=16, generating lists is cheap.
        # Let's just use random sampling of pairs.
        
        # Pick remove edge (u, v)
        while True:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and (adj[u] >> v) & 1:
                break
        
        # Pick add edge (x, y)
        while True:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x != y and not ((adj[x] >> y) & 1):
                # Ensure we are not adding the same edge we just removed
                if not ((x == u and y == v) or (x == v and y == u)):
                    break
        
        # Apply swap
        # Remove (u, v)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # Add (x, y)
        adj[x] |= (1 << y)
        adj[y] |= (1 << x)
        
        new_score = get_score(adj)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
                # If we reach 23, we can try to hold it or lower T faster?
                # Just let it run.
        else:
            # Revert swap
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            adj[x] &= ~(1 << y)
            adj[y] &= ~(1 << x)
            
        T *= alpha
        iterations += 1
        
        # If we found a very good score, we might want to do a local search polish?
        # The SA itself is a global search.
        
    # Final check
    if get_score(best_adj) != best_score:
        # Safety check, though best_adj is copied
        pass
        
    edges = get_edges_from_adj(best_adj)
    
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
