import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Try to find a valid graph for n from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random instances for this n
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Generate a random triangle-free graph
            # Strategy: Start with a random bipartite graph
            # This ensures no triangles.
            # Split vertices into two sets A and B
            # Randomly connect vertices in A to vertices in B
            # This is a bipartite graph, so triangle-free.
            # However, bipartite graphs can have large independent sets.
            # We need to check alpha.
            
            # To get denser graphs, we can try to add edges within A or B?
            # No, that creates triangles if there's a path of length 2.
            # So we must stick to bipartite or carefully add edges.
            # Let's stick to random bipartite graphs first.
            
            # Random split
            split = n // 2
            A = list(range(split))
            B = list(range(split, n))
            
            # Random edges between A and B
            # Probability p
            # We want high density to reduce alpha
            # Let's try p = 0.5, 0.6, 0.7
            p = random.choice([0.5, 0.6, 0.7, 0.8])
            
            adj = np.zeros((n, n), dtype=bool)
            edges = []
            
            for u in A:
                for v in B:
                    if random.random() < p:
                        adj[u, v] = True
                        adj[v, u] = True
                        edges.append([u, v])
            
            # Check if alpha < 10
            # Use a fast heuristic or exact solver
            # For n <= 40, exact solver is feasible
            if check_alpha_less_than_10(adj, n):
                return {'n': n, 'edges': edges}
            
            # If not, we could try to modify the graph
            # But for now, just try another random graph
    
    # Fallback to the seed graph (n=26)
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

def check_alpha_less_than_10(adj, n):
    # Branch and bound for maximum independent set
    # If max IS < 10, return True
    # If max IS >= 10, return False
    
    # Use bitsets for speed
    # adj[i] is a bitset of neighbors of i
    
    # Precompute bitsets
    # Python integers are arbitrary precision, good for bitsets
    # But n=40, so we can use int
    
    neighbors = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i, j]:
                mask |= (1 << j)
        neighbors[i] = mask
    
    # Branch and bound
    # State: current set of vertices (bitmask), current size, best found
    # We want to find if there is an IS of size 10
    # If we find one, return False
    # If we exhaust search without finding size 10, return True
    
    # Optimization: If we can greedily find an IS of size 10, return False quickly
    # Greedy: Pick vertex with min degree, add to IS, remove it and its neighbors
    # Repeat until no vertices left
    
    # But greedy might miss.
    # Let's do a simple recursive search with pruning
    
    best_size = 0
    found_large = [False]
    
    def dfs(vertices_mask, current_size):
        if found_large[0]:
            return
        if current_size >= 10:
            found_large[0] = True
            return
        
        # If no vertices left
        if vertices_mask == 0:
            return
        
        # Pruning: If current_size + popcount(vertices_mask) <= best_size, return
        # But we are looking for size 10, not max.
        # So if current_size + popcount(vertices_mask) < 10, return
        
        # Pick a vertex
        # Heuristic: Pick vertex with min degree in the induced subgraph?
        # Or just the first one
        # To speed up, pick a vertex with fewest neighbors in the remaining set
        
        # Get list of vertices in mask
        # This is slow if done every time.
        # Alternative: Iterate through bits
        
        # Let's try a simple approach:
        # If popcount(vertices_mask) + current_size < 10:
        #    return
        
        v_count = bin(vertices_mask).count('1')
        if current_size + v_count < 10:
            return
            
        # Pick a vertex v
        # Find the first set bit
        # Or pick one with min degree
        # Let's pick the first one for simplicity, but order might matter
        # Better: Pick vertex with min degree in the subgraph
        
        # To find min degree, we need to count neighbors for each v in mask
        # This is O(n^2) per node, which is expensive
        
        # Let's just pick the first vertex
        # Find index of least significant set bit
        lsb = vertices_mask & -vertices_mask
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        # New mask: remove v and its neighbors
        new_mask = vertices_mask & ~(neighbors[v] | lsb)
        dfs(new_mask, current_size + 1)
        
        if found_large[0]:
            return
            
        # Branch 2: Exclude v
        new_mask = vertices_mask & ~lsb
        dfs(new_mask, current_size)
        
    dfs((1 << n) - 1, 0)
    
    return not found_large[0]
