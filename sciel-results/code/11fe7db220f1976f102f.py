import random
import time

def construct():
    # The task is to maximize n for a C4-free graph on n vertices
    # with minimum degree at least n - 39.
    # This is equivalent to finding a graph H (complement) on n vertices
    # with max degree <= 38 and |N(u) intersect N(v)| <= 1 for all u != v.
    # We use a greedy edge addition algorithm on G (the C4-free graph).
    # Adding edge (u, v) to G is allowed if u and v have fewer than 2 common neighbors.
    
    best_result = {'n': 39, 'edges': []}
    
    start_time = time.time()
    # Set a safety deadline, e.g., 100 seconds
    deadline = 100.0
    
    # We want to maximize n, so we try from 47 downwards.
    # The known answer is 46 or 47.
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts to find a valid graph
        # A valid graph has min degree >= n - 39
        target_min_deg = n - 39
        
        for attempt in range(1000):
            if time.time() > deadline:
                break
            
            # Initialize adjacency sets
            neighbors = [set() for _ in range(n)]
            edges = []
            
            # Create list of all possible edges
            possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(possible_edges)
            
            # Greedily add edges
            for u, v in possible_edges:
                # A C4 is formed if u and v have >= 2 common neighbors
                # because u-w1-v-w2-u would be a C4
                common = neighbors[u] & neighbors[v]
                if len(common) < 2:
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v])
            
            # Check if the degree condition is met
            min_deg = min(len(nei) for nei in neighbors)
            
            if min_deg >= target_min_deg:
                # We found a valid graph with this n
                return {'n': n, 'edges': edges}
        
        # If we exhausted attempts for this n, try smaller n

    # If no graph found for n > 39, return the seed
    return best_result
