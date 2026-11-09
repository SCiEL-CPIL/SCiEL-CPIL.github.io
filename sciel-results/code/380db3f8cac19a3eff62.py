import time
import random
import math

def construct():
    n = 16
    m = 65
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Helper to calculate max triangle degree sum
    def calc_score(neighbors, degrees):
        max_sum = 0
        # Iterate over all triangles
        # Using bitmasks might be faster, but sets are fine for n=16
        # neighbors[u] is a set
        for u in range(n):
            for v in neighbors[u]:
                if v > u:
                    # Common neighbors of u and v
                    # Intersection of neighbors[u] and neighbors[v]
                    # But we need w such that u-v-w is triangle
                    # w must be in neighbors[u] AND neighbors[v]
                    # And w != u, w != v (guaranteed by simple graph)
                    # Actually, if u-v is edge, and w in N(u) and w in N(v), then u-v-w is triangle.
                    # We just need to iterate w in intersection.
                    # To avoid double counting, we can just check all triples u<v<w
                    pass
        
        # Simpler O(n^3) approach
        for u in range(n):
            for v in range(u + 1, n):
                if v in neighbors[u]:
                    for w in range(v + 1, n):
                        if w in neighbors[u] and w in neighbors[v]:
                            s = degrees[u] + degrees[v] + degrees[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    # Initialize best solution
    # Start with K_{8,8} + edge (0,1)
    # Vertices 0-7 and 8-15
    neighbors = [set() for _ in range(n)]
    degrees = [0] * n
    
    def add_edge(u, v):
        neighbors[u].add(v)
        neighbors[v].add(u)
        degrees[u] += 1
        degrees[v] += 1
        
    def remove_edge(u, v):
        neighbors[u].remove(v)
        neighbors[v].remove(u)
        degrees[u] -= 1
        degrees[v] -= 1

    # Build K_{8,8}
    for u in range(8):
        for v in range(8, 16):
            add_edge(u, v)
    # Add one edge inside part 0
    add_edge(0, 1)
    
    current_score = calc_score(neighbors, degrees)
    best_score = current_score
    best_neighbors = [s.copy() for s in neighbors]
    best_degrees = degrees.copy()
    
    # Simulated Annealing
    T = 10.0
    cool_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random edge to remove
        # We need a list of current edges
        # Rebuilding edge list is O(m). Or we can pick random u, v and check if edge exists.
        # Picking random u, v and checking is O(1).
        # Probability of hitting an edge is m / (n*(n-1)/2) = 65 / 120 ~ 0.54
        # So it's efficient enough.
        
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if v not in neighbors[u]:
            continue
            
        # Pick a random non-edge to add
        x = random.randint(0, n - 1)
        y = random.randint(0, n - 1)
        if x == y:
            continue
        if y in neighbors[x]:
            continue
            
        # Perform move
        remove_edge(u, v)
        add_edge(x, y)
        
        new_score = calc_score(neighbors, degrees)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_neighbors = [s.copy() for s in neighbors]
                best_degrees = degrees.copy()
        else:
            # Accept with probability
            delta = new_score - current_score
            if delta < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / T)
                
            if not accept:
                # Revert
                remove_edge(x, y)
                add_edge(u, v)
            else:
                current_score = new_score
                
        T *= cool_rate
        if T < 0.1:
            T = 0.1

    # Convert best_neighbors to edge list
    edges = []
    for u in range(n):
        for v in best_neighbors[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
