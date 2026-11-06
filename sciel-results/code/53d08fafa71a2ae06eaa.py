import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 16
    
    # Grötzsch graph edges
    core_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 2), (6, 0),
        (7, 3), (7, 1),
        (8, 4), (8, 2),
        (9, 0), (9, 3),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    def make_adj_from_edges(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_triangle_free_add(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def close_graph(adj):
        # Return a new adjacency list with all possible triangle-free edges added
        new_adj = adj[:]
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (new_adj[u] >> v) & 1:
                        if is_triangle_free_add(new_adj, u, v):
                            new_adj[u] |= (1 << v)
                            new_adj[v] |= (1 << u)
                            changed = True
        return new_adj

    def count_edges(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def is_not_3_colorable(adj):
        # Check if graph is not 3-colorable using backtracking with bitmasks
        # Returns True if NOT 3-colorable (i.e., chromatic number >= 4)
        color = [-1] * n
        uncolored = (1 << n) - 1
        
        def backtrack(uncolored_mask):
            if uncolored_mask == 0:
                return True
            
            # Pick the vertex with the fewest available colors (MRV heuristic)
            best_v = -1
            best_avail = -1
            for v in range(n):
                if (uncolored_mask >> v) & 1:
                    # Available colors are those not used by neighbors
                    forbidden = 0
                    for u in range(n):
                        if (adj[v] >> u) & 1 and color[u] >= 0:
                            forbidden |= (1 << color[u])
                    avail = 7 & ~forbidden  # 3 colors: 0, 1, 2
                    if avail == 0:
                        return False
                    if best_v == -1 or bin(avail).count('1') < bin(best_avail).count('1'):
                        best_v = v
                        best_avail = avail
                        if bin(avail).count('1') == 1:
                            break
            
            for c in range(3):
                if (best_avail >> c) & 1:
                    color[best_v] = c
                    if backtrack(uncolored_mask & ~(1 << best_v)):
                        return True
                    color[best_v] = -1
            
            return False
        
        return not backtrack((1 << n) - 1)

    base_adj = make_adj_from_edges(core_edges)
    
    best_edges = []
    best_count = 0
    best_adj = None
    
    # All possible edges not in base_adj
    candidate_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (base_adj[u] >> v) & 1:
                candidate_edges.append((u, v))
    
    random.shuffle(candidate_edges)
    
    # Strategy 1: Greedy construction with random restarts
    num_restarts = 5000
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        current_adj = base_adj[:]
        
        # Randomly order candidate edges and greedily add if triangle-free
        shuffled = candidate_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if is_triangle_free_add(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
        
        # Close the graph
        closed_adj = close_graph(current_adj)
        
        # Count edges
        count = count_edges(closed_adj)
        
        if count > best_count:
            # Check if not 3-colorable
            if is_not_3_colorable(closed_adj):
                best_count = count
                best_adj = closed_adj[:]
                best_edges = edges_from_adj(closed_adj)
    
    # Strategy 2: Local search from best found so far
    if best_adj is not None and time.time() < deadline - 10:
        current_adj = best_adj[:]
        current_count = best_count
        
        # Try to add edges one by one
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (current_adj[u] >> v) & 1:
                        if is_triangle_free_add(current_adj, u, v):
                            # Try adding this edge
                            new_adj = current_adj[:]
                            new_adj[u] |= (1 << v)
                            new_adj[v] |= (1 << u)
                            # Close
                            closed = close_graph(new_adj)
                            new_count = count_edges(closed)
                            if new_count > current_count:
                                if is_not_3_colorable(closed):
                                    current_adj = closed
                                    current_count = new_count
                                    best_count = new_count
                                    best_adj = current_adj[:]
                                    best_edges = edges_from_adj(current_adj)
                                    improved = True
                                else:
                                    # Revert
                                    pass
    
    # Strategy 3: More aggressive random search with higher edge density
    if time.time() < deadline - 10:
        for i in range(3000):
            if time.time() > deadline - 5:
                break
            
            current_adj = base_adj[:]
            
            # Add edges with higher probability
            shuffled = candidate_edges[:]
            random.shuffle(shuffled)
            
            for u, v in shuffled:
                if random.random() < 0.7:
                    if is_triangle_free_add(current_adj, u, v):
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
            
            closed_adj = close_graph(current_adj)
            count = count_edges(closed_adj)
            
            if count > best_count:
                if is_not_3_colorable(closed_adj):
                    best_count = count
                    best_adj = closed_adj[:]
                    best_edges = edges_from_adj(closed_adj)
    
    if best_edges:
        return best_edges
    
    # Fallback: return the core graph
    return edges_from_adj(base_adj)
