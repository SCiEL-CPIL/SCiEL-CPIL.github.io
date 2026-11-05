"""Construct a K3,3-free graph on 20 vertices maximizing edges."""
import time
import random
import itertools

def construct():
    N = 20
    deadline = time.time() + 110  # safety margin for 120s limit
    
    def init_graph():
        return [set() for _ in range(N)]
    
    def add_edge(neighbors, u, v):
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    def remove_edge(neighbors, u, v):
        neighbors[u].discard(v)
        neighbors[v].discard(u)
    
    def would_create_k33(neighbors, u, v):
        """Check if adding edge (u,v) creates any K3,3."""
        neighbors[u].add(v)
        neighbors[v].add(u)
        
        result = False
        for a, b, c in itertools.combinations(range(N), 3):
            if u in (a, b, c) or v in (a, b, c):
                common = neighbors[a] & neighbors[b] & neighbors[c]
                if len(common) >= 3:
                    result = True
                    break
        
        neighbors[u].discard(v)
        neighbors[v].discard(u)
        return result
    
    def edge_count(neighbors):
        return sum(len(s) for s in neighbors) // 2
    
    # Greedy construction with random restarts
    all_edges = list(itertools.combinations(range(N), 2))
    
    best_edges = 0
    best_neighbors = None
    
    num_restarts = 5
    edges_per_restart = (len(all_edges) + num_restarts - 1) // num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        neighbors = init_graph()
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time.time() > deadline - 20:
                break
            if u not in neighbors[v]:
                if not would_create_k33(neighbors, u, v):
                    add_edge(neighbors, u, v)
        
        ec = edge_count(neighbors)
        if ec > best_edges:
            best_edges = ec
            best_neighbors = [set(s) for s in neighbors]
        
        # Local search from this solution
        if time.time() > deadline - 15:
            break
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try adding any missing edge
            for u, v in all_edges:
                if time.time() > deadline - 10:
                    break
                if u not in neighbors[v]:
                    if not would_create_k33(neighbors, u, v):
                        add_edge(neighbors, u, v)
                        improved = True
                        ec = edge_count(neighbors)
                        if ec > best_edges:
                            best_edges = ec
                            best_neighbors = [set(s) for s in neighbors]
            
            if time.time() > deadline - 8:
                break
            
            # Try edge swaps: remove one edge, add another
            current_edges = [(u, v) for u in range(N) for v in neighbors[u] if u < v]
            random.shuffle(current_edges)
            
            for a, b in current_edges:
                if time.time() > deadline - 8:
                    break
                # Try swapping with a random missing edge
                for _ in range(5):  # try a few candidates
                    if time.time() > deadline - 8:
                        break
                    c, d = random.choice(all_edges)
                    if c in neighbors[d]:
                        continue
                    if (c, d) == (a, b) or (c, d) == (b, a):
                        continue
                    
                    # Perform swap
                    remove_edge(neighbors, a, b)
                    if not would_create_k33(neighbors, c, d):
                        add_edge(neighbors, c, d)
                        improved = True
                        ec = edge_count(neighbors)
                        if ec > best_edges:
                            best_edges = ec
                            best_neighbors = [set(s) for s in neighbors]
                    else:
                        add_edge(neighbors, a, b)
    
    if best_neighbors is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, N)]
    
    # Convert best to list of edges
    edges = []
    for u in range(N):
        for v in best_neighbors[u]:
            if u < v:
                edges.append([u, v])
    
    return edges

