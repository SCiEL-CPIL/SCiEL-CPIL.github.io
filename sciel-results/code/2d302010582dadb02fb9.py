import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n = 20
    best_edges = 0
    best_graph = []
    
    # Precompute all triples for faster iteration?
    # There are C(20, 3) = 1140 triples.
    triples = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triples.append((u, v, w))
                
    def find_violation(adj):
        # adj is list of sets
        for u, v, w in triples:
            common = adj[u] & adj[v] & adj[w]
            if len(common) >= 3:
                return u, v, w, common
        return None

    def is_valid(adj):
        return find_violation(adj) is None

    def count_edges(adj):
        return sum(len(adj[u]) for u in range(n)) // 2

    while time.time() < deadline:
        # Initialize with complete graph
        adj = [set(range(n)) for _ in range(n)]
        for u in range(n):
            adj[u].remove(u)
        
        # Greedy removal
        while True:
            if time.time() > deadline:
                break
            
            res = find_violation(adj)
            if res is None:
                break
            
            u, v, w, common = res
            
            # Pick a random neighbor from common to remove
            x = random.choice(list(common))
            
            # Remove edge (u, x) or (v, x) or (w, x)
            # Random choice
            target = random.choice([u, v, w])
            
            adj[target].remove(x)
            adj[x].remove(target)
            
        if is_valid(adj):
            edges = count_edges(adj)
            if edges > best_edges:
                best_edges = edges
                best_graph = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            best_graph.append([u, v])
                
                if best_edges >= 74:
                    return best_graph

    return best_graph
