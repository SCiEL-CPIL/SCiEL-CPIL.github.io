import random
import time

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    best_edges = None
    best_score = float('inf')
    
    def get_books(edges_set):
        # adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            
        max_book = 0
        for u, v in edges_set:
            # common neighbors
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        return max_book

    def random_graph():
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                edges.add((min(u,v), max(u,v)))
        return list(edges)

    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        edges = random_graph()
        edges_set = set(edges)
        
        current_score = get_books(edges_set)
        
        # Local search
        T = 1.0
        no_improve = 0
        while time.time() < deadline and no_improve < 1000:
            # Find max book edge
            # To speed up, we can just recompute max
            # But we need the edge itself to pick a common neighbor
            adj = {i: set() for i in range(n)}
            for u, v in edges_set:
                adj[u].add(v)
                adj[v].add(u)
                
            max_book = 0
            max_edge = None
            for u, v in edges_set:
                common = len(adj[u] & adj[v])
                if common > max_book:
                    max_book = common
                    max_edge = (u, v)
                    
            if max_book <= 7:
                # Try to go lower? The bound is ~6.67 so 7 is likely min.
                # Let's assume 7 is the target.
                break
                
            # Pick a common neighbor w of max_edge
            u, v = max_edge
            common_neighbors = list(adj[u] & adj[v])
            if not common_neighbors:
                break
            w = random.choice(common_neighbors)
            
            # Decide which edge to remove: (u,w) or (v,w)
            # Both are in the graph because w is a common neighbor
            if random.random() < 0.5:
                remove_edge = (min(u, w), max(u, w))
            else:
                remove_edge = (min(v, w), max(v, w))
                
            # Add a random non-edge
            added = False
            attempts = 0
            while not added and attempts < 100:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y:
                    new_edge = (min(x, y), max(x, y))
                    if new_edge not in edges_set:
                        edges_set.remove(remove_edge)
                        edges_set.add(new_edge)
                        added = True
                        break
                attempts += 1
                
            if not added:
                continue
                
            new_score = get_books(edges_set)
            
            if new_score <= current_score:
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(edges_set)
            else:
                # Simulated annealing acceptance
                prob = pow(2.718, -(new_score - current_score) / T)
                if random.random() < prob:
                    current_score = new_score
                    no_improve += 1
                else:
                    # Revert
                    edges_set.remove(new_edge)
                    edges_set.add(remove_edge)
                    no_improve += 1
                    
            T *= 0.99 # Cool down slightly? Or keep constant?
            # For discrete max, constant T might be better to escape local minima
            # Let's keep T constant at 1.0 for now
            
        # End local search for this restart
        
    # Ensure we return exactly 401 distinct edges
    # If best_edges is None, return seed
    if best_edges is None:
         return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return best_edges
