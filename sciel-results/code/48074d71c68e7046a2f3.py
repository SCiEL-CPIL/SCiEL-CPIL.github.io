import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_addition(adj, common, u, v):
        # Check if adding (u, v) causes any pair to have 3 common neighbors
        for w in adj[v]:
            if common[u][w] >= 3:
                return False
        for w in adj[u]:
            if common[v][w] >= 3:
                return False
        return True

    def try_greedy(order):
        adj = [set() for _ in range(n)]
        common = [[0] * n for _ in range(n)]
        edges = []
        
        for u, v in order:
            if is_valid_addition(adj, common, u, v):
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                
                # Update common counts
                # Optimization: only update pairs that are affected
                for w in adj[v]:
                    common[u][w] += 1
                    common[w][u] += 1
                for w in adj[u]:
                    common[v][w] += 1
                    common[w][v] += 1
        
        return edges

    # Run greedy with different random orderings
    # We can run many restarts since n is small
    while time.time() < deadline - 10.0:
        random.shuffle(all_edges)
        edges = try_greedy(all_edges)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
            # If we find a very high number, we might want to stop early?
            # No, let's keep searching.
            
    # Polish: try to add more edges to the best solution found
    if best_edges and time.time() < deadline - 2.0:
        adj = [set() for _ in range(n)]
        common = [[0] * n for _ in range(n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
            for w in adj[v]:
                common[u][w] += 1
                common[w][u] += 1
            for w in adj[u]:
                common[v][w] += 1
                common[w][v] += 1
        
        # Try adding missing edges
        existing = set(best_edges)
        missing = [e for e in all_edges if e not in existing]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 1.0:
                break
            if is_valid_addition(adj, common, u, v):
                adj[u].add(v)
                adj[v].add(u)
                best_edges.append((u, v))
                best_count += 1
                for w in adj[v]:
                    common[u][w] += 1
                    common[w][u] += 1
                for w in adj[u]:
                    common[v][w] += 1
                    common[w][v] += 1

    return [[u, v] for u, v in best_edges]
