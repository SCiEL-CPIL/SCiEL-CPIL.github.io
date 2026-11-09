import random
import itertools
import time

def construct():
    # We want to maximize n.
    # Known valid n=5.
    # Try to find n=20, 19, 18...
    
    def is_valid_graph(n, edges):
        # Build adjacency matrix or set
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check all subsets of size >= 6
        # To speed up, we can check smaller subsets first?
        # Actually, if a subset is regular, its supersets might not be.
        # But we must check ALL subsets of size >= 6.
        
        # Optimization: 
        # If n is small, brute force is fine.
        # For n=20, number of subsets is ~10^6.
        # Checking each subset takes O(k^2) or O(k).
        # Total time ~ 10^6 * 20 = 20M ops. Feasible in Python? Maybe 1-2 seconds.
        
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                # Check if induced subgraph is regular
                degrees = [0] * k
                for i, u in enumerate(subset):
                    for j, v in enumerate(subset):
                        if i != j and v in adj[u]:
                            degrees[i] += 1
                # Check if all degrees are same
                if len(set(degrees)) == 1:
                    return False
        return True

    def generate_random_graph(n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    edges.append([u, v])
        return edges

    def try_find(n, max_attempts=100):
        # Try random graphs
        for _ in range(max_attempts):
            edges = generate_random_graph(n)
            if is_valid_graph(n, edges):
                return edges
        
        # If random fails, try a specific structure or lower p
        # Let's try p=0.3
        for _ in range(max_attempts):
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.3:
                        edges.append([u, v])
            if is_valid_graph(n, edges):
                return edges
                
        return None

    # Try n=20
    start_time = time.time()
    edges = try_find(20, max_attempts=50)
    if edges is not None:
        return {'n': 20, 'edges': edges}
        
    if time.time() - start_time > 60:
        # Fallback to smaller n
        pass

    edges = try_find(19, max_attempts=50)
    if edges is not None:
        return {'n': 19, 'edges': edges}

    edges = try_find(18, max_attempts=50)
    if edges is not None:
        return {'n': 18, 'edges': edges}

    edges = try_find(15, max_attempts=100)
    if edges is not None:
        return {'n': 15, 'edges': edges}

    # Fallback to seed
    return {'n': 5, 'edges': []}
