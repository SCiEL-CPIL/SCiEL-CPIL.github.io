import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    n = 20
    # Initialize random graph
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i][j] = 1
                adj[j][i] = 1

    def is_regular(subset):
        k = len(subset)
        if k < 6:
            return False
        degrees = []
        for u in subset:
            deg = 0
            for v in subset:
                if u != v and adj[u][v]:
                    deg += 1
            degrees.append(deg)
        return all(d == degrees[0] for d in degrees)

    def find_violation():
        # Check sizes 6 to n
        # To save time, check size 6 first, then 7, etc.
        # But checking all is safer.
        # Optimization: If a subset of size k is regular, 
        # we just need to find one.
        
        # We can iterate over all masks? 2^20 is 1M.
        # For each mask, check if popcount >= 6 and if regular.
        
        # Better: iterate by size.
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                if is_regular(subset):
                    return subset
        return None

    best_graph = {'n': n, 'edges': []}
    
    def get_edges():
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    # Initial check
    violation = find_violation()
    
    while violation is not None:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random edge from the violating subset to flip
        # Or pick an edge that appears in many violations? 
        # For simplicity, pick a random edge within the subset or incident to it.
        # Flipping an edge inside the subset is most likely to break regularity.
        u, v = random.sample(violation, 2)
        adj[u][v] ^= 1
        adj[v][u] ^= 1
        
        violation = find_violation()
        
        # If we are stuck or taking too long, try a random restart or accept if no violations?
        # The loop continues until no violation is found.
        
    # Final verification
    if find_violation() is None:
        return {'n': n, 'edges': get_edges()}
    else:
        # Fallback to a known valid small graph if we failed
        # But n=20 should be reachable.
        # Return the current state if it happens to be valid? No, it's not.
        # We should ensure we return a valid graph.
        # If we fail, return n=5 empty graph to be safe? 
        # But the task says "Maximize n". 
        # Let's try to return the best valid graph found.
        # Since we only update adj when we break a violation, and we stop when no violation,
        # if we exit due to time, we might have a violation.
        # We should probably just return the graph if it's valid, else a smaller valid one.
        
        # Let's check if the current graph is valid.
        if find_violation() is None:
            return {'n': n, 'edges': get_edges()}
        
        # If not, we have to return something valid.
        # A random graph on n=5 is likely valid? No, empty graph is valid.
        # Let's return the empty graph on 5 vertices as a last resort.
        return {'n': 5, 'edges': []}
