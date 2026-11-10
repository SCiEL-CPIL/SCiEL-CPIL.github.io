import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try n from 22 down to 6? Or start with 20?
    # Known: 20 is possible. 21 is unknown/hard. 22 is likely impossible.
    # Let's try 21 first, then 20.
    
    best_result = {'n': 5, 'edges': []}
    
    for n in [21, 20]:
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        edges = set()
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    edges.add((i, j))
        
        # Precompute combinations for each size >= 6
        # To save time, we only generate them if needed
        # But generating combinations is fast.
        
        def get_cost(edge_set):
            # Build adjacency list/bitmask
            adj = [0] * n
            for u, v in edge_set:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Check sizes from 6 to n
            # We can break early if we find violations?
            # We need the exact count for SA, but for hill climbing, 
            # just finding 1 violation is enough to know cost > 0.
            # However, for SA, magnitude matters.
            # Let's just count.
            
            total_violations = 0
            for k in range(6, n + 1):
                # Iterate combinations
                for comb in itertools.combinations(range(n), k):
                    mask = 0
                    for v in comb:
                        mask |= (1 << v)
                    
                    # Check if regular
                    # Get degrees for vertices in comb
                    degs = []
                    for v in comb:
                        # Count neighbors of v within comb
                        # adj[v] & mask
                        deg = bin(adj[v] & mask).count('1')
                        degs.append(deg)
                    
                    if len(set(degs)) == 1:
                        total_violations += 1
                        # Early exit if too many?
                        # If we just want to know if it's 0, we can return 1 immediately.
                        # But for optimization, we want to minimize.
                        # Let's return 1 if we find any, to speed up search towards 0.
                        # Actually, if we return 1 for any violation, the landscape is flat 
                        # for all invalid graphs. This might make search slow.
                        # But it's very fast.
                        # Let's try returning 1 immediately.
                        return 1
                        
            return 0

        # Local Search
        current_edges = set(edges)
        current_cost = get_cost(current_edges)
        
        if current_cost == 0:
            return {'n': n, 'edges': [list(e) for e in current_edges]}
            
        T = 1.0
        best_cost = current_cost
        best_edges = set(current_edges)
        
        while time.time() < deadline - 5:
            # Flip random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            new_edges = set(current_edges)
            if (u, v) in new_edges:
                new_edges.remove((u, v))
            else:
                new_edges.add((u, v))
                
            new_cost = get_cost(new_edges)
            
            if new_cost < current_cost or random.random() < pow(2.718, -(new_cost - current_cost) / T):
                current_edges = new_edges
                current_cost = new_cost
                if current_cost == 0:
                    return {'n': n, 'edges': [list(e) for e in current_edges]}
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_edges = set(current_edges)
            T *= 0.99
            
            if T < 0.01:
                break
        
        # If we found a best graph with cost 0, it would have returned.
        # If not, we didn't find a valid graph for this n.
        
    # If we couldn't find valid 21 or 20, try smaller?
    # Or just return n=5?
    # Let's try 19, 18... just in case.
    # But 20 is known. If we fail at 20, it's likely due to time/search.
    # We should return the best valid graph found.
    # Since we only return if cost==0, we haven't stored invalid ones.
    # We should store the best VALID graph.
    # Since we didn't find any, we fall back to n=5.
    
    return best_result
