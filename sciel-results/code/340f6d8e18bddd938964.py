import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # We will try to find a valid graph for n=21, then 20, etc.
    # The task says maximize n.
    
    def get_adj_from_edges(edges, n):
        adj = [[0]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def count_regular_6(adj, n):
        """Count subsets of size 6 that induce a regular graph."""
        count = 0
        # Precompute popcount for masks? No, use bit_count
        # Iterate over all 6-subsets
        for subset in combinations(range(n), 6):
            # Check if regular
            degs = []
            for u in subset:
                deg = 0
                for v in subset:
                    if u != v and adj[u][v]:
                        deg += 1
                degs.append(deg)
            if all(d == degs[0] for d in degs):
                count += 1
        return count

    def find_violation_ge_6(adj, n):
        """Find a subset of size >= 6 that induces a regular graph. Returns subset or None."""
        # Check size 6 first
        for subset in combinations(range(n), 6):
            degs = []
            for u in subset:
                deg = 0
                for v in subset:
                    if u != v and adj[u][v]:
                        deg += 1
                degs.append(deg)
            if all(d == degs[0] for d in degs):
                return subset
        # Check larger sizes
        for size in range(7, n + 1):
            for subset in combinations(range(n), size):
                degs = []
                for u in subset:
                    deg = 0
                    for v in subset:
                        if u != v and adj[u][v]:
                            deg += 1
                    degs.append(deg)
                if all(d == degs[0] for d in degs):
                    return subset
        return None

    def is_valid(adj, n):
        return find_violation_ge_6(adj, n) is None

    best_n = 5
    best_edges = []
    
    # Try n from 21 down to 10
    for target_n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 5:
            break
            
        found = False
        # Multiple restarts
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
                
            # Initialize random graph
            adj = [[0]*target_n for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < 0.5:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            # Simulated Annealing to minimize count_regular_6
            current_cost = count_regular_6(adj, target_n)
            
            # If already 0, check for larger violations
            if current_cost == 0:
                if is_valid(adj, target_n):
                    best_n = target_n
                    best_edges = edges_from_adj(adj, target_n)
                    found = True
                    break
                else:
                    # Has larger violation, try to fix it locally
                    # We'll do a local search to remove larger violations while keeping 6-cost low
                    pass
            
            T = 1.0
            # SA Loop
            # Limit steps per attempt to avoid hanging
            max_steps = 5000
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                
                # Pick random edge to flip
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                # Flip
                adj[u][v] = 1 - adj[u][v]
                adj[v][u] = 1 - adj[v][u]
                
                new_cost = count_regular_6(adj, target_n)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    current_cost = new_cost
                    # If cost is 0, check full validity
                    if current_cost == 0:
                        if is_valid(adj, target_n):
                            best_n = target_n
                            best_edges = edges_from_adj(adj, target_n)
                            found = True
                            break
                else:
                    # Revert
                    adj[u][v] = 1 - adj[u][v]
                    adj[v][u] = 1 - adj[v][u]
                
                T *= 0.999
                if T < 0.01:
                    break
            
            if found:
                break
        
        if found:
            continue
        else:
            # If we failed for this n, we might try smaller n
            # But we want to maximize n.
            # If we found a valid graph for n=20, we should keep it.
            # The loop breaks out of n loop if found?
            # No, if found, it continues to next n (smaller).
            # We should break if we found a valid graph for the LARGEST possible n?
            # The loop goes 21, 20, 19...
            # If we find 21, we break and return 21.
            # If we don't find 21, we try 20.
            # If we find 20, we break and return 20.
            # So we should break the n-loop if we find a valid graph.
            if best_n == target_n:
                break

    return {'n': best_n, 'edges': best_edges}
