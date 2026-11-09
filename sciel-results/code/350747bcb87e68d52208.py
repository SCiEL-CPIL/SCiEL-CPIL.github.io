import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def is_regular_subset(subset, adj):
        # subset is a tuple of vertex indices
        # adj is a list of sets or bitmasks
        n_sub = len(subset)
        if n_sub < 6:
            return False
        
        # Check if all vertices in subset have same degree within subset
        degs = []
        for v in subset:
            deg = 0
            for u in subset:
                if u != v and adj[v][u]:
                    deg += 1
            degs.append(deg)
        
        return len(set(degs)) == 1

    def count_regular_6_sets(adj, n):
        count = 0
        # Optimize by iterating combinations
        # For n=21, C(21,6) = 54264. This is fast enough for a few checks.
        for comb in itertools.combinations(range(n), 6):
            if is_regular_subset(comb, adj):
                count += 1
        return count

    def try_find_graph(n, max_time=100):
        # Initialize with random graph
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Local Search
        current_violations = count_regular_6_sets(adj, n)
        best_violations = current_violations
        best_adj = [row[:] for row in adj]
        
        t = 1.0
        t_min = 0.01
        cooling = 0.995
        
        while time.time() < max_time + start_time:
            # Pick a random edge to flip
            i, j = random.sample(range(n), 2)
            if i == j: continue
            
            # Flip edge
            old_val = adj[i][j]
            adj[i][j] = not old_val
            adj[j][i] = not old_val
            
            new_violations = count_regular_6_sets(adj, n)
            
            # Check larger subsets if 6-set violations are 0
            if new_violations == 0:
                # Quick check for k>=7
                valid = True
                for k in range(7, n + 1):
                    # Sample a few subsets or check all if n is small
                    # For n=21, C(21,7) is ~116k. Too many to check every step.
                    # We will only do a full check if 6-violations are 0.
                    # To save time, we assume if 6-violations are 0, we need to be careful.
                    # But for the search, we primarily drive by 6-violations.
                    pass 
                # If we want to be strict, we should check. 
                # But checking all k>=7 is expensive.
                # Let's just return if 6-violations is 0 and hope for the best, 
                # or do a periodic full check.
                
            delta = new_violations - current_violations
            
            if delta < 0 or (t > t_min and random.random() < np.exp(-delta / t)):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_adj = [row[:] for row in adj]
                    if best_violations == 0:
                        # Found a candidate! Verify fully
                        if fully_valid(best_adj, n):
                            return best_adj
            else:
                # Reject flip
                adj[i][j] = old_val
                adj[j][i] = old_val
                
            t *= cooling
            
        return best_adj if best_violations == 0 else None

    def fully_valid(adj, n):
        # Check all subsets of size >= 6
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                if is_regular_subset(comb, adj):
                    return False
        return True

    # Try n=21
    # Allocate most time to n=21
    time_left = 110 - (time.time() - start_time)
    if time_left > 5:
        adj_21 = try_find_graph(21, max_time=time_left - 5)
        if adj_21 is not None:
            # Double check
            if fully_valid(adj_21, 21):
                edges = []
                for i in range(21):
                    for j in range(i + 1, 21):
                        if adj_21[i][j]:
                            edges.append([i, j])
                return {'n': 21, 'edges': edges}
    
    # Try n=20
    time_left = 110 - (time.time() - start_time)
    if time_left > 5:
        adj_20 = try_find_graph(20, max_time=time_left)
        if adj_20 is not None:
            if fully_valid(adj_20, 20):
                edges = []
                for i in range(20):
                    for j in range(i + 1, 20):
                        if adj_20[i][j]:
                            edges.append([i, j])
                return {'n': 20, 'edges': edges}
        
        # If search didn't find 0 violations, try a known construction or random retry
        # For n=20, it's known to exist.
        # Let's do a quick random search for n=20 if the above failed
        for _ in range(10):
            if time.time() > deadline: break
            adj_20 = try_find_graph(20, max_time=5)
            if adj_20 is not None:
                if fully_valid(adj_20, 20):
                    edges = []
                    for i in range(20):
                        for j in range(i + 1, 20):
                            if adj_20[i][j]:
                                edges.append([i, j])
                    return {'n': 20, 'edges': edges}
    
    # Fallback: Return a small valid graph
    # The empty graph on 5 vertices is always valid.
    # Or we can return a known valid graph for n=20 if we have one hardcoded?
    # No, we should generate it.
    # If we fail to find a 20-vertex graph in time, return n=5.
    return {'n': 5, 'edges': []}
