import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u, v), max(u, v)))
    
    # Precompute all possible non-base edges
    all_nonbase = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                all_nonbase.append((u, v))
    
    # Adjacency list for the base graph
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)

    def make_adj(added):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_triangle_free(adj, u, v):
        # Check if adding (u,v) creates a triangle
        # We need to check if there is a w such that (u,w) and (w,v) are in adj
        # Since we are checking before adding, we check common neighbors
        if len(adj[u]) < len(adj[v]):
            for w in adj[u]:
                if w != v and w in adj[v]:
                    return False
        else:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
        return True

    def diameter_le_4(adj):
        # BFS from each vertex to depth 4
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                node = q.popleft()
                if dist[node] == 4:
                    continue
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        q.append(nb)
            for d in dist:
                if d == -1:
                    return False
        return True

    def full_diameter(adj):
        maxd = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        q.append(nb)
            for d in dist:
                if d > maxd:
                    maxd = d
        return maxd

    def greedy_construction(seed):
        random.seed(seed)
        adj = make_adj([])
        added = []
        added_set = set()
        
        # We want to reduce diameter to 4
        # Greedily add edges that reduce the diameter the most
        while True:
            if time.time() > deadline:
                break
            if diameter_le_4(adj):
                break
            
            current_diam = full_diameter(adj)
            best_edge = None
            best_new_diam = current_diam + 1
            
            # Shuffle candidates to break ties randomly
            candidates = [e for e in all_nonbase if e not in added_set]
            random.shuffle(candidates)
            
            for u, v in candidates:
                if not is_triangle_free(adj, u, v):
                    continue
                
                # Tentatively add
                adj[u].add(v)
                adj[v].add(u)
                
                new_diam = full_diameter(adj)
                
                # Remove
                adj[u].discard(v)
                adj[v].discard(u)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
                    # Early stop if we reached diameter 4? 
                    # It's possible to reach 4, but maybe we can do it with fewer edges later?
                    # No, greedy adds one by one. If we reach 4, we stop.
                    # But maybe a different edge leads to a solution with fewer total edges?
                    # Greedy is not optimal. But it gives a starting point.
                    if new_diam <= 4:
                         # We found an edge that achieves the goal. 
                         # Is it the best? Maybe not. But we stop here to save time.
                         # Actually, we should just take it and stop.
                         break
            
            if best_edge is None:
                break
            
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            added_set.add((u, v))
            added.append((u, v))
            
        return added

    def local_search_minimize(added, time_limit):
        added_set = set(added)
        adj = make_adj(added)
        best_added = list(added_set)
        best_len = len(added_set)
        
        # 1. Try to remove edges
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # Randomize order of removal attempts
            edges_to_try = list(added_set)
            random.shuffle(edges_to_try)
            for e in edges_to_try:
                u, v = e
                # Try removing
                adj[u].discard(v)
                adj[v].discard(u)
                
                if diameter_le_4(adj):
                    added_set.discard(e)
                    best_added = list(added_set)
                    best_len = len(added_set)
                    changed = True
                    # Restart loop as set changed
                    break
                else:
                    # Restore
                    adj[u].add(v)
                    adj[v].add(u)
        
        # 2. If stuck, try swaps to escape local minima
        # We try to replace one edge with another, hoping to allow further removals
        # Or simply accept a different configuration of same size, then try removal again.
        # Given time constraints, we might just return the best found so far.
        # However, a simple swap heuristic:
        # Remove e1, add e2 (valid), then try to remove e2 or others.
        
        return best_added

    # Run multiple greedy constructions with different seeds
    best_solution = None
    best_size = float('inf')
    
    # First, try to get a feasible solution quickly
    # The current program's approach is a good baseline.
    
    for seed in range(50):
        if time.time() > deadline:
            break
        
        added = greedy_construction(seed)
        
        if not added:
            continue
            
        # Verify feasibility
        adj = make_adj(added)
        if not diameter_le_4(adj):
            continue
            
        # Local search to minimize
        improved = local_search_minimize(added, deadline)
        
        if len(improved) < best_size:
            best_size = len(improved)
            best_solution = improved
            
            # If we find a very small solution, we can stop early?
            # Theoretical lower bound is likely around 6-8.
            if best_size <= 6:
                break

    if best_solution is None:
        # Fallback: return empty list if something went wrong, though it shouldn't
        # Or return the greedy result from seed 0
        added = greedy_construction(0)
        return [[int(u), int(v)] for u, v in added]

    return [[int(u), int(v)] for u, v in best_solution]
