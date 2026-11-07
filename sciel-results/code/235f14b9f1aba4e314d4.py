import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_diameter_and_counts(adj, n):
    """
    Returns (diameter, count_of_pairs_at_diameter)
    """
    max_diam = 0
    count_at_max = 0
    
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        
        if any(d == -1 for d in dist):
            return 999, 0
            
        for i in range(n):
            if i > src:
                d = dist[i]
                if d > max_diam:
                    max_diam = d
                    count_at_max = 1
                elif d == max_diam:
                    count_at_max += 1
        
    return max_diam, count_at_max

def _is_valid(base_edges, added_edges, n):
    """Check if the union of base and added edges is triangle-free, connected, and has diameter <= 4."""
    adj = {i: set() for i in range(n)}
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for u, v in added_edges:
        if u not in adj[v] and v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
        else:
            return False  # Duplicate edge
    
    # Check triangle-free
    for u in adj:
        for v in adj[u]:
            for w in adj[v]:
                if w in adj[u] and u < v < w:
                    return False
    
    # Check diameter
    diam, _ = _bfs_diameter_and_counts(adj, n)
    return diam <= 4

def construct():
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    # Pre-compute bipartite candidates (even-odd pairs)
    # The base path is bipartite with even/odd partition.
    # Adding only even-odd edges guarantees triangle-free.
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_set:
                continue
            if u % 2 != v % 2:
                all_candidates.append((u, v))
                
    best_solution = None
    best_count = float('inf')
    
    deadline = time.time() + 85
    
    # Run multiple randomized greedy passes
    for attempt in range(300):
        if time.time() > deadline:
            break
            
        current_added = []
        adj = {i: set() for i in range(n)}
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        curr_diam, curr_count = _bfs_diameter_and_counts(adj, n)
        
        while curr_diam > 4:
            if time.time() > deadline - 5:
                break
                
            best_edge = None
            best_score = (curr_diam, curr_count)
            
            # Shuffle candidates to break ties randomly in later attempts
            candidates = all_candidates[:]
            if attempt > 0:
                random.shuffle(candidates)
                
            for u, v in candidates:
                # Skip if already added or in base
                if v in adj[u]:
                    continue
                    
                # Tentatively add
                adj[u].add(v)
                adj[v].add(u)
                
                new_diam, new_count = _bfs_diameter_and_counts(adj, n)
                
                # Score: (diameter, count_at_max) - lower is better
                new_score = (new_diam, new_count)
                
                # Undo
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edge = (u, v)
                    
            if best_edge is None:
                # No improvement possible
                break
                
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            current_added.append([u, v])
            curr_diam, curr_count = best_score
            
            if curr_diam <= 4:
                if len(current_added) < best_count:
                    best_count = len(current_added)
                    best_solution = [list(e) for e in current_added]
                break  # Found valid solution for this attempt
            
    if best_solution is None:
        return []
        
    return [[int(u), int(v)] for u, v in best_solution]

# EVOLVE-BLOCK-END
