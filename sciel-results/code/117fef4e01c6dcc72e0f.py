"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_diameter(n, adj):
    max_d = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return -1
        max_d = max(max_d, max(dist))
    return max_d

def construct():
    n = P['n']
    base_edges = P['edges']
    base_set = {tuple(e) for e in base_edges}
    
    # Candidates: Must be different parity to ensure triangle-free (bipartite)
    # And not in base edges.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_set:
                continue
            if u % 2 != v % 2:
                candidates.append((u, v))
                
    # Sort candidates by "usefulness" - maybe longer jumps first?
    # Or just iterate.
    
    added_edges = []
    current_edges = list(base_edges)
    
    # Greedy loop
    # We want diameter <= 4.
    # Check current diameter.
    
    # To speed up, we can just try to add edges that connect distant parts.
    # A simple heuristic: 
    # 1. Calculate diameter.
    # 2. If > 4, find a pair (u,v) with max dist.
    # 3. Find a candidate edge (a,b) that minimizes dist(u,v) in the new graph.
    # 4. Add it.
    
    # Optimization: Pre-calculate adjacency updates? No, N=24 is small.
    
    deadline = time.time() + 50
    
    while time.time() < deadline:
        adj = _build_adj(n, current_edges)
        diam = _bfs_diameter(n, adj)
        
        if diam <= 4:
            break
            
        if diam == -1:
            # Should not happen as we start with a path
            break
            
        # Find the pair with maximum distance
        # We need to find u, v such that dist(u,v) > 4.
        # Actually, if diam > 4, there exists such a pair.
        # Let's find one.
        worst_u, worst_v = -1, -1
        max_dist_found = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                        if dist[v] > max_dist_found:
                            max_dist_found = dist[v]
                            worst_u = s
                            worst_v = v
                            
        if worst_u == -1:
            break
            
        # We have a pair (worst_u, worst_v) with distance > 4.
        # We want to add an edge (a,b) from candidates that reduces this distance.
        # Ideally, we add an edge that creates a shortcut.
        
        best_candidate = None
        best_new_dist = max_dist_found + 1 # Infinity
        
        # Try all candidates? There are ~200. 200 * BFS(24) is fast.
        for a, b in candidates:
            if (a, b) in {tuple(e) for e in added_edges}:
                continue
            # Check if adding (a,b) reduces dist(worst_u, worst_v)
            # We can simulate adding (a,b) to adj and running BFS from worst_u to worst_v.
            # To be faster, we can just check if a or b is "in the middle" or helps.
            # But BFS is safe.
            
            # Create a temporary adj? Or just modify and revert?
            # Modifying sets is fast.
            adj[worst_u].add(worst_v) # Dummy? No, just add a,b
            adj[a].add(b)
            adj[b].add(a)
            
            # BFS from worst_u to worst_v
            dist = [-1] * n
            dist[worst_u] = 0
            queue = [worst_u]
            qi = 0
            found = -1
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if u == worst_v:
                    found = dist[u]
                    break
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                        
            # Revert
            adj[a].remove(b)
            adj[b].remove(a)
            
            if found != -1 and found < best_new_dist:
                best_new_dist = found
                best_candidate = (a, b)
                
        if best_candidate is None:
            # No single edge improves this specific pair? 
            # Maybe we need to add multiple? 
            # Or maybe we picked a bad pair?
            # If no edge helps the worst pair, we might be stuck or need a different strategy.
            # But usually, adding an edge between the "halves" helps.
            # Let's just pick the candidate that minimizes the overall diameter estimate?
            # For now, if no improvement, break to avoid infinite loop.
            break
            
        added_edges.append(best_candidate)
        current_edges.append(best_candidate)
        
    # Final check
    if _bfs_diameter(n, _build_adj(n, current_edges)) <= 4:
        return [[u, v] for u, v in added_edges]
        
    # Fallback: If greedy failed, try a known sparse structure.
    # Connect 0-6, 6-12, 12-18, 18-23? 
    # Let's try a fixed set of "long" bipartite edges.
    fallback = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23), (0, 10), (10, 20)]
    # Verify fallback
    fb_edges = base_edges + fallback
    fb_adj = _build_adj(n, fb_edges)
    # Check triangle free (bipartite check)
    is_bipartite = True
    colors = [-1] * n
    for i in range(n):
        if colors[i] == -1:
            colors[i] = 0
            q = [i]
            while q:
                u = q.pop(0)
                for v in fb_adj[u]:
                    if colors[v] == -1:
                        colors[v] = 1 - colors[u]
                        q.append(v)
                    elif colors[v] == colors[u]:
                        is_bipartite = False
    if is_bipartite and _bfs_diameter(n, fb_adj) <= 4:
        return [[u, v] for u, v in fallback]
        
    # Last resort: return empty list (invalid) or best effort
    return [[u, v] for u, v in added_edges]

# EVOLVE-BLOCK-END
