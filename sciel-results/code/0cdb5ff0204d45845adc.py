"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    """
    Strategy: Exact Branch and Bound search for minimum edge augmentation.
    
    1. Base graph is a path P24.
    2. We add edges to reduce diameter to <= 4.
    3. Constraint: Resulting graph must be triangle-free.
    4. Objective: Minimize number of added edges.
    
    The search explores sets of added edges. At each step, it identifies the 
    pair of vertices with the maximum distance and attempts to add an edge 
    that reduces this distance. It prunes branches if the number of edges 
    already added exceeds the best known solution.
    """
    
    n = P['n']
    base_edges = set(tuple(sorted(e)) for e in P['edges'])
    
    # Precompute all potential candidate edges (not in base)
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                all_candidates.append((u, v))
    
    # Sort candidates by length (longer edges likely more useful for diameter reduction)
    # Actually, for diameter reduction, edges that connect distant parts are key.
    # We will generate candidates dynamically in the search based on the "bottleneck" pair.
    
    best_solution = []
    best_count = float('inf')
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # State: 
    # current_edges: set of added edges
    # adj: list of sets for adjacency
    
    # Initial adjacency
    init_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        init_adj[u].add(v)
        init_adj[v].add(u)
        
    # To speed up triangle check, we can precompute for each candidate (u,v) 
    # the list of vertices w that would form a triangle if (u,v) is added 
    # given the current graph. But current graph changes.
    # Instead, check on the fly.
    
    def check_triangle_free(adj, u, v):
        """Check if adding edge (u,v) to graph with adjacency 'adj' creates a triangle."""
        # A triangle is formed if there is a w such that w is neighbor of u AND w is neighbor of v.
        # Note: u and v are not currently connected (otherwise we wouldn't add it).
        common_neighbors = adj[u].intersection(adj[v])
        if common_neighbors:
            return False
        return True

    def get_diameter_and_worst_pair(adj):
        """Compute diameter and the pair achieving it. Returns (diam, u, v)."""
        max_dist = 0
        worst_u, worst_v = 0, 0
        for i in range(n):
            # BFS from i
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                curr = q.popleft()
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        q.append(nei)
            # Check max dist from i
            for j in range(i + 1, n):
                if dist[j] > max_dist:
                    max_dist = dist[j]
                    worst_u, worst_v = i, j
        return max_dist, worst_u, worst_v

    def search(current_edges, adj, depth):
        nonlocal best_solution, best_count
        
        if time.time() > deadline:
            return
            
        if len(current_edges) >= best_count:
            return
            
        diam, u, v = get_diameter_and_worst_pair(adj)
        
        if diam <= 4:
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_solution = list(current_edges)
            return
            
        # If we can't improve, prune
        # Simple lower bound: 1 more edge needed
        if len(current_edges) + 1 >= best_count:
            return
            
        # Identify candidates that reduce the distance between u and v
        # Or more generally, candidates that reduce the diameter.
        # To keep search space small, we only consider edges (a,b) such that 
        # adding them reduces dist(u,v) in the current graph.
        
        # Compute dist from u and v in current graph
        def bfs_dist(src):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                curr = q.popleft()
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        q.append(nei)
            return dist
            
        dist_u = bfs_dist(u)
        dist_v = bfs_dist(v)
        
        candidates = []
        for (a, b) in all_candidates:
            if (a, b) in current_edges or (b, a) in current_edges:
                continue
            if (a, b) in base_edges:
                continue
                
            # Check triangle constraint
            if not check_triangle_free(adj, a, b):
                continue
                
            # Check if it reduces distance between u and v
            # New dist via a-b: dist(u,a) + 1 + dist(b,v) or dist(u,b) + 1 + dist(a,v)
            new_dist = dist_u[v] # current
            if dist_u[a] != -1 and dist_v[b] != -1:
                new_dist = min(new_dist, dist_u[a] + 1 + dist_v[b])
            if dist_u[b] != -1 and dist_v[a] != -1:
                new_dist = min(new_dist, dist_u[b] + 1 + dist_v[a])
                
            if new_dist < dist_u[v]:
                candidates.append((a, b, new_dist))
                
        # Sort candidates by how much they reduce the distance (most reduction first)
        candidates.sort(key=lambda x: x[2])
        
        # Limit branching factor to avoid explosion, but ensure we explore promising ones
        # If best_count is small, we can be more aggressive.
        for a, b, new_dist in candidates:
            if time.time() > deadline:
                return
                
            # Add edge (a,b)
            adj[a].add(b)
            adj[b].add(a)
            current_edges.add((a, b))
            
            search(current_edges, adj, depth + 1)
            
            # Backtrack
            adj[a].remove(b)
            adj[b].remove(a)
            current_edges.remove((a, b))
            
            if time.time() > deadline:
                return

    # Start search
    search(set(), init_adj, 0)
    
    # If no solution found within time or search space, fallback to known good solution
    if not best_solution:
        # Fallback: Hub 11 strategy (10 edges)
        added = []
        hub = 11
        for v in range(0, 24, 2):
            if v == 10 or v == 12:
                continue
            if v == hub:
                continue
            added.append([hub, v])
        best_solution = added
    else:
        best_solution = [[int(e[0]), int(e[1])] for e in best_solution]
        
    return best_solution
# EVOLVE-BLOCK-END
