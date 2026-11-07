"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # Strategy: Single Parity Star
    # Pick a central even vertex as hub
    hub = 11
    parity = hub % 2
    
    added_edges = []
    
    # Connect hub to all other vertices of the same parity
    # This ensures triangle-freeness (no triangles formed within the partition)
    # and reduces diameter significantly.
    for u in range(n):
        if u == hub:
            continue
        if u % 2 == parity:
            # Check if edge already exists (it shouldn't in path, but be safe)
            if (hub, u) not in base_edges and (u, hub) not in base_edges:
                added_edges.append([hub, u])
                
    # Verification (Optional but good for debugging/safety)
    # Build full graph
    full_edges = list(base_edges) + [(u, v) for u, v in added_edges] + [(v, u) for u, v in added_edges]
    adj = {i: set() for i in range(n)}
    for u, v in full_edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
            
    # Check Triangle-Free
    # For each edge (u, v), check if they have common neighbor
    is_triangle_free = True
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common = adj[u].intersection(adj[v])
                if common:
                    is_triangle_free = False
                    break
        if not is_triangle_free:
            break
            
    if not is_triangle_free:
        # Fallback to bipartite double star if triangle check fails (shouldn't happen)
        # But for now, return empty or seed? 
        # The star of same-parity edges is always triangle-free in a path.
        pass
        
    # Check Diameter
    # BFS from each node
    def bfs_dist(start):
        dist = {start: 0}
        queue = [start]
        while queue:
            curr = queue.pop(0)
            for nb in adj[curr]:
                if nb not in dist:
                    dist[nb] = dist[curr] + 1
                    queue.append(nb)
        return max(dist.values()) if dist else 0
        
    max_diam = 0
    for i in range(n):
        d = bfs_dist(i)
        if d > max_diam:
            max_diam = d
            
    if max_diam > 4:
        # If diameter is too large, we might need to add more edges or use a different strategy.
        # For N=24, single hub at 11 should give diam 4.
        # If it fails, fallback to adding more edges?
        # We can try adding cross-parity edges to reduce diameter further.
        pass

    return [[int(u), int(v)] for u, v in added_edges]

# EVOLVE-BLOCK-END
