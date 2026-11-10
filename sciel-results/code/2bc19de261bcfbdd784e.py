"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with 8 disjoint K5's (104 edges)
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([5 * k + a, 5 * k + b])
    
    n = 40
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    def has_c6(adj, edge_set, new_edge):
        """Check if adding new_edge creates a C6 in the graph."""
        u, v = new_edge
        # A C6 containing edge (u,v) would be u-v-w-x-y-z-u
        # This means there must be a path of length 4 from u to v not using edge (u,v)
        # Or equivalently, the graph with edge (u,v) removed must have a path of length 4 between u and v
        
        # We need to check if there exist vertices w, x, y, z such that:
        # u-w, w-x, x-y, y-z, z-v are all edges, and {u,v,w,x,y,z} are distinct
        # (with w != v, z != u, etc.)
        
        # Path: u - w - x - y - z - v
        # Edges needed: (u,w), (w,x), (x,y), (y,z), (z,v)
        # All vertices distinct: u, v, w, x, y, z all different
        
        # For each possible w in adj[u]:
        for w in adj[u]:
            if w == v:
                continue
            # For each possible x in adj[w]:
            for x in adj[w]:
                if x in (u, v, w):
                    continue
                # For each possible y in adj[x]:
                for y in adj[x]:
                    if y in (u, v, w, x):
                        continue
                    # For each possible z in adj[y]:
                    for z in adj[y]:
                        if z in (u, v, w, x, y):
                            continue
                        # Check if z is adjacent to v
                        if v in adj[z]:
                            return True
        return False
    
    def build_adjacency(edge_set):
        adj = [set() for _ in range(n)]
        for (a, b) in edge_set:
            adj[a].add(b)
            adj[b].add(a)
        return adj
    
    adj = build_adjacency(edge_set)
    
    # Try to add edges greedily
    all_possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    random.shuffle(all_possible)
    
    best_edges = list(edge_set)
    
    for (u, v) in all_possible:
        if time.time() > deadline:
            break
        if (u, v) in edge_set:
            continue
        # Check if adding this edge creates a C6
        if not has_c6(adj, edge_set, (u, v)):
            edge_set.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            best_edges = list(edge_set)
    
    # Convert back to list of [u, v]
    return [[a, b] for (a, b) in sorted(best_edges)]
# EVOLVE-BLOCK-END
