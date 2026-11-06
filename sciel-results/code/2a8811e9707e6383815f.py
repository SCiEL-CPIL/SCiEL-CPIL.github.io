"""
C6-free graph construction on 40 vertices.
Strategy: Start with 8 disjoint K6s (120 edges, C6-free) and use randomized
edge-swapping local search to find a denser C6-free graph.
The initial 80-edge solution is a lower bound. The theoretical max is 105.
We start dense and try to "fix" C6s by swapping edges to maintain density while removing cycles.
Actually, K6 contains no C6? 
Wait. A K6 has 6 vertices. A C6 is a cycle of length 6. 
Does K6 contain a C6? Yes, any 6 vertices in a K6 form a C6 (the perimeter) and also chords.
The definition of "containing no six-cycle" usually means no induced C6 or no C6 as a subgraph?
The task says "containing no six-cycle (exact search)". 
Standard graph theory: "C6-free" means no subgraph isomorphic to C6.
In K6, pick vertices 0,1,2,3,4,5. Edges (0,1),(1,2),(2,3),(3,4),(4,5),(5,0) exist. 
So K6 CONTAINS a C6. 
Therefore, the seed "8 disjoint K5s" is correct because K5 has only 5 vertices, so it cannot contain a C6.
My previous thought that K6 is C6-free is WRONG.
A graph with max degree or small components might be safer.
K5 is safe. K6 is NOT safe.
So the seed 80 edges (8xK5) is the correct baseline.
The inspiration says "Randomized greedy edge addition... should find a denser C6-free graph... likely approaching 105".
The current program just returns the 80 edges.
I need to implement a search that adds edges to the 80-edge base.
Since I cannot start with K6, I must start with K5s and add edges between them or within?
Within K5 is already complete. So I must add edges BETWEEN the K5s.
This is a "greedy edge addition" strategy.
Let's implement a fast C6 check and greedy addition with random restarts.
"""

import time
import random
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill
    
    # Initial graph: 8 disjoint K5s
    # Vertices 0-4 in K5_0, 5-9 in K5_1, etc.
    adj = [set() for _ in range(n)]
    edges = []
    
    # Add K5s
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
    
    # Helper: Check if adding edge (u,v) creates a C6
    # A C6 is formed if there is a path of length 4 between u and v.
    # i.e., u - x - y - z - v
    # We need to check if there exist x,y,z such that:
    # x in N(u), z in N(v), y in N(x) intersect N(z)
    # And the path is simple (no repeated vertices).
    # Since we are adding an edge to a graph that is currently C6-free,
    # any new C6 must use the new edge (u,v).
    # So we look for a path u-x-y-z-v of length 4.
    
    def would_create_c6(u, v):
        # N(u) and N(v)
        nu = adj[u]
        nv = adj[v]
        
        # We need x in nu, z in nv, and a common neighbor y of x and z.
        # Also x != z, x != v, z != u (automatically true if graph is simple and u,v not adjacent yet)
        # Also x,y,z distinct.
        
        # Iterate over x in nu
        for x in nu:
            if x == v: continue
            # Iterate over z in nv
            for z in nv:
                if z == u: continue
                if z == x: continue
                # Check if x and z have a common neighbor y
                # y must be in N(x) and N(z)
                # Also y must not be u or v (to keep path simple? 
                # If y==u, path is u-x-u-z-v? No, x-u is edge, u-z? z in nv so z-v is edge. 
                # Path u-x-u... not simple. 
                # Simple path u-x-y-z-v requires distinct vertices.
                # So y != u, y != v, y != x, y != z.
                common = adj[x].intersection(adj[z])
                # Filter out u, v, x, z
                for y in common:
                    if y == u or y == v or y == x or y == z:
                        continue
                    # Found a path u-x-y-z-v
                    return True
        return False

    # Greedy addition with random restarts
    best_edges = list(edges)
    best_count = len(best_edges)
    
    # Pre-compute non-edges for faster iteration?
    # There are 780 total possible edges. 80 are present. 700 are absent.
    # We can just generate all non-edges.
    non_edges = []
    present_set = set()
    for e in edges:
        present_set.add((e[0], e[1]))
        present_set.add((e[1], e[0]))
    
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in present_set:
                non_edges.append([u, v])
    
    # Shuffle non-edges
    random.seed(42)
    
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Reset to base
        adj = [set() for _ in range(n)]
        edges = []
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
        
        # Shuffle non-edges
        current_non_edges = non_edges.copy()
        random.shuffle(current_non_edges)
        
        added_any = True
        while added_any and time.time() < deadline - 10:
            added_any = False
            # Try to add one edge
            # We iterate through shuffled list, but we need to remove successfully added ones
            # Actually, just iterate through a copy, and if we add one, we restart the scan?
            # Or just try all in random order once.
            
            # Efficiently: keep a list of candidates
            candidates = current_non_edges
            for i in range(len(candidates)):
                if time.time() > deadline - 10:
                    break
                u, v = candidates[i]
                if u in adj[v]: # Already added in this restart?
                    continue
                if not would_create_c6(u, v):
                    # Add edge
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
                    added_any = True
                    # Remove from candidates? 
                    # Since we are iterating a static list for this pass, 
                    # we can't easily remove without shifting indices.
                    # But adding an edge might make other edges invalid.
                    # A simple greedy pass is fine.
                    pass
            
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)
                
                # If we hit the known best, we can stop early? 
                # No, maybe we can go higher? Unlikely, but let's keep going if time permits.
                # Actually, 105 is the best known. If we reach 105, we can break.
                if best_count >= 105:
                    break
    
    return best_edges

# EVOLVE-BLOCK-END
