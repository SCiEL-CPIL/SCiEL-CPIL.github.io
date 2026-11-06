"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    
    # Try to build C4-free graphs for n=47, 46, 45, ...
    # For n vertices, we need min_degree >= n - 39
    # So for n=47: min_deg >= 8, for n=46: min_deg >= 7, etc.
    
    best_result = {'n': 39, 'edges': []}
    
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
        
        min_deg_needed = n - 39
        if min_deg_needed <= 0:
            # Trivial: empty graph works
            best_result = {'n': n, 'edges': []}
            continue
        
        # Greedy edge addition with C4 check
        # Start with a random graph and try to add edges while maintaining C4-free
        # Strategy: start with high-degree structure, add edges greedily
        
        # Use a randomized greedy approach:
        # 1. Start with an empty graph
        # 2. Repeatedly try to add the edge that gives the most "room" 
        #    (i.e., creates no C4) while trying to balance degrees
        
        # Better approach: Start from a known C4-free structure
        # For high min-degree C4-free graphs, polar graphs and incidence structures work well.
        
        # Let's try a greedy approach:
        # - Maintain adjacency sets
        # - For each pair (u,v) not yet connected, check if adding edge creates C4
        # - An edge (u,v) creates a C4 if there exists w such that u-w and v-w are both edges
        #   (i.e., u and v have a common neighbor)
        # - So we can only add edge (u,v) if they have NO common neighbors
        
        # This means the graph must be C4-free, which is equivalent to:
        # for any two vertices, they share at most one common neighbor (actually zero for the edge addition)
        # Wait, no. A C4 is u-v-w-x-u. Adding edge (u,v) creates a C4 if there's a path u-w-v of length 2.
        # So u and v must have no common neighbor.
        
        # This is a strong constraint. Let's think about it differently.
        # In a C4-free graph, any two vertices have at most 1 common neighbor.
        # When adding edge (u,v), we need that u and v currently have 0 common neighbors.
        
        # Let's use a greedy heuristic:
        # Priority: add edges between vertices with lower degrees first (to balance)
        
        success = False
        for attempt in range(50):  # multiple random restarts
            if time.time() > deadline:
                break
            
            random.seed(attempt * 12345 + n)
            
            # Build graph greedily
            adj = [set() for _ in range(n)]
            edges = []
            
            # Precompute all possible edges
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            # Greedy: try to add edges, prioritizing those that help balance degrees
            # We'll do multiple passes
            for pass_num in range(3):
                # Sort edges by (deg[u] + deg[v]) ascending to balance
                all_edges.sort(key=lambda e: len(adj[e[0]]) + len(adj[e[1]]))
                
                added_this_pass = 0
                for (u, v) in all_edges:
                    if time.time() > deadline:
                        break
                    if v in adj[u]:
                        continue
                    
                    # Check if adding (u,v) creates a C4:
                    # This happens if u and v have a common neighbor
                    common = adj[u] & adj[v]
                    if len(common) == 0:
                        # Safe to add
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.append([u, v])
                        added_this_pass += 1
                if added_this_pass == 0:
                    break
            
            # Check if minimum degree condition is met
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= min_deg_needed:
                success = True
                best_result = {'n': n, 'edges': edges}
                break
        
        if success:
            break
    
    # Verify the result
    if best_result['n'] > 39:
        n = best_result['n']
        edges = best_result['edges']
        # Quick verification: check no C4 and min degree
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check min degree
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg < n - 39:
            best_result = {'n': 39, 'edges': []}
        
        # Check C4: for each pair of edges sharing a vertex, check for C4
        # Actually, let's verify more carefully
        # A C4 exists if there are vertices a,b,c,d with edges ab, bc, cd, da
        # Equivalently, for any two vertices u,v, if they have 2+ common neighbors, 
        # and those two neighbors are connected... wait no.
        # C4: u-w-v-x-u means u and v share neighbors w and x, AND w and x are not directly relevant
        # Actually C4 is a cycle of length 4: v1-v2-v3-v4-v1
        # So we need edges (v1,v2), (v2,v3), (v3,v4), (v4,v1)
        # This means v1 and v3 share two common neighbors: v2 and v4
        # So: C4 exists iff there exist two vertices with >= 2 common neighbors
        
        has_c4 = False
        for u in range(n):
            for v in range(u+1, n):
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    has_c4 = True
                    break
            if has_c4:
                break
        
        if has_c4:
            best_result = {'n': 39, 'edges': []}
    
    return best_result
# EVOLVE-BLOCK-END
