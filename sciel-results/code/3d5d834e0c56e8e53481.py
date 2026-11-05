"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import deque

def _has_c5(adj, n):
    """Return True if graph with adjacency matrix adj (list of lists of bools) on n vertices has a 5-cycle."""
    # A 5-cycle is a cycle of length 5. We can check by looking for paths of length 4 where
    # the endpoints are adjacent and no intermediate vertex repeats.
    # More efficiently: check all pairs (u, v) with u < v, and see if there's a path of length 4
    # from u to v that doesn't use the edge (u,v) directly... 
    # Actually, let's use a different approach: for each pair of non-adjacent vertices, check if
    # they share a common neighbor structure that forms a C5.
    # 
    # Standard approach: A graph has a C5 iff there exist vertices v0, v1, v2, v3, v4 such that
    # v0-v1-v2-v3-v4-v0 are edges and all distinct.
    # 
    # We can enumerate: for each starting vertex v0, do a DFS/BFS to find paths of length 4
    # back to v0 with all distinct vertices. But this is O(n^5) in worst case.
    #
    # Better: For each pair (a, b) that are NOT adjacent, check if there's a path a-x-y-z-b of
    # length 4 where x, y, z are distinct and not equal to a or b, and all edges exist.
    # Then a-x-y-z-b-a would be a 5-cycle only if a-b is an edge... wait no.
    #
    # Let me think again. A 5-cycle v0-v1-v2-v3-v4-v0 means:
    # - v0 adjacent to v1 and v4
    # - v1 adjacent to v2
    # - v2 adjacent to v3
    # - v3 adjacent to v4
    # All 5 vertices distinct.
    #
    # Approach: For each vertex v0, for each pair (v1, v4) of neighbors of v0 (v1 != v4),
    # check if there's a path v1-v2-v3-v4 of length 3 where v2, v3 are not v0, v1, or v4,
    # and v2 != v3.
    #
    # This is O(n * deg(v0)^2 * n^2) in worst case. For n=35, this should be fast enough.

    # Build neighbor lists
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                neighbors[i].append(j)
                neighbors[j].append(i)
    
    for v0 in range(n):
        nbrs0 = neighbors[v0]
        for idx1 in range(len(nbrs0)):
            for idx4 in range(idx1+1, len(nbrs0)):
                v1 = nbrs0[idx1]
                v4 = nbrs0[idx4]
                # Find path v1-v2-v3-v4 with v2, v3 not in {v0, v1, v4}
                for v2 in neighbors[v1]:
                    if v2 == v0 or v2 == v4:
                        continue
                    for v3 in neighbors[v2]:
                        if v3 == v0 or v3 == v1 or v3 == v4:
                            continue
                        if adj[v3][v4]:
                            return True
    return False

def _max_independent_set_size(adj, n):
    """Return the size of the maximum independent set."""
    # Use a simple branch and bound
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                neighbors[i].add(j)
                neighbors[j].add(i)
    
    best = [0]
    
    def branch(candidates, current_size):
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + len(candidates) <= best[0]:
            return
        # Pick a vertex with max degree in the candidate subgraph for branching
        cand_list = list(candidates)
        # Choose vertex with highest degree among candidates
        max_deg = -1
        chosen = cand_list[0]
        for v in cand_list:
            deg = sum(1 for u in cand_list if u != v and adj[v][u])
            if deg > max_deg:
                max_deg = deg
                chosen = v
        
        # Branch 1: include chosen
        new_cand = {u for u in candidates if u != chosen and not adj[chosen][u]}
        branch(new_cand, current_size + 1)
        
        # Branch 2: exclude chosen
        new_cand2 = candidates - {chosen}
        branch(new_cand2, current_size)
    
    branch(set(range(n)), 0)
    return best[0]

def _is_valid(adj, n):
    """Check if graph is C5-free and has no independent set of size 9."""
    if _has_c5(adj, n):
        return False
    if _max_independent_set_size(adj, n) >= 9:
        return False
    return True

def _build_adj(edges, n):
    """Build adjacency matrix from edge list."""
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Best known: 8 disjoint K4s gives n=32
    # Try to find better by adding vertices and edges
    
    # Start with 8 disjoint K4s (n=32)
    base_n = 32
    base_edges = []
    for b in range(8):
        for i in range(4):
            for j in range(i+1, 4):
                base_edges.append([4*b + i, 4*b + j])
    
    best_result = {'n': base_n, 'edges': base_edges}
    best_n = base_n
    
    # Verify base is valid
    base_adj = _build_adj(base_edges, base_n)
    if _is_valid(base_adj, base_n):
        best_n = base_n
    
    # Try to add vertices one by one
    # Strategy: try adding a new vertex and connecting it to existing vertices
    # to maintain C5-free and alpha < 9
    
    # For n=33, 34, 35: try to find valid graphs
    for target_n in range(base_n + 1, 36):
        if time.time() > deadline:
            break
        
        # Start from base and try to extend
        # Use local search: start with a random valid graph on target_n vertices
        
        # Approach: Start with 8 K4s + (target_n - 32) extra vertices
        # The extra vertices need to be connected such that:
        # 1. No C5 is formed
        # 2. Alpha remains < 9
        
        # Let's try a randomized local search approach
        found = False
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Create adjacency matrix for target_n vertices
            # Initialize with 8 K4s
            adj = [[False]*target_n for _ in range(target_n)]
            for b in range(8):
                for i in range(4):
                    for j in range(i+1, 4):
                        adj[4*b+i][4*b+j] = True
                        adj[4*b+j][4*b+i] = True
            
            # For extra vertices (32 to target_n-1), connect them randomly
            # but avoid creating C5s
            for v in range(32, target_n):
                # Connect v to some subset of existing vertices
                # To avoid C5, v should not create a path of length 4 back to itself
                # Simple heuristic: connect v to at most 2 vertices from each K4
                # or connect to a limited set
                
                # Try connecting to random subsets
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    
                    # Reset adj for vertex v
                    for u in range(32):
                        adj[v][u] = False
                        adj[u][v] = False
                    
                    # Connect v to random vertices, but check for C5
                    candidates = list(range(32))
                    random.shuffle(candidates)
                    connected = []
                    for u in candidates:
                        # Check if adding edge (v, u) creates a C5
                        adj[v][u] = True
                        adj[u][v] = True
                        if _has_c5(adj, target_n):
                            adj[v][u] = False
                            adj[u][v] = False
                        else:
                            connected.append(u)
                            if len(connected) >= 4:  # Limit connections
                                break
                    
                    # Check if alpha < 9
                    if _max_independent_set_size(adj, target_n) < 9:
                        # Valid!
                        edges = []
                        for i in range(target_n):
                            for j in range(i+1, target_n):
                                if adj[i][j]:
                                    edges.append([i, j])
                        best_result = {'n': target_n, 'edges': edges}
                        best_n = target_n
                        found = True
                        break
                    else:
                        # Revert
                        for u in connected:
                            adj[v][u] = False
                            adj[u][v] = False
            
            if found:
                break
    
    # Return best result
    return best_result
# EVOLVE-BLOCK-END
