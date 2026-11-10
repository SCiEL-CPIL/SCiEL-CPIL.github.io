"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin
    
    # Use complete bipartite graph K_{7,8} as base
    # Partition A: {0,1,2,3,4,5,6} (7 vertices)
    # Partition B: {7,8,9,10,11,12,13,14} (8 vertices)
    # In K_{7,8}, every vertex in A has degree 8, every vertex in B has degree 7
    
    A = list(range(7))
    B = list(range(7, 15))
    
    # Base edges: complete bipartite
    base_edges = [(a, b) for a in A for b in B]
    base_set = set(tuple(sorted(e)) for e in base_edges)
    
    def get_degrees(edge_set, n):
        deg = [0] * n
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edge_set):
        """Check if no simple three-edge path has endpoints of equal degree."""
        deg = get_degrees(edge_set, n)
        
        # Build adjacency for efficiency
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each edge (u,v), check all paths s-u-v-t with s != t
        for (u, v) in edge_set:
            # s is a neighbor of u (not v), t is a neighbor of v (not u)
            # and s != t
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    # Path s-u-v-t exists
                    if deg[s] == deg[t]:
                        return False
        return True
    
    # Start with base graph
    best_edges = set(base_set)
    best_count = len(best_edges)
    
    # All possible internal edges (within A or within B)
    all_internal = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            all_internal.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            all_internal.append((B[i], B[j]))
    
    # Greedy: add edges one by one if valid
    # Sort by some heuristic: edges within same partition first
    # Actually, let's just try all and add if valid
    added = set()
    for e in all_internal:
        if time.time() > deadline:
            break
        if e in added:
            continue
        candidate = best_edges | {tuple(sorted(e))}
        if is_valid(candidate):
            best_edges = candidate
            best_count = len(best_edges)
            added.add(e)
    
    # Local search: try to add more edges after greedy
    # Now try to add edges that were skipped
    remaining = [e for e in all_internal if e not in added]
    for e in remaining:
        if time.time() > deadline:
            break
        candidate = best_edges | {tuple(sorted(e))}
        if is_valid(candidate):
            best_edges = candidate
            best_count = len(best_edges)
            added.add(e)
    
    # Convert to list format
    result = [list(e) for e in best_edges]
    result.sort()
    return result
# EVOLVE-BLOCK-END
