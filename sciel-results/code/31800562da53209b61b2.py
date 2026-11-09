"""Program to find the densest triangle-free graph on 16 vertices with chi >= 4."""
import sys
from itertools import combinations

def construct():
    n = 16
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m_total = len(all_edges)  # 120
    
    # Edge index
    edge_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # Build conflict graph: two edges conflict if they share a vertex (would form triangle)
    # conflict_mask[e] is a bitset of edges that conflict with edge e
    conflict_mask = [0] * m_total
    for i, (u1, v1) in enumerate(all_edges):
        mask = 0
        for j, (u2, v2) in enumerate(all_edges):
            if i == j:
                continue
            # Check if edges (u1,v1) and (u2,v2) share a vertex
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Precompute adjacency for fast triangle checking
    # adj[u] is a bitset of vertices adjacent to u
    # For checking if adding edge (u,v) creates a triangle:
    # we need no common neighbor of u and v
    
    # 3-colorability check using backtracking with bitsets
    def is_3_colorable(adj_list):
        """Check if graph with adjacency list is 3-colorable."""
        colors = [-1] * n
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -len(adj_list[x]))
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Try each color
            for c in range(3):
                # Check if color c is available (no neighbor has color c)
                conflict = False
                for nb in adj_list[v]:
                    if colors[nb] == c:
                        conflict = True
                        break
                if not conflict:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)
    
    # Maximum independent set in conflict graph using branch and bound
    # We want the largest set of edges with no conflicts (triangle-free)
    
    # Use a greedy + local search approach for speed
    # First, try to find a large triangle-free set
    
    best_edges = set()
    best_count = 0
    
    # Greedy construction: pick edges that don't conflict
    # Try multiple random orderings
    import random
    random.seed(42)
    
    for trial in range(50):
        order = list(range(m_total))
        random.shuffle(order)
        
        current = set()
        for e in order:
            # Check if e conflicts with any edge in current
            conflict = False
            for ce in current:
                if (conflict_mask[e] >> ce) & 1:
                    conflict = True
                    break
            if not conflict:
                current.add(e)
        
        if len(current) > best_count:
            best_count = len(current)
            best_edges = current.copy()
    
    # Local search: try to add more edges or swap
    # Check if current best is 3-colorable
    def build_adj(edge_set):
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Verify best_edges is triangle-free and check colorability
    adj = build_adj(best_edges)
    if not is_3_colorable(adj):
        return [[u, v] for (u, v) in best_edges]
    
    # Try to improve via edge swaps
    # For each edge not in best_edges, try adding it and removing conflicting edges
    improved = True
    while improved:
        improved = False
        for e in range(m_total):
            if e in best_edges:
                continue
            # Check if e can be added
            # Find all edges in best_edges that conflict with e
            conflicting = set()
            for ce in best_edges:
                if (conflict_mask[e] >> ce) & 1:
                    conflicting.add(ce)
            
            # If we can add e and remove fewer than 1 edges, it's an improvement
            # Actually, we need to check if the new set is still triangle-free
            # and potentially larger
            
            # Try: add e, remove all conflicting edges, see if we can re-add some
            new_set = (best_edges - conflicting) | {e}
            
            # Check triangle-free
            adj_new = build_adj(new_set)
            # Verify no triangles
            triangle_free = True
            for u in range(n):
                for v in adj_new[u]:
                    if v > u:
                        for w in adj_new[v]:
                            if w > v and w in adj_new[u]:
                                triangle_free = False
                                break
                        if not triangle_free:
                            break
                if not triangle_free:
                    break
            
            if not triangle_free:
                continue
            
            # Check 3-colorability
            if not is_3_colorable(adj_new):
                if len(new_set) > best_count:
                    best_count = len(new_set)
                    best_edges = new_set
                    improved = True
                    break
    
    # Final verification
    adj = build_adj(best_edges)
    if not is_3_colorable(adj):
        return [[u, v] for (u, v) in best_edges]
    
    # Fallback: return Clebsch graph
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            xor = u ^ v
            bits = bin(xor).count('1')
            if bits == 1 or bits == 4:
                edges.append([u, v])
    return edges
