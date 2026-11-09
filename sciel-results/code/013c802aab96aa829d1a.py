"""Rewritten program: construct() returns 91 edges on vertices 0..30 minimizing C4s.
Uses a finite-field polarity graph construction for PG(2,5) to get a C4-free core with 90 edges,
then adds one edge chosen to minimize new C4s, followed by local swap search.
"""
import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s kill

    q = 5
    # Enumerate all points of PG(2,5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx, ny, nz = 1, (y * inv_x) % 5, (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx, ny, nz = 0, 1, (z * inv_y) % 5
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Try multiple polarity forms to find the best core
    # B(p,q) = a*x1*x2 + b*y1*y2 + c*z1*z2 mod 5
    # Need non-degenerate symmetric bilinear form
    # The form must have the property that the resulting graph is C4-free
    # and has exactly 90 edges (self-orthogonal points + pairs)
    
    best_result = None
    best_total_c4 = float('inf')
    
    # Coefficients for the bilinear form
    # For PG(2,5), we need a symmetric non-degenerate bilinear form
    # Try forms: (a,b,c) where the matrix [[a,0,0],[0,b,0],[0,0,c]] is non-degenerate
    # i.e., a*b*c != 0 mod 5
    # Also need the form to be a "polarity" - every point is self-orthogonal
    # Wait, for a polarity, we need B(p,p)=0 for all p, which means a*x^2+b*y^2+c*z^2=0 for all x,y,z
    # That's only possible if a=b=c=0, which is degenerate.
    # Actually, in characteristic 5 (odd), a polarity is a symmetric bilinear form where
    # the associated quadratic form Q(p) = B(p,p) can be nonzero.
    # The graph connects p to q if B(p,q)=0.
    # For this to be a "polarity graph" with nice properties, we typically use
    # B(p,q) = x1*x2 + y1*y2 + z1*z2 or similar.
    
    # Let's try several forms:
    forms = [
        (1, 1, -1),   # x1*x2 + y1*y2 - z1*z2
        (1, 1, 1),    # x1*x2 + y1*y2 + z1*z2
        (1, -1, 1),   # x1*x2 - y1*y2 + z1*z2
        (1, -1, -1),  # x1*x2 - y1*y2 - z1*z2
        (-1, 1, 1),   # -x1*x2 + y1*y2 + z1*z2
        (-1, 1, -1),  # -x1*x2 + y1*y2 - z1*z2
        (-1, -1, 1),  # -x1*x2 - y1*y2 + z1*z2
        (-1, -1, -1), # -x1*x2 - y1*y2 - z1*z2
    ]
    
    for (a, b, c) in forms:
        if time.time() > deadline - 30:
            break
        
        def make_adjacent(a, b, c):
            def adjacent(i, j):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                val = (a*x1*x2 + b*y1*y2 + c*z1*z2) % 5
                return val == 0
            return adjacent
        
        adjacent = make_adjacent(a, b, c)
        
        # Build edge list
        edges_core = []
        for i in range(31):
            for j in range(i+1, 31):
                if adjacent(i, j):
                    edges_core.append([i, j])
        
        if len(edges_core) != 90:
            continue
        
        # Build adjacency
        adj = [set() for _ in range(31)]
        for u, v in edges_core:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find best edge to add
        best_edge = None
        best_add = float('inf')
        
        for u in range(31):
            if time.time() > deadline - 25:
                break
            for v in range(u+1, 31):
                if v in adj[u]:
                    continue
                add = 0
                for a2 in adj[u]:
                    for b2 in adj[v]:
                        if b2 in adj[a2]:
                            add += 1
                if add < best_add:
                    best_add = add
                    best_edge = [u, v]
        
        if best_edge is None:
            continue
        
        total_c4 = best_add  # Core is C4-free, so total = added C4s
        
        if total_c4 < best_total_c4:
            best_total_c4 = total_c4
            best_result = edges_core + [best_edge]
    
    if best_result is None:
        # Fallback
        return _greedy_construct()
    
    # Now do local swap search on the best result
    final_edges = best_result
    adj = [set() for _ in range(31)]
    for u, v in final_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build edge set for fast lookup
    edge_set = set(tuple(sorted(e)) for e in final_edges)
    
    # Count total C4s in current graph
    def count_total_c4(adj_local):
        total = 0
        for u in range(31):
            for v in range(u+1, 31):
                cn = len(adj_local[u] & adj_local[v])
                total += cn * (cn - 1) // 2
        return total // 4
    
    current_c4 = count_total_c4(adj)
    
    # Local search: try swapping one edge for another
    # Remove edge (u,v), add edge (a,b), check if C4s decrease
    swapped = False
    for _ in range(3):  # Multiple passes
        if time.time() > deadline - 10:
            break
        improved = False
        for idx in range(len(final_edges)):
            if time.time() > deadline - 10:
                break
            u, v = final_edges[idx]
            # Try removing this edge and adding a different one
            # First, count C4s containing edge (u,v)
            c4_with_uv = 0
            for a2 in adj[u]:
                for b2 in adj[v]:
                    if a2 != v and b2 != u and b2 in adj[a2]:
                        c4_with_uv += 1
            
            # Try adding various non-edges
            for a3 in range(31):
                if time.time() > deadline - 10:
                    break
                for b3 in range(a3+1, 31):
                    if b3 in adj[a3]:
                        continue
                    if (a3, b3) in edge_set or (b3, a3) in edge_set:
                        continue
                    # C4s created by adding (a3,b3)
                    c4_new = 0
                    for a2 in adj[a3]:
                        for b2 in adj[b3]:
                            if b2 in adj[a2]:
                                c4_new += 1
                    
                    # Net change: -c4_with_uv + c4_new
                    # But this is approximate - removing (u,v) also affects C4s not containing (u,v)
                    # For a quick heuristic, just check if c4_new < c4_with_uv
                    if c4_new < c4_with_uv:
                        # Do the swap
                        adj[u].remove(v)
                        adj[v].remove(u)
                        edge_set.discard((u, v))
                        edge_set.discard((v, u))
                        
                        final_edges[idx] = [a3, b3]
                        adj[a3].add(b3)
                        adj[b3].add(a3)
                        edge_set.add(tuple(sorted([a3, b3])))
                        
                        current_c4 = count_total_c4(adj)
                        improved = True
                        swapped = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    # Validate
    assert len(final_edges) == 91
    assert len(set(tuple(sorted(e)) for e in final_edges)) == 91
    
    return final_edges

def _greedy_construct():
    """Fallback greedy construction."""
    import random
    random.seed(42)
    
    edges = []
    adj = [set() for _ in range(31)]
    
    remaining = []
    for i in range(31):
        for j in range(i+1, 31):
            remaining.append([i, j])
    
    random.shuffle(remaining)
    
    for _ in range(91):
        best_e = None
        best_add = float('inf')
        for e in remaining:
            u, v = e
            if v in adj[u]:
                continue
            add = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        add += 1
            if add < best_add:
                best_add = add
                best_e = e
            if add == 0:
                break
        if best_e is None:
            break
        u, v = best_e
        edges.append(best_e)
        adj[u].add(v)
        adj[v].add(u)
        remaining.remove(best_e)
    
    while len(edges) < 91 and remaining:
        for e in remaining:
            u, v = e
            if v not in adj[u]:
                edges.append(e)
                adj[u].add(v)
                adj[v].add(u)
                remaining.remove(e)
                break
    
    return edges[:91]
