"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    n = 31
    m = 91
    q = 5
    
    # Build the polarity graph of PG(2, q) for q=5
    # Points in PG(2, q): (x, y, z) up to scalar, not all zero
    # For q=5: q^2 + q + 1 = 31 points
    
    # Enumerate all points in PG(2, 5)
    points = []
    for z in range(5):
        for y in range(5):
            for x in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                pt = [x, y, z]
                for i in range(3):
                    if pt[i] != 0:
                        inv = pow(pt[i], q - 2, q)
                        pt = [(pt[j] * inv) % q for j in range(3)]
                        break
                if pt not in points:
                    points.append(pt)
    
    # Should have 31 points
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define the polarity: x^2 + y^2 + z^2 = 0 (or use a non-degenerate quadratic form)
    # Two points are adjacent if they are orthogonal under the bilinear form
    def is_orthogonal(u, v):
        return (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q == 0
    
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=int)
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            if is_orthogonal(points[i], points[j]):
                A[i, j] = 1
                A[j, i] = 1
                edges.add((i, j))
    
    edge_count = len(edges)
    
    # The polarity graph of PG(2,5) should have 90 edges and 0 C4s
    # Verify edge count
    if edge_count != 90:
        # Fallback if construction is slightly off, though it should be exact
        # For now, assume it is 90. If not, we adjust.
        pass
        
    # If we have more than 90 edges, we must remove some to get a C4-free core?
    # But the theory says it is 90. Let's assume it is.
    
    # We need exactly 91 edges.
    # We have 90 edges with 0 C4s.
    # We must add 1 edge.
    # Adding edge (u, v) creates P3(u, v) new C4s.
    # We want to minimize P3(u, v).
    
    # Precompute neighbors for fast path counting
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                neighbors[i].add(j)
    
    # Find the non-edge (u, v) that minimizes the number of length-3 paths between u and v
    min_c4s = float('inf')
    best_edge = None
    
    # List of all non-edges
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edges:
                non_edges.append((i, j))
                
    for u, v in non_edges:
        # Count paths of length 3 from u to v
        # Path: u - x - y - v
        # x in N(u), y in N(x) \ {u}, v in N(y)
        count = 0
        for x in neighbors[u]:
            for y in neighbors[x]:
                if y == u:
                    continue
                if v in neighbors[y]:
                    count += 1
                    # Early exit if we already have 0? No, we are counting.
                    # If count > min_c4s, we can break early to save time
                    if count > min_c4s:
                        break
            if count > min_c4s:
                break
        
        if count < min_c4s:
            min_c4s = count
            best_edge = (u, v)
            if min_c4s == 0:
                # Theoretically impossible if ex(31, C4)=90, but check just in case
                break

    # Add the best edge
    if best_edge is not None:
        u, v = best_edge
        A[u, v] = 1
        A[v, u] = 1
        edges.add(best_edge)
        neighbors[u].add(v)
        neighbors[v].add(u)
    else:
        # Fallback: add any edge
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edges:
                    A[i, j] = 1
                    A[j, i] = 1
                    edges.add((i, j))
                    neighbors[i].add(j)
                    neighbors[j].add(i)
                    break
            if len(edges) == m:
                break

    assert len(edges) == m, f"Expected {m} edges, got {len(edges)}"
    
    # Count C4s in the current graph to confirm
    def count_c4s(adj):
        n = adj.shape[0]
        total = 0
        for u in range(n):
            for w in range(u+1, n):
                if adj[u, w] == 1:
                    continue
                common = np.sum(adj[u] & adj[w])
                total += common * (common - 1) // 2
        return total // 2
    
    current_c4 = count_c4s(A)
    
    # Optional: Local search to try to reduce C4s further
    # Since we started with the theoretical minimum structure, improvement is unlikely,
    # but we can try a few swaps if time permits.
    
    best_c4 = current_c4
    best_A = A.copy()
    best_edges = set(edges)
    
    # If we have time, try to improve
    if time.time() < deadline - 10:
        # Simple local search
        all_edges = list(best_edges)
        non_edges_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in best_edges:
                    non_edges_set.add((i, j))
        non_edge_list = list(non_edges_set)
        
        improvement_found = True
        while improvement_found and time.time() < deadline - 5:
            improvement_found = False
            random.shuffle(all_edges)
            random.shuffle(non_edge_list)
            
            # Try a limited number of swaps
            for rem_e in all_edges[:100]: # Limit to first 100 to save time
                if time.time() > deadline - 5:
                    break
                for add_e in non_edge_list[:100]:
                    if time.time() > deadline - 5:
                        break
                    if add_e in best_edges:
                        continue
                    if add_e == rem_e:
                        continue
                    
                    # Estimate C4 change
                    # This is a heuristic
                    ru, rv = rem_e
                    au, av = add_e
                    
                    # Count C4s containing rem_e
                    c4s_rem = 0
                    for x in best_A[ru].nonzero()[0]:
                        if x == rv: continue
                        for y in best_A[rv].nonzero()[0]:
                            if y == ru or y == x: continue
                            if best_A[x, y] == 1:
                                c4s_rem += 1
                    # This counts each C4 twice? Or once?
                    # Path ru-x-rv-y-ru? No.
                    # C4 containing (ru, rv): ru-rv-x-y-ru.
                    # x in N(rv), y in N(ru), (x,y) edge.
                    # So we iterate x in N(rv), y in N(ru), check (x,y).
                    
                    c4s_add = 0
                    for x in best_A[au].nonzero()[0]:
                        for y in best_A[av].nonzero()[0]:
                            if x == y: continue
                            if best_A[x, y] == 1:
                                c4s_add += 1
                            
                    # Note: This heuristic is rough. 
                    # A more accurate delta is needed for reliable improvement.
                    # Given the low starting C4 count, random swaps might increase C4s.
                    # We will skip complex local search to avoid worsening the score.
                    pass
                    
    # Return the best edges found
    result = [[u, v] for u, v in sorted(best_edges)]
    assert len(result) == 91
    return result
# EVOLVE-BLOCK-END
