"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _count_c4(A):
    """Count 4-cycles in graph with adjacency matrix A."""
    # Number of C4 = (trace(A^4) - 2*m - 4*sum_i(deg_i choose 2)) / 8
    # More efficiently: count via common neighbors
    n = A.shape[0]
    c4_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j]:
                continue
            # Count common neighbors of i and j
            common = np.sum(A[i] & A[j])
            c4_count += common
    return c4_count // 2  # Each C4 counted twice (once for each pair of opposite vertices)

def _count_c4_fast(A):
    """Fast count of 4-cycles using matrix operations."""
    # A^2[i][j] = number of length-2 paths from i to j
    A2 = A @ A
    # trace(A^4) counts closed walks of length 4
    # Each C4 contributes 8 to trace(A^4) (4 starting points × 2 directions)
    # Each 2-walk (back and forth) contributes 2
    # Each triangle contributes 12 (but we assume no triangles for C4-free base)
    # For a simple graph: #C4 = (trace(A^4) - 2*sum_i deg_i^2 + 4*m) / 8
    # Actually: trace(A^4) = 8*C4 + 2*sum_i C(deg_i,2) + 2*m*2
    # Let me use the direct formula:
    # #C4 = (1/8) * (trace(A^4) - 2*sum_i deg_i^2 + 4*m)
    deg = np.sum(A, axis=1)
    m = np.sum(A) // 2
    A2 = A @ A
    trace_A4 = np.sum(A2 * A2)  # trace(A^4) = sum_ij (A^2[i][j])^2
    # For each pair (i,j), A2[i][j] counts paths of length 2
    # A closed walk of length 4: i->a->b->c->i
    # If i,j are not adjacent: paths i-a-j and i-c-j form a C4
    # If i,j are adjacent: paths i-a-j and i-b-j where a,b are common neighbors
    # Actually, let me just count directly:
    c4 = 0
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            if not A[i][j]:
                c4 += np.sum(A[i] & A[j])
    return c4 // 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    q = 5
    n = 31
    
    # Enumerate points of PG(2,5)
    # 25 affine points (x, y, 1)
    # 5 points at infinity (1, t, 0)
    # 1 point at infinity (0, 1, 0)
    points = []
    for x in range(q):
        for y in range(q):
            points.append([x, y, 1])
    for t in range(q):
        points.append([1, t, 0])
    points.append([0, 1, 0])
    
    # Verify n
    assert len(points) == n
    
    # Build adjacency matrix
    # A[i][j] = 1 if dot(points[i], points[j]) % q == 0 and i != j
    A = np.zeros((n, n), dtype=np.int64)
    
    for i in range(n):
        for j in range(i + 1, n):
            dot = (points[i][0]*points[j][0] + points[i][1]*points[j][1] + points[i][2]*points[j][2]) % q
            if dot == 0:
                A[i][j] = 1
                A[j][i] = 1
    
    # Count edges
    edge_count = np.sum(A) // 2
    
    # The polarity graph of PG(2,5) should have 90 edges and be C4-free.
    if edge_count != 90:
        pass
    
    # We need 91 edges.
    # We add one edge (u, v) that is not currently present.
    # The number of new C4s created is the number of length-3 paths from u to v in the original graph.
    # This is given by (A^3)[u][v].
    
    # Compute A^3
    A2 = A @ A
    A3 = A2 @ A
    
    # Find the pair (u, v) with u < v, A[u][v] == 0, and minimal A3[u][v]
    min_c4s = float('inf')
    best_u = -1
    best_v = -1
    
    for u in range(n):
        for v in range(u + 1, n):
            if A[u][v] == 0:
                c4s = A3[u][v]
                if c4s < min_c4s:
                    min_c4s = c4s
                    best_u = u
                    best_v = v
    
    # Add the best edge
    if best_u != -1:
        A[best_u][best_v] = 1
        A[best_v][best_u] = 1
    else:
        # Should not happen for n=31, m=90
        # Pick any non-edge
        for u in range(n):
            for v in range(u + 1, n):
                if A[u][v] == 0:
                    A[u][v] = 1
                    A[v][u] = 1
                    best_u = u
                    best_v = v
                    break
            if best_u != -1:
                break
    
    # Now perform local search: try to swap edges to reduce C4 count
    # A swap: remove edge (a,b) and add edge (c,d) where (c,d) is a non-edge
    # This maintains exactly 91 edges
    # We want to minimize the number of 4-cycles
    
    def _get_edges(A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i][j]:
                    edges.append((i, j))
        return edges
    
    def _try_swap(A, remove_e, add_e):
        """Try swapping: remove edge remove_e, add edge add_e.
        Return delta in C4 count."""
        a, b = remove_e
        c, d = add_e
        
        # Remove edge (a,b)
        A[a][b] = 0
        A[b][a] = 0
        
        # Count C4s after removal
        c4_after_remove = _count_c4_fast(A)
        
        # Add edge (c,d)
        A[c][d] = 1
        A[d][c] = 1
        
        # Count C4s after addition
        c4_after_add = _count_c4_fast(A)
        
        # Undo
        A[c][d] = 0
        A[d][c] = 0
        A[a][b] = 1
        A[b][a] = 1
        
        return c4_after_add - c4_after_remove
    
    # Local search: try all single-edge swaps
    best_c4 = _count_c4_fast(A)
    improved = True
    max_iterations = 50
    
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        current_edges = _get_edges(A)
        
        # Try swapping each edge with each non-edge
        for remove_e in current_edges:
            if (time.time() - start_time) > deadline - 5:
                break
            a, b = remove_e
            # Temporarily remove
            A[a][b] = 0
            A[b][a] = 0
            
            # Find best non-edge to add
            best_delta = 0
            best_add = None
            
            # Compute A^3 for the graph without edge (a,b)
            A_temp = A.copy()
            A2_temp = A_temp @ A_temp
            A3_temp = A2_temp @ A_temp
            
            for c in range(n):
                for d in range(c + 1, n):
                    if A_temp[c][d] == 0 and (c != a and d != b and c != b and d != a):
                        # This is a non-edge not involving a or b
                        c4s = A3_temp[c][d]
                        if c4s < best_delta:
                            best_delta = c4s
                            best_add = (c, d)
            
            # Restore
            A[a][b] = 1
            A[b][a] = 1
            
            if best_add is not None and best_delta < 0:
                # Perform the swap
                c, d = best_add
                A[a][b] = 0
                A[b][a] = 0
                A[c][d] = 1
                A[d][c] = 1
                best_c4 = _count_c4_fast(A)
                improved = True
                break
        
        if not improved:
            # Try random swaps if no improving swap found
            for _ in range(100):
                if (time.time() - start_time) > deadline - 5:
                    break
                current_edges = _get_edges(A)
                if not current_edges:
                    break
                remove_e = random.choice(current_edges)
                a, b = remove_e
                # Find a random non-edge
                non_edges = []
                for c in range(n):
                    for d in range(c + 1, n):
                        if A[c][d] == 0:
                            non_edges.append((c, d))
                if not non_edges:
                    break
                add_e = random.choice(non_edges)
                
                # Try the swap
                delta = _try_swap(A, remove_e, add_e)
                if delta < 0:
                    # Accept
                    A[a][b] = 0
                    A[b][a] = 0
                    c, d = add_e
                    A[c][d] = 1
                    A[d][c] = 1
                    best_c4 = _count_c4_fast(A)
                    improved = True
                    break
    
    # Extract edges list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] == 1:
                edges.append([i, j])
    
    # Verify edge count is 91
    assert len(edges) == 91, f"Edge count is {len(edges)}, expected 91"
    
    return edges
# EVOLVE-BLOCK-END
