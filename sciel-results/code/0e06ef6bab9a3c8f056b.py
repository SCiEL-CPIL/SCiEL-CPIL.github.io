import numpy as np

def construct():
    n = 31
    target_edges = 91
    
    # Generate all points in PG(2, 5)
    seen = set()
    pts = []
    for z in range(5):
        for y in range(5):
            for x in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    nx = 1
                    ny = (y * inv_x) % 5
                    nz = (z * inv_x) % 5
                    rep = (nx, ny, nz)
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    nx = 0
                    ny = 1
                    nz = (z * inv_y) % 5
                    rep = (nx, ny, nz)
                else: # z != 0
                    inv_z = pow(z, -1, 5)
                    nx = 0
                    ny = 0
                    nz = 1
                    rep = (nx, ny, nz)
                
                if rep not in seen:
                    seen.add(rep)
                    pts.append(rep)
    
    assert len(pts) == 31, f"Expected 31 points, got {len(pts)}"
    
    def build_graph(M):
        A = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                u = np.array(pts[i])
                v = np.array(pts[j])
                val = (u @ M @ v) % 5
                if val == 0:
                    A[i, j] = 1
                    A[j, i] = 1
        return A

    def count_edges(A):
        return int(np.sum(A) / 2)

    def count_4_cycles(A):
        common = A @ A
        iu = np.triu_indices(n, k=1)
        vals = common[iu]
        return int(np.sum(vals * (vals - 1) / 2))

    def get_edge_4_cycle_counts(A):
        counts = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v]:
                    N_u = np.where(A[u] == 1)[0]
                    N_v = np.where(A[v] == 1)[0]
                    # Exclude each other
                    N_u = N_u[N_u != v]
                    N_v = N_v[N_v != u]
                    if len(N_u) > 0 and len(N_v) > 0:
                        sub = A[np.ix_(N_u, N_v)]
                        cnt = int(np.sum(sub))
                        counts[u, v] = cnt
                        counts[v, u] = cnt
        return counts

    def remove_edge_most_4_cycles(A):
        counts = get_edge_4_cycle_counts(A)
        max_val = np.max(counts)
        if max_val == 0:
            # If no 4-cycles, remove any edge.
            # Find an edge
            idx = np.where(A > 0)
            if len(idx[0]) == 0:
                return None
            u, v = idx[0][0], idx[1][0]
            A[u, v] = 0
            A[v, u] = 0
            return (u, v)
        idx = np.unravel_index(np.argmax(counts), counts.shape)
        u, v = idx
        A[u, v] = 0
        A[v, u] = 0
        return (u, v)

    def add_edge_min_4_cycles(A):
        best_u, best_v = -1, -1
        min_cyc = float('inf')
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 0:
                    N_v = np.where(A[v] == 1)[0]
                    N_u = np.where(A[u] == 1)[0]
                    if len(N_v) > 0 and len(N_u) > 0:
                        sub = A[np.ix_(N_v, N_u)]
                        new_cyc = int(np.sum(sub))
                    else:
                        new_cyc = 0
                        
                    if new_cyc < min_cyc:
                        min_cyc = new_cyc
                        best_u, best_v = u, v
        if best_u == -1:
            return None
        A[best_u, best_v] = 1
        A[best_v, best_u] = 1
        return (best_u, best_v)

    # Candidate matrices
    candidates = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=int),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 3]], dtype=int),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 4]], dtype=int),
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=int),
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 2]], dtype=int),
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 3]], dtype=int),
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 4]], dtype=int),
    ]

    best_A = None
    best_edges = -1
    best_score = float('inf')
    
    # First, try to find a graph with >= 91 edges
    for M in candidates:
        A = build_graph(M)
        e = count_edges(A)
        if e >= target_edges:
            # We can just remove edges to get to 91.
            # Since it's C4-free, score will be 0.
            # We pick the one with smallest e >= 91 to be safe? 
            # Actually any works.
            # Let's pick the first one found.
            best_A = A
            best_edges = e
            break
    
    if best_A is None:
        # No graph has >= 91 edges.
        # Pick the one with the most edges (closest to 91) to minimize additions.
        max_edges_found = -1
        for M in candidates:
            A = build_graph(M)
            e = count_edges(A)
            if e > max_edges_found:
                max_edges_found = e
                best_A = A
                best_edges = e
        
        # Now adjust best_A to 91 edges by adding
        while best_edges < target_edges:
            res = add_edge_min_4_cycles(best_A)
            if res is None:
                break
            best_edges += 1
            
    # If we found a graph with >= 91 edges, we need to remove edges
    if best_edges > target_edges:
        while best_edges > target_edges:
            res = remove_edge_most_4_cycles(best_A)
            if res is None:
                break
            best_edges -= 1
            
    # Extract edges
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v]:
                edges_list.append([u, v])
                
    return edges_list
