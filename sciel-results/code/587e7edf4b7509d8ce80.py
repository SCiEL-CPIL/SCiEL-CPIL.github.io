import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    q = 5
    n = q * q + q + 1  # 31
    
    # Generate points of PG(2, q) as homogeneous coordinates
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                if x != 0:
                    inv_x = pow(x, -1, q)
                    points.append([1, y * inv_x % q, z * inv_x % q])
                elif y != 0:
                    inv_y = pow(y, -1, q)
                    points.append([0, 1, z * inv_y % q])
                else:
                    points.append([0, 0, 1])
    
    # Verify count
    if len(points) != n:
        return [[i, j] for i in range(n) for j in range(i + 1, n)][:91]
        
    def count_c4(edges):
        # Count 4-cycles in graph given by edge list
        # For small n, adjacency matrix is efficient
        adj = [[0]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
        c4_count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    continue
                # Count common neighbors
                common = 0
                for k in range(n):
                    if adj[i][k] and adj[j][k]:
                        common += 1
                c4_count += common
        return c4_count // 2 # Each C4 counted 4 times (once per edge? No, once per pair of opposite edges)
        # Actually, number of C4s is (1/4) * sum_{u!=v} (common_neighbors(u,v))^2? No.
        # Number of C4s is (1/4) * trace(A^4) - ...
        # Simpler: A 4-cycle is a set of 4 vertices {a,b,c,d} such that edges (a,b),(b,c),(c,d),(d,a) exist.
        # My loop counts pairs (i,j) that are NOT connected.
        # If i,j not connected, common neighbors k,l form a path i-k-j and i-l-j.
        # If k != l, then i-k-j-l-i is a 4-cycle.
        # So for each non-edge (i,j), the number of 4-cycles containing the "diagonal" (i,j) is C(common, 2).
        # Total C4s = sum_{non-edge (i,j)} C(common(i,j), 2).
        # Wait, the previous logic `c4_count += common` was wrong.
        # Let's re-implement count_c4 correctly.
        
    def count_c4_correct(edges):
        adj = [[0]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
        c4_count = 0
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    common = 0
                    for k in range(n):
                        if adj[i][k] and adj[j][k]:
                            common += 1
                    c4_count += common * (common - 1) // 2
        return c4_count

    # Try to find the polarity graph
    best_edges = None
    best_score = float('inf')
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        # Random symmetric matrix
        M = [[0]*3 for _ in range(3)]
        M[0][0] = random.randint(0, q-1)
        M[1][1] = random.randint(0, q-1)
        M[2][2] = random.randint(0, q-1)
        M[0][1] = random.randint(0, q-1)
        M[1][0] = M[0][1]
        M[0][2] = random.randint(0, q-1)
        M[2][0] = M[0][2]
        M[1][2] = random.randint(0, q-1)
        M[2][1] = M[1][2]
        
        # Check if M is non-singular (determinant != 0)
        det = (M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1]) -
               M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0]) +
               M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0])) % q
        if det == 0:
            continue
            
        # Count absolute points
        abs_count = 0
        for p in points:
            val = (p[0]*(M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]*p[2]) +
                   p[1]*(M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]*p[2]) +
                   p[2]*(M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]*p[2])) % q
            if val == 0:
                abs_count += 1
                 
        if abs_count == 6:
            # Construct graph
            edges = []
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    u = points[i]
                    v = points[j]
                    dot = (u[0]*(M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2]) +
                           u[1]*(M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2]) +
                           u[2]*(M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2])) % q
                    if dot == 0:
                        edges.append([i, j])
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            if len(edges) == 90:
                # Find non-edge with min common neighbors
                min_common = float('inf')
                best_pair = None
                for i in range(n):
                    for j in range(i+1, n):
                        if not adj[i][j]:
                            common = 0
                            for k in range(n):
                                if adj[i][k] and adj[j][k]:
                                    common += 1
                            if common < min_common:
                                min_common = common
                                best_pair = (i, j)
                
                if best_pair:
                    new_edges = edges + [[best_pair[0], best_pair[1]]]
                    score = count_c4_correct(new_edges)
                    if score < best_score:
                        best_score = score
                        best_edges = new_edges
                        
                if best_score == 1:
                    return best_edges

    if best_edges:
        return best_edges
        
    # Fallback: Random graph + Local Search
    edges = []
    adj = [[0]*n for _ in range(n)]
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    for i in range(91):
        u, v = all_pairs[i]
        edges.append([u, v])
        adj[u][v] = 1
        adj[v][u] = 1
        
    current_score = count_c4_correct(edges)
    
    while time.time() < deadline:
        # Try random swaps
        improved = False
        for _ in range(100):
            if time.time() > deadline:
                break
            # Pick random edge to remove
            e_idx = random.randint(0, len(edges)-1)
            u1, v1 = edges[e_idx]
            # Pick random non-edge to add
            while True:
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 != v2 and not adj[u2][v2]:
                    break
            
            # Remove e1
            adj[u1][v1] = 0
            adj[v1][u1] = 0
            edges.pop(e_idx)
            
            # Add e2
            edges.append([u2, v2])
            adj[u2][v2] = 1
            adj[v2][u2] = 1
            
            new_score = count_c4_correct(edges)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                # Revert
                adj[u2][v2] = 0
                adj[v2][u2] = 0
                edges.pop()
                edges.insert(e_idx, [u1, v1])
                adj[u1][v1] = 1
                adj[v1][u1] = 1
                
        if not improved:
            # Perturb
            for _ in range(5):
                e_idx = random.randint(0, len(edges)-1)
                u1, v1 = edges[e_idx]
                while True:
                    u2 = random.randint(0, n-1)
                    v2 = random.randint(0, n-1)
                    if u2 != v2 and not adj[u2][v2]:
                        break
                adj[u1][v1] = 0
                adj[v1][u1] = 0
                edges.pop(e_idx)
                edges.append([u2, v2])
                adj[u2][v2] = 1
                adj[v2][u2] = 1
                current_score = count_c4_correct(edges)

    return edges
