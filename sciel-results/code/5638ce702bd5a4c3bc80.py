import random
import time
import numpy as np

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 148  # Safety margin before 150s limit
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Build adjacency as numpy matrix for fast C4 counting
    def count_c4_from_adj_matrix(A):
        """Count C4s given adjacency as numpy matrix.
        C4 count = sum over all pairs (i,j) of C(common_neighbors(i,j), 2)
        Using matrix: (A^2)[i][j] = number of common neighbors of i and j
        C4 count = 1/4 * sum_{i!=j} C((A^2)[i][j], 2)
        But we need to be careful: each C4 is counted 4 times (once per edge in the C4)
        Actually, let's use the formula:
        Number of C4s = 1/4 * sum_{i,j} C((A^2)[i][j], 2) - sum_i C(d_i, 2)
        Wait, that's not right either. Let me think again.
        
        A C4 is a sequence of 4 distinct vertices v1-v2-v3-v4-v1.
        For each pair of non-adjacent vertices (i,j), the number of C4s containing edge (i,j)
        is not directly given. 
        
        Better approach: 
        For each pair of vertices (i,j), let c_ij = (A^2)[i][j] = number of common neighbors.
        The number of 4-cycles is:
        1/4 * sum_{i != j} C(c_ij, 2)
        because each C4 v1-v2-v3-v4-v1 is counted 4 times:
        - once for pair (v1,v3) with common neighbors v2,v4
        - once for pair (v2,v4) with common neighbors v1,v3
        Wait, that's only 2. Let me reconsider.
        
        Actually, a C4 v1-v2-v3-v4-v1 has two diagonals: (v1,v3) and (v2,v4).
        For diagonal (v1,v3): common neighbors include v2 and v4, so C(common,2) >= 1
        For diagonal (v2,v4): common neighbors include v1 and v3, so C(common,2) >= 1
        So each C4 is counted exactly 2 times in sum_{i!=j} C(c_ij, 2).
        
        But wait, we also need to subtract cases where i and j are adjacent? No,
        C(c_ij, 2) counts all pairs of common neighbors, which forms a C4 only if
        the two common neighbors are not adjacent to each other... no wait, they form
        the C4 regardless.
        
        Let me verify: if i and j have common neighbors a and b, then i-a-j-b-i is a C4
        provided a != b and all four are distinct. Since a,b are neighbors of both i and j,
        and a != b, we have vertices i,a,j,b all distinct (a != i,j since simple graph,
        b != i,j similarly, a != b by choice). So yes, this is a C4.
        
        So: C4_count = (1/2) * sum_{i != j} C((A^2)[i][j], 2)
        
        But (A^2)[i][i] = degree(i), and C(degree,2) would count triangles, not C4s.
        We only sum over i != j.
        """
        A2 = A @ A
        # For each pair i != j, compute C(A2[i,j], 2)
        c4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = A2[i, j]
                if c >= 2:
                    c4_count += c * (c - 1) // 2
        # Each C4 is counted exactly 2 times (once for each diagonal)
        return c4_count // 2
    
    def edge_set_to_matrix(edge_set):
        A = np.zeros((n, n), dtype=np.int32)
        for u, v in edge_set:
            A[u, v] = 1
            A[v, u] = 1
        return A
    
    def matrix_to_edge_set(A):
        edge_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j]:
                    edge_set.add((i, j))
        return edge_set
    
    # PG(2,5) construction: 31 vertices as projective points over GF(5)
    # Represent points as 3-tuples (x,y,z) not all zero, modulo scaling
    # There are (5^3 - 1)/(5-1) = 124/4 = 31 points
    
    def build_pg25_points():
        """Build the 31 points of PG(2,5)."""
        points = []
        seen = set()
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize: make first non-zero entry = 1
                    if x != 0:
                        key = (1, y * pow(x, 4, 5), z * pow(x, 4, 5))
                    elif y != 0:
                        key = (0, 1, z * pow(y, 4, 5))
                    else:
                        key = (0, 0, 1)
                    if key not in seen:
                        seen.add(key)
                        points.append(key)
        return points
    
    def build_polarity_graph(points):
        """Build a polarity graph of PG(2,5).
        A polarity is a symmetric bilinear form B: V x V -> GF(5)
        We use the standard polarity: B((x1,y1,z1),(x2,y2,z2)) = x1*x2 + y1*y2 + z1*z2
        Two points are adjacent if B(p,q) = 0 (orthogonal).
        """
        edge_set = set()
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p = points[i]
                q = points[j]
                # Compute bilinear form
                b = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
                if b == 0:
                    edge_set.add((i, j))
        return edge_set
    
    # Try the polarity graph as starting point
    points = build_pg25_points()
    polarity_edges = build_polarity_graph(points)
    
    # Count edges in polarity graph
    pol_adj = edge_set_to_matrix(polarity_edges)
    pol_c4 = count_c4_from_adj_matrix(pol_adj)
    
    # We need exactly 91 edges. The polarity graph may have more or fewer.
    # If it has exactly 90 (ex(31,C4)), we add 1 edge.
    # If it has more, we need to remove edges.
    
    best_edges_set = None
    best_c4 = float('inf')
    
    # Start from polarity graph and adjust to 91 edges
    current_edges = set(polarity_edges)
    
    # If we have too many edges, remove some
    while len(current_edges) > target_edges:
        # Remove edge that is in most C4s
        A = edge_set_to_matrix(current_edges)
        best_remove = None
        best_score = float('inf')
        for e in list(current_edges):
            u, v = e
            # C4s containing edge (u,v)
            common = set()
            for w in range(n):
                if A[u, w] and A[v, w] and w != u and w != v:
                    common.add(w)
            c4s = 0
            cn_list = list(common)
            for idx, a in enumerate(cn_list):
                for b in cn_list[idx+1:]:
                    if A[a, b]:
                        c4s += 1
            if c4s < best_score:
                best_score = c4s
                best_remove = e
        if best_remove:
            current_edges.discard(best_remove)
    
    # If we have too few edges, add some
    while len(current_edges) < target_edges:
        # Add edge that creates fewest C4s
        A = edge_set_to_matrix(current_edges)
        best_add = None
        best_score = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                if not A[i, j]:
                    common = 0
                    for w in range(n):
                        if A[i, w] and A[j, w]:
                            common += 1
                    c4s = common * (common - 1) // 2
                    if c4s < best_score:
                        best_score = c4s
                        best_add = (i, j)
        if best_add:
            current_edges.add(best_add)
    
    current_adj = edge_set_to_matrix(current_edges)
    current_c4 = count_c4_from_adj_matrix(current_adj)
    
    if current_c4 < best_c4:
        best_c4 = current_c4
        best_edges_set = set(current_edges)
    
    # Local search: swap edges to reduce C4s
    # Use numpy for fast delta computation
    improved = True
    max_local_iters = 2000
    local_iter = 0
    
    while improved and local_iter < max_local_iters and time.time() < deadline:
        improved = False
        local_iter += 1
        
        A = edge_set_to_matrix(current_edges)
        edge_list = list(current_edges)
        random.shuffle(edge_list)
        
        for remove_u, remove_v in edge_list:
            if time.time() > deadline:
                break
            
            # Compute C4s destroyed by removing edge (remove_u, remove_v)
            # A C4 contains edge (u,v) if there exist a,b with u-a-v-b-u
            # i.e., a in N(u)\{v}, b in N(v)\{u}, a~b
            # Number = number of edges in induced subgraph on N(u) ∩ N(v)
            common_uv = []
            for w in range(n):
                if w != remove_u and w != remove_v and A[remove_u, w] and A[remove_v, w]:
                    common_uv.append(w)
            
            c4s_destroyed = 0
            for idx, a in enumerate(common_uv):
                for b in common_uv[idx+1:]:
                    if A[a, b]:
                        c4s_destroyed += 1
            
            # Find best edge to add
            best_add = None
            best_delta = float('inf')
            
            # Try all non-edges (only 465 total, manageable)
            for i in range(n):
                for j in range(i + 1, n):
                    if not A[i, j]:
                        # New C4s created by adding (i,j)
                        common_ij = 0
                        for w in range(n):
                            if A[i, w] and A[j, w]:
                                common_ij += 1
                        # But we also need to account for edges among common neighbors
                        # that were already present
                        new_c4s = 0
                        cn_list = []
                        for w in range(n):
                            if A[i, w] and A[j, w]:
                                cn_list.append(w)
                        for idx2, a in enumerate(cn_list):
                            for b in cn_list[idx2+1:]:
                                if A[a, b]:
                                    new_c4s += 1
                        
                        delta = -c4s_destroyed + new_c4s
                        if delta < best_delta:
                            best_delta = delta
                            best_add = (i, j)
            
            if best_add is not None and best_delta < 0:
                # Apply swap
                current_edges.discard((remove_u, remove_v))
                A[remove_u, remove_v] = 0
                A[remove_v, remove_u] = 0
                
                add_u, add_v = best_add
                current_edges.add((add_u, add_v))
                A[add_u, add_v] = 1
                A[add_v, add_u] = 1
                
                current_c4 += best_delta
                improved = True
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges_set = set(current_edges)
                
                break  # Restart after a change
    
    # Additional random restarts if time permits
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random graph near the current best
            current_edges = set(best_edges_set)
            # Perturb: remove 5 random edges, add 5 random non-edges
            edge_list = list(current_edges)
            random.shuffle(edge_list)
            for _ in range(5):
                if edge_list:
                    current_edges.discard(edge_list.pop())
            
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                        if (i, j) not in current_edges]
            random.shuffle(non_edges)
            for _ in range(5):
                if non_edges:
                    current_edges.add(non_edges.pop())
            
            A = edge_set_to_matrix(current_edges)
            current_c4 = count_c4_from_adj_matrix(A)
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges_set = set(current_edges)
            
            # Quick local search
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                A = edge_set_to_matrix(current_edges)
                edge_list = list(current_edges)
                random.shuffle(edge_list)
                for remove_u, remove_v in edge_list[:20]:
                    common_uv = []
                    for w in range(n):
                        if w != remove_u and w != remove_v and A[remove_u, w] and A[remove_v, w]:
                            common_uv.append(w)
                    c4s_destroyed = 0
                    for idx, a in enumerate(common_uv):
                        for b in common_uv[idx+1:]:
                            if A[a, b]:
                                c4s_destroyed += 1
                    
                    best_add = None
                    best_delta = float('inf')
                    for i in range(n):
                        for j in range(i + 1, n):
                            if not A[i, j]:
                                cn_list = []
                                for w in range(n):
                                    if A[i, w] and A[j, w]:
                                        cn_list.append(w)
                                new_c4s = 0
                                for idx2, a in enumerate(cn_list):
                                    for b in cn_list[idx2+1:]:
                                        if A[a, b]:
                                            new_c4s += 1
                                delta = -c4s_destroyed + new_c4s
                                if delta < best_delta:
                                    best_delta = delta
                                    best_add = (i, j)
                    
                    if best_add is not None and best_delta < 0:
                        current_edges.discard((remove_u, remove_v))
                        A[remove_u, remove_v] = 0
                        A[remove_v, remove_u] = 0
                        add_u, add_v = best_add
                        current_edges.add((add_u, add_v))
                        A[add_u, add_v] = 1
                        A[add_v, add_u] = 1
                        current_c4 += best_delta
                        if current_c4 < best_c4:
                            best_c4 = current_c4
                            best_edges_set = set(current_edges)
                        break
    
    # Convert to list of lists
    if best_edges_set is None:
        best_edges = all_edges[:target_edges]
    else:
        best_edges = [[u, v] for u, v in sorted(best_edges_set)]
    
    return best_edges
