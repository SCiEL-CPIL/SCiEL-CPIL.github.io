import numpy as np
import time

def _count_c4s(adj_matrix):
    """Count 4-cycles in a graph given by its adjacency matrix."""
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s buffer before 150s kill
    
    n = 31
    
    # Step 1: Generate the 31 points of PG(2, 5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    p = (1, (y * pow(x, 4, 5)) % 5, (z * pow(x, 4, 5)) % 5)
                elif y != 0:
                    p = (0, 1, (z * pow(y, 4, 5)) % 5)
                else:
                    p = (0, 0, 1)
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    # Step 2: Construct Adjacency Matrix for Polarity Graph
    A = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            val = (p1[0]*p2[0] + p1[1]*p2[1] + 4*p1[2]*p2[2]) % 5
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
    
    edge_count = int(A.sum()) // 2
    
    # Count C4s using O(n^4) - fast for n=31
    def count_c4s(adj):
        c4 = 0
        for a in range(n):
            for b in range(a+1, n):
                if adj[a,b] == 0:
                    continue
                for c in range(b+1, n):
                    if adj[b,c] == 0:
                        continue
                    for d in range(c+1, n):
                        if adj[c,d] == 0:
                            continue
                        if adj[d,a] and adj[a,c] == 0 and adj[b,d] == 0:
                            c4 += 1
        return c4
    
    # Better C4 count: count all 4-cycles properly
    # A 4-cycle is {a,b,c,d} with edges ab, bc, cd, da (in that cyclic order)
    # We need to count each C4 exactly once.
    # For each 4-set {a,b,c,d} with a<b<c<d, check if it forms a C4.
    # A C4 on 4 vertices means exactly 4 edges forming a cycle.
    # The 4 vertices can form: 0 edges, 1 edge, 2 edges (adjacent or opposite), 3 edges (path), 4 edges (C4 or K4 minus one edge... wait 4 edges on 4 vertices is always a C4), 5 edges, 6 edges (K4).
    # So count 4-sets with exactly 4 edges.
    
    def count_c4s_v2(adj):
        c4 = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        e = (adj[a,b] + adj[a,c] + adj[a,d] + 
                             adj[b,c] + adj[b,d] + 
                             adj[c,d])
                        if e == 4:
                            c4 += 1
        return c4
    
    current_c4s = count_c4s_v2(A)
    
    # If polarity graph has more than 91 edges, remove edges to get to 91
    # If it has fewer, add edges
    if edge_count > 91:
        # Remove edges that are in the most C4s
        # For each edge, count how many C4s it participates in
        edge_c4_count = {}
        for a in range(n):
            for b in range(a+1, n):
                if A[a,b] == 1:
                    # Count C4s containing edge (a,b)
                    cnt = 0
                    for c in range(n):
                        if c != a and c != b and A[a,c]:
                            for d in range(n):
                                if d != a and d != b and d != c and A[b,d] and A[c,d]:
                                    cnt += 1
                    edge_c4_count[(a,b)] = cnt
                    edge_c4_count[(b,a)] = cnt
        
        # Sort edges by C4 participation (descending) and remove top (edge_count - 91)
        to_remove = edge_count - 91
        sorted_edges = sorted(edge_c4_count.keys(), key=lambda e: edge_c4_count[e], reverse=True)
        # Only consider each edge once (a,b) where a<b
        removed = 0
        for e in sorted_edges:
            if removed >= to_remove:
                break
            a, b = e
            if a > b:
                a, b = b, a
            if A[a,b] == 1:
                A[a,b] = 0
                A[b,a] = 0
                removed += 1
    
    elif edge_count < 91:
        # Add edges that create the fewest C4s
        to_add = 91 - edge_count
        # For each non-edge, count C4s it would create
        # Adding (u,v) creates a C4 for each path u-x-y-v of length 3
        non_edge_delta = {}
        for u in range(n):
            for v in range(u+1, n):
                if A[u,v] == 0:
                    # Count length-3 paths from u to v
                    cnt = 0
                    for x in range(n):
                        if x != u and x != v and A[u,x]:
                            for y in range(n):
                                if y != u and y != v and y != x and A[v,y] and A[x,y]:
                                    cnt += 1
                    non_edge_delta[(u,v)] = cnt
        
        sorted_non_edges = sorted(non_edge_delta.keys(), key=lambda e: non_edge_delta[e])
        added = 0
        for e in sorted_non_edges:
            if added >= to_add:
                break
            u, v = e
            A[u,v] = 1
            A[v,u] = 1
            added += 1
    
    current_c4s = count_c4s_v2(A)
    best_c4s = current_c4s
    best_A = A.copy()
    
    # Local search: 1-in-1-out edge swaps
    def count_paths_len3(u, v, adj_bits):
        """Count paths of length 3 from u to v using bitmask representation."""
        count = 0
        nu = adj_bits[u]
        nv = adj_bits[v]
        # For each neighbor x of u, for each neighbor y of v, check if x-y is edge
        # x != v, y != u, x != y
        x_mask = nu & ~(1 << v)
        y_mask = nv & ~(1 << u)
        # For each x in x_mask, for each y in y_mask, check edge x-y
        # Iterate over set bits
        xx = x_mask
        while xx:
            lb = xx & (-xx)
            x = lb.bit_length() - 1
            xx ^= lb
            yy = y_mask & ~(1 << x)  # y != x
            while yy:
                lb2 = yy & (-yy)
                y = lb2.bit_length() - 1
                yy ^= lb2
                if adj_bits[x] & (1 << y):
                    count += 1
        return count
    
    def get_adj_bits(adj):
        bits = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i,j]:
                    mask |= (1 << j)
            bits[i] = mask
        return bits
    
    current_bits = get_adj_bits(A)
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        best_swap = None
        best_delta = 0
        
        curr_edges = [(i,j) for i in range(n) for j in range(i+1,n) if A[i,j]]
        curr_non_edges = [(i,j) for i in range(n) for j in range(i+1,n) if not A[i,j]]
        
        # For efficiency, precompute path counts for all edges and non-edges
        edge_path = {}
        for (a,b) in curr_edges:
            edge_path[(a,b)] = count_paths_len3(a, b, current_bits)
        
        non_edge_path = {}
        for (c,d) in curr_non_edges:
            non_edge_path[(c,d)] = count_paths_len3(c, d, current_bits)
        
        for (a,b) in curr_edges:
            delta_remove = -edge_path[(a,b)]
            for (c,d) in curr_non_edges:
                delta_add = non_edge_path[(c,d)]
                delta = delta_remove + delta_add
                if delta < best_delta:
                    best_delta = delta
                    best_swap = ((a,b), (c,d))
        
        if best_swap and best_delta < 0:
            (a,b), (c,d) = best_swap
            A[a,b] = 0
            A[b,a] = 0
            current_bits[a] &= ~(1 << b)
            current_bits[b] &= ~(1 << a)
            A[c,d] = 1
            A[d,c] = 1
            current_bits[c] |= (1 << d)
            current_bits[d] |= (1 << c)
            
            new_c4s = count_c4s_v2(A)
            if new_c4s < best_c4s:
                best_c4s = new_c4s
                best_A = A.copy()
            improved = True
    
    # Final polish: try single edge additions/removals if we're at 91 edges
    # (shouldn't be needed but just in case)
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i,j]:
                final_edges.append([i, j])
    
    return final_edges
