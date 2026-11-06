import time
import random
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    q = 5
    n = q**2 + q + 1  # 31
    
    # 1. Generate points of PG(2, q)
    # Points are triples (x, y, z) in GF(q)^3, not all zero, up to scaling.
    # We can fix z=1 for affine points, and z=0 for points at infinity.
    points = []
    # Affine points: (x, y, 1) for x, y in GF(q)
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # Points at infinity: (x, 1, 0) for x in GF(q), and (1, 0, 0)
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    # Verify count
    if len(points) != n:
        return [[i, j] for i in range(n) for j in range(i+1, n)][:91]
        
    # 2. Construct the orthogonal polarity graph
    # Adjacency: dot product is 0 mod q
    def dot(p1, p2):
        return (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
        
    adj = [[0]*n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append((i, j))
                
    # The polarity graph should have 90 edges
    if len(edges) != 90:
        # Fallback to random if construction fails
        # But it should be 90
        pass

    # 3. Add the 91st edge
    # We want to add the edge (u, v) that minimizes the increase in C4 count.
    # Increase is C(|N(u) ∩ N(v)|, 2)
    
    def count_c4_increase(adj, u, v):
        # Count common neighbors
        common = 0
        for k in range(n):
            if adj[u][k] == 1 and adj[v][k] == 1:
                common += 1
        return common * (common - 1) // 2

    best_edge = None
    min_increase = float('inf')
    
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 0:
                inc = count_c4_increase(adj, i, j)
                if inc < min_increase:
                    min_increase = inc
                    best_edge = (i, j)
                    
    if best_edge:
        u, v = best_edge
        adj[u][v] = 1
        adj[v][u] = 1
        edges.append((u, v))
        
    # Initial C4 count
    def count_c4(adj):
        c4_count = 0
        for i in range(n):
            for j in range(i+1, n):
                common = 0
                for k in range(n):
                    if adj[i][k] == 1 and adj[j][k] == 1:
                        common += 1
                c4_count += common * (common - 1) // 2
        return c4_count
        
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = [row[:] for row in adj]
    
    # 4. Local Search
    # Swap an edge (u,v) with a non-edge (x,y) if it reduces C4 count.
    # Delta_remove(u,v) = C(common(u,v), 2)
    # Delta_add(x,y) = C(common(x,y), 2)
    # Note: common(u,v) in Delta_remove is calculated in the graph WITHOUT (u,v).
    # But since u and v are not neighbors of each other in the "common neighbor" calculation 
    # (we look for k such that adj[u][k] and adj[v][k]), and adj[u][v] is 1, 
    # does it affect common neighbors?
    # No, common neighbors are k != u, k != v.
    # So common(u,v) is the same whether (u,v) is present or not, 
    # EXCEPT that if we remove (u,v), u is no longer in N(v) and v is no longer in N(u).
    # But we are counting k such that k is neighbor of u AND k is neighbor of v.
    # If k=u, adj[v][u] is 1, but adj[u][u] is 0. So u is never a common neighbor.
    # So the set of common neighbors is independent of the edge (u,v) itself.
    
    def get_common_count(adj, u, v):
        count = 0
        for k in range(n):
            if adj[u][k] and adj[v][k]:
                count += 1
        return count
        
    # Optimization: Precompute common neighbor counts?
    # n=31, so O(n^3) is 31^3 ~ 30k ops. Very fast.
    # We can just recompute or maintain a matrix.
    
    # Maintain common counts matrix
    common_counts = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            common_counts[i][j] = get_common_count(adj, i, j)
            common_counts[j][i] = common_counts[i][j]
            
    def update_common_counts(adj, common_counts, u, v, added):
        # If added, for all k, if adj[u][k] and adj[v][k]... no.
        # Adding (u,v) changes N(u) and N(v).
        # It adds v to N(u) and u to N(v).
        # This affects common_counts[i][j] if i or j is u or v.
        # Specifically, common(u, k) increases by 1 if v is in N(k) (and v != k).
        # i.e., if adj[v][k] is 1.
        # Similarly common(v, k) increases by 1 if adj[u][k] is 1.
        # And common(i, j) for i,j != u,v is unchanged.
        
        if added:
            for k in range(n):
                if k == u or k == v: continue
                if adj[v][k]: # v is neighbor of k
                    common_counts[u][k] += 1
                    common_counts[k][u] += 1
                if adj[u][k]: # u is neighbor of k
                    common_counts[v][k] += 1
                    common_counts[k][v] += 1
        else:
            for k in range(n):
                if k == u or k == v: continue
                if adj[v][k]:
                    common_counts[u][k] -= 1
                    common_counts[k][u] -= 1
                if adj[u][k]:
                    common_counts[v][k] -= 1
                    common_counts[k][v] -= 1

    # Local search loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order to avoid local minima
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        
        for idx in edge_indices:
            u, v = edges[idx]
            # Calculate delta remove
            # common(u,v) is common_counts[u][v]
            c_uv = common_counts[u][v]
            delta_remove = c_uv * (c_uv - 1) // 2
            
            # Try to find a non-edge to add
            # We want min delta_add.
            # delta_add(x,y) = C(common_counts[x][y], 2)
            # We can just iterate all non-edges
            best_x, best_y = -1, -1
            min_delta_add = float('inf')
            
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j] == 0:
                        c_ij = common_counts[i][j]
                        delta_add = c_ij * (c_ij - 1) // 2
                        if delta_add < min_delta_add:
                            min_delta_add = delta_add
                            best_x, best_y = i, j
                            
            if min_delta_add < delta_remove:
                # Perform swap
                # Remove (u,v)
                adj[u][v] = 0
                adj[v][u] = 0
                update_common_counts(adj, common_counts, u, v, False)
                edges[idx] = (best_x, best_y)
                
                # Add (best_x, best_y)
                adj[best_x][best_y] = 1
                adj[best_y][best_x] = 1
                update_common_counts(adj, common_counts, best_x, best_y, True)
                
                current_c4 -= delta_remove
                current_c4 += min_delta_add
                improved = True
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj = [row[:] for row in adj]
                    best_edges = [tuple(e) for e in edges]
                    
    # Return best edges found
    # Ensure we return a list of lists
    # If best_adj is the polarity graph + 1 edge, best_edges should correspond to it.
    # But if we did swaps, edges list is updated.
    # We need to return the edges corresponding to best_adj.
    # It's safer to reconstruct edges from best_adj.
    
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                final_edges.append([i, j])
                
    # Check if we have exactly 91 edges
    if len(final_edges) != 91:
        # Fallback
        return [[i, j] for i in range(n) for j in range(i+1, n)][:91]
        
    return final_edges
