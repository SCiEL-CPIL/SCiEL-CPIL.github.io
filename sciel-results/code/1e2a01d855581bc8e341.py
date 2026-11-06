import numpy as np
import time
import random

def construct():
    # 1. Generate all points in PG(2, 5)
    # Points are equivalence classes of non-zero vectors in GF(5)^3
    points = []
    seen = set()
    inv_map = {1: 1, 2: 3, 3: 2, 4: 4}
    
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                
                vec = (x, y, z)
                # Find first non-zero
                if vec[0] != 0:
                    c = inv_map[vec[0]]
                elif vec[1] != 0:
                    c = inv_map[vec[1]]
                else:
                    c = inv_map[vec[2]]
                
                canon = ((x * c) % 5, (y * c) % 5, (z * c) % 5)
                if canon not in seen:
                    seen.add(canon)
                    points.append(canon)
    
    n = len(points)  # Should be 31
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # 2. Construct the polarity graph (90 edges)
    # Edge if dot product is 0
    adj = np.zeros((n, n), dtype=np.uint8)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if dot == 0:
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append([i, j])
    
    # 3. Find the non-edge that adds the fewest C4s
    # Number of C4s added by (u,v) is number of edges between N(u) and N(v)
    # We want to minimize this.
    
    # Precompute neighbor sets for speed
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                neighbors[i].add(j)
    
    def count_c4s_added(u, v):
        # u, v are not adjacent
        # Count edges (a,b) where a in N(u), b in N(v), a != b
        # Since the graph is undirected, if a in N(u) and b in N(v) and adj[a][b] is 1,
        # then (u,a,b,v) is a C4.
        # We just need to count pairs (a,b) such that a in N(u), b in N(v), and adj[a][b] == 1.
        # Note: if a==b, it's not a C4 (it's a triangle u-a-v plus u-v, but u-v is not an edge).
        # Actually, if a==b, then a is a common neighbor. u-a-v is a path of length 2.
        # For a C4 u-a-b-v, we need a != b.
        # So we count pairs (a,b) with a in N(u), b in N(v), a != b, and adj[a][b] == 1.
        
        count = 0
        # Iterate over the smaller set? Or just nested loops.
        # n=31, so max 30*30 = 900 operations. Very fast.
        for a in neighbors[u]:
            for b in neighbors[v]:
                if a != b and adj[a][b]:
                    count += 1
        return count

    min_c4s = float('inf')
    best_edge = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                c4_count = count_c4s_added(i, j)
                if c4_count < min_c4s:
                    min_c4s = c4_count
                    best_edge = [i, j]
    
    # Add the best edge
    if best_edge is not None:
        edges.append(best_edge)
        adj[best_edge[0]][best_edge[1]] = 1
        adj[best_edge[1]][best_edge[0]] = 1
        neighbors[best_edge[0]].add(best_edge[1])
        neighbors[best_edge[1]].add(best_edge[0])
    
    # 4. Local Search to minimize C4s
    # We will perform a random walk in the space of graphs with 91 edges.
    # Move: remove an edge (u,v) and add a non-edge (x,y).
    # Accept if the number of C4s decreases, or with probability exp(-delta/T) (Simulated Annealing).
    
    def count_total_c4s():
        # Count C4s in the current graph.
        # A C4 is a set of 4 vertices {a,b,c,d} such that edges (a,b), (b,c), (c,d), (d,a) exist.
        # Alternatively, count pairs of edges (a,b) and (c,d) that share no vertices and form a cycle?
        # Simpler: For each pair of vertices (u,v), let common_neighbors = |N(u) ∩ N(v)|.
        # The number of C4s is (1/2) * sum_{u < v} (common_neighbors choose 2)?
        # No, that counts C4s where u and v are opposite vertices.
        # Let's use the formula: #C4 = (1/2) * sum_{i} (d_i^2 - 1) ... no.
        # #C4 = (1/2) * (sum_{i,j} (A^2)_{ij}^2 - ... )
        # Let's just count directly for n=31.
        # Iterate over all pairs of non-adjacent vertices (u,v).
        # For each pair, count common neighbors. If common neighbors are a,b, then u-a-v-b-u is a C4?
        # No, u-a-v-b-u is a C4 if a,b are distinct common neighbors of u and v, and a-b is NOT an edge?
        # Actually, any two common neighbors a,b of u and v form a C4 u-a-v-b-u.
        # Wait, if a and b are adjacent, it's still a C4 subgraph?
        # A C4 is a cycle of length 4. The vertices are u,a,v,b. Edges (u,a), (a,v), (v,b), (b,u).
        # It doesn't matter if (a,b) is an edge or not. The subgraph induced by {u,a,v,b} contains the cycle.
        # So, for each pair of vertices (u,v), the number of C4s where u and v are opposite is C(common_neighbors(u,v), 2).
        # Each C4 has two pairs of opposite vertices. So total C4s = (1/2) * sum_{u < v} C(common(u,v), 2).
        
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                # Count common neighbors
                common = 0
                # Use bitsets for speed? n=31, so uint64 is enough.
                # Let's just loop.
                for w in range(n):
                    if adj[u][w] and adj[v][w]:
                        common += 1
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2

    current_c4s = count_total_c4s()
    best_c4s = current_c4s
    best_adj = adj.copy()
    best_edges = [list(e) for e in edges]
    
    start_time = time.time()
    timeout = 110  # Leave 10s buffer
    
    T = 1.0
    cooling = 0.999
    
    # Precompute list of edges and non-edges for faster sampling?
    # Or just sample randomly.
    
    while time.time() - start_time < timeout:
        # Pick a random edge to remove
        # We need to maintain a list of edges.
        # Current edges are in 'edges' list.
        # Let's pick a random index from edges.
        if len(edges) == 0:
            break
        
        idx_remove = random.randint(0, len(edges) - 1)
        u, v = edges[idx_remove]
        
        # Remove edge (u,v)
        adj[u][v] = 0
        adj[v][u] = 0
        neighbors[u].remove(v)
        neighbors[v].remove(u)
        edges.pop(idx_remove)
        
        # Calculate C4s after removal
        c4s_after_remove = count_total_c4s()
        
        # Pick a random non-edge to add
        # Find a non-edge.
        # We can just pick random pairs until we find one.
        found = False
        while not found:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x != y and not adj[x][y]:
                found = True
        
        # Add edge (x,y)
        adj[x][y] = 1
        adj[y][x] = 1
        neighbors[x].add(y)
        neighbors[y].add(x)
        edges.append([x, y])
        
        c4s_after_add = count_total_c4s()
        
        delta = c4s_after_add - current_c4s
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_c4s = c4s_after_add
            if current_c4s < best_c4s:
                best_c4s = current_c4s
                best_adj = adj.copy()
                best_edges = [list(e) for e in edges]
        else:
            # Revert
            # Remove (x,y)
            adj[x][y] = 0
            adj[y][x] = 0
            neighbors[x].remove(y)
            neighbors[y].remove(x)
            edges.pop()
            
            # Re-add (u,v)
            adj[u][v] = 1
            adj[v][u] = 1
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v])
            
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    # Return the best edges found
    return best_edges
