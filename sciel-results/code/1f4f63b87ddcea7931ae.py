"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _count_c4(adj_mat):
    """Count the number of 4-cycles in the graph represented by adj_mat."""
    n = adj_mat.shape[0]
    c4_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            common = np.sum(adj_mat[u, :] & adj_mat[v, :])
            if common >= 2:
                c4_count += common * (common - 1) // 2
    return c4_count // 2

def _build_polarity_graph():
    """Build the C4-free polarity graph of PG(2,5) with 31 vertices and 90 edges."""
    # Points in PG(2,5) as normalized vectors in GF(5)^3
    points = []
    for x in range(6):
        for y in range(6):
            for z in range(6):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    points.append((1, (y * inv_x) % 5, (z * inv_x) % 5))
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    points.append((0, 1, (z * inv_y) % 5))
                else:
                    points.append((0, 0, 1))
    
    n = len(points)
    assert n == 31
    
    def dot(p, q):
        return (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                edges.append((i, j))
    
    assert len(edges) == 90, f"Expected 90 edges, got {len(edges)}"
    return edges

def construct():
    """
    Construct a graph with 31 vertices and 91 edges minimizing the number of 4-cycles.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Step 1: Start with the C4-free polarity graph (90 edges)
    base_edges = _build_polarity_graph()
    
    adj = np.zeros((31, 31), dtype=np.int8)
    for u, v in base_edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Step 2: Find the non-edge that creates the fewest new 4-cycles
    best_edge = None
    best_common = float('inf')
    
    for u in range(31):
        for v in range(u + 1, 31):
            if adj[u, v] == 0:
                common = np.sum(adj[u, :] & adj[v, :])
                if common < best_common:
                    best_common = common
                    best_edge = (u, v)
    
    # Add the best edge
    adj[best_edge[0], best_edge[1]] = 1
    adj[best_edge[1], best_edge[0]] = 1
    
    current_c4 = _count_c4(adj)
    current_edges = set(base_edges)
    current_edges.add(best_edge)
    
    # Step 3: Local search to improve if time permits
    if time.time() < deadline - 20:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_swap = None
            best_swap_c4 = current_c4
            
            # Try removing each edge and adding a non-edge
            for u, v in list(current_edges):
                if time.time() >= deadline - 5:
                    break
                
                # Temporarily remove edge (u,v)
                adj[u, v] = 0
                adj[v, u] = 0
                
                # Find best non-edge to add
                for a in range(31):
                    for b in range(a + 1, 31):
                        if adj[a, b] == 0:
                            # Compute new C4 count if we add (a,b)
                            adj[a, b] = 1
                            adj[b, a] = 1
                            new_c4 = _count_c4(adj)
                            adj[a, b] = 0
                            adj[b, a] = 0
                            
                            if new_c4 < best_swap_c4:
                                best_swap_c4 = new_c4
                                best_swap = ((u, v), (a, b))
                
                # Restore edge (u,v)
                adj[u, v] = 1
                adj[v, u] = 1
                
                if best_swap and best_swap_c4 < current_c4:
                    break
            
            if best_swap and best_swap_c4 < current_c4:
                # Perform the swap
                remove_edge, add_edge = best_swap
                current_edges.remove(remove_edge)
                current_edges.add(add_edge)
                adj[remove_edge[0], remove_edge[1]] = 0
                adj[remove_edge[1], remove_edge[0]] = 0
                adj[add_edge[0], add_edge[1]] = 1
                adj[add_edge[1], add_edge[0]] = 1
                current_c4 = best_swap_c4
                improved = True
    
    # Return the final list of edges
    return list(current_edges)
# EVOLVE-BLOCK-END
