import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    q = 5
    # Generate points in Projective Plane over F_q
    # Points are (x, y, z) != (0,0,0), normalized so first non-zero is 1
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: divide by first non-zero
                if x != 0:
                    points.append((x, y, z))
                elif y != 0:
                    points.append((0, y, z))
                elif z != 0:
                    points.append((0, 0, z))
    
    # There should be q^2 + q + 1 = 31 points
    # Map points to indices 0..30
    point_index = {p: i for i, p in enumerate(points)}
    n = len(points)
    
    # Define the bilinear form B(u, v) = u0*v2 + u2*v0 + 2*u1*v1
    # u ~ v if B(u, v) == 0 and u != v
    adj = np.zeros((n, n), dtype=int)
    edges = []
    
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            # Calculate B(u, v)
            # u0*v2 + u2*v0 + 2*u1*v1
            val = (u[0] * v[2] + u[2] * v[0] + 2 * u[1] * v[1]) % q
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append([i, j])
    
    # We have 93 edges. We need 91.
    # Count 4-cycles
    # A 4-cycle is (a, b, c, d) with a<b<c<d? No, just distinct vertices.
    # To avoid duplicates, we can identify a cycle by the set of its 4 edges.
    # Or simply, iterate over pairs of edges sharing a vertex?
    
    # Efficient way to find 4-cycles:
    # For each pair of vertices (u, v) that are NOT connected, count common neighbors.
    # If count > 1, then there are C(count, 2) 4-cycles?
    # No, a 4-cycle is u-w-v-x-u.
    # This requires w, x to be common neighbors of u and v?
    # No, w is neighbor of u and v. x is neighbor of u and v.
    # And (w, x) must be an edge.
    # So for each non-edge (u, v), let C = {w | adj[u,w] and adj[v,w]}.
    # The number of 4-cycles containing the "diagonal" (u,v) is the number of edges within C.
    # Wait, the cycle is u-w-v-x-u.
    # Edges: (u,w), (w,v), (v,x), (x,u).
    # This uses edges (u,w), (w,v), (v,x), (x,u).
    # This cycle does NOT contain the edge (u,v).
    # So counting cycles by diagonals (non-edges) is a valid way to enumerate all 4-cycles.
    # Each 4-cycle has 2 diagonals (non-edges in the cycle, but might be edges in the graph).
    # Wait, if (u,v) is an edge, it's not a diagonal.
    # In a 4-cycle u-w-v-x-u, the pairs (u,v) and (w,x) are the diagonals.
    # If (u,v) is an edge in the graph, then the cycle is u-w-v-x-u.
    # The edges are (u,w), (w,v), (v,x), (x,u).
    # This is a valid 4-cycle.
    # So we should count for ALL pairs (u, v), whether edge or not.
    # For each pair (u, v), let C = common neighbors.
    # Number of 4-cycles where (u,v) is a "diagonal" is the number of edges in the subgraph induced by C?
    # No.
    # Cycle: u-w-v-x-u.
    # w, x are in C.
    # We need (w,x) to be an edge.
    # So for each pair (u,v), the number of such cycles is the number of edges between vertices in C?
    # No, w and x are distinct vertices in C.
    # So we need to count the number of edges (w,x) where w,x in C.
    # Let E_C be the set of edges in the graph induced by C.
    # Then the number of 4-cycles with diagonal (u,v) is |E_C|.
    #
    # Total 4-cycles = Sum over all pairs (u,v) of |E_{C_{uv}}| / 2 ?
    # Each 4-cycle has 2 diagonals. So each cycle is counted twice.
    
    cycles = [] # List of sets of 4 edges
    # To avoid duplicates, we can just store the sorted tuple of the 4 edge indices.
    
    # Precompute common neighbors for all pairs
    # adj is n x n
    # common_neighbors[u][v] = list of w such that adj[u,w] and adj[v,w]
    
    # This might be memory heavy if stored as lists, but n=31 is small.
    # n=31, pairs ~ 500.
    
    all_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            all_pairs.append((u, v))
            
    # For each pair, find common neighbors
    # And count edges among them
    
    # We need to map edges to indices
    edge_to_idx = {}
    for idx, e in enumerate(edges):
        edge_to_idx[tuple(sorted(e))] = idx
        
    # List of all 4-cycles
    # We will store them as frozensets of edge indices
    
    cycle_edges = []
    
    for u, v in all_pairs:
        # Find common neighbors
        # adj[u] is a row
        common = []
        for w in range(n):
            if w != u and w != v:
                if adj[u, w] == 1 and adj[v, w] == 1:
                    common.append(w)
                    
        # Count edges among common
        # For each pair (w1, w2) in common, check if edge exists
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                w1 = common[i]
                w2 = common[j]
                if adj[w1, w2] == 1:
                    # Cycle: u-w1-v-w2-u
                    # Edges: (u,w1), (w1,v), (v,w2), (w2,u)
                    e1 = tuple(sorted((u, w1)))
                    e2 = tuple(sorted((w1, v)))
                    e3 = tuple(sorted((v, w2)))
                    e4 = tuple(sorted((w2, u)))
                    
                    # Get indices
                    idx1 = edge_to_idx.get(e1)
                    idx2 = edge_to_idx.get(e2)
                    idx3 = edge_to_idx.get(e3)
                    idx4 = edge_to_idx.get(e4)
                    
                    if idx1 is not None and idx2 is not None and idx3 is not None and idx4 is not None:
                        cycle_set = frozenset([idx1, idx2, idx3, idx4])
                        cycle_edges.append(cycle_set)
                        
    # Deduplicate cycles
    # A cycle is counted twice (once for each diagonal)
    unique_cycles = set(cycle_edges)
    
    # Now we have the set of all 4-cycles in G_93.
    # We want to remove 2 edges to maximize the number of cycles removed.
    
    # Map each edge to the list of cycles it belongs to
    edge_cycles = {i: [] for i in range(len(edges))}
    for c in unique_cycles:
        for e_idx in c:
            edge_cycles[e_idx].append(c)
            
    # If no cycles, we are done (score 0, but impossible)
    # We want to pick e1, e2 to maximize |S_e1 U S_e2|
    
    best_removed = 0
    best_e1 = -1
    best_e2 = -1
    
    num_edges = len(edges)
    
    # Iterate over pairs of edges
    # Optimization: Only check edges that are in at least one cycle
    active_edges = [e for e in range(num_edges) if len(edge_cycles[e]) > 0]
    
    for i in range(len(active_edges)):
        e1 = active_edges[i]
        s1 = set(edge_cycles[e1])
        for j in range(i + 1, len(active_edges)):
            e2 = active_edges[j]
            s2 = set(edge_cycles[e2])
            union_size = len(s1.union(s2))
            if union_size > best_removed:
                best_removed = union_size
                best_e1 = e1
                best_e2 = e2
                
    # Remove best_e1 and best_e2
    if best_e1 != -1 and best_e2 != -1:
        edges_to_remove = {best_e1, best_e2}
        final_edges = [e for idx, e in enumerate(edges) if idx not in edges_to_remove]
    else:
        # If no cycles found, just remove any 2 edges
        # This case shouldn't happen if ex(31,C4)=90
        final_edges = edges[:91]
        
    return final_edges

# EVOLVE-BLOCK-END
