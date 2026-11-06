import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    n = 31
    target_edges = 91
    
    # PG(2,5) polarity: 31 vertices, 90 edges, C4-free
    # Vertices are projective classes of 3-dim vectors over GF(5)
    # Polarity: x·y = 0 in a symmetric bilinear form
    # Use the standard polarity of PG(2,5)
    
    # Generate all 31 points of PG(2,5)
    # Points are equivalence classes of nonzero vectors in GF(5)^3
    # Two vectors are equivalent if one is a scalar multiple of the other
    
    def gf5_mul(a, b):
        return (a * b) % 5
    
    # Generate all points of PG(2,5)
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Canonical form: first nonzero coordinate = 1
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 3, 5)  # inverse in GF(5): x^3 = x^-1
                        vec = [gf5_mul(v, inv) for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Polarity: x·y = x0*y0 + x1*y1 + x2*y2 = 0 (mod 5)
    # This gives a self-polar polarity with 90 edges
    def is_polar(u, v):
        """Check if points u and v are polar (orthogonal under the polarity)"""
        s = 0
        for i in range(3):
            s = (s + points[u][i] * points[v][i]) % 5
        return s == 0
    
    # Build the polarity graph (C4-free, 90 edges)
    polarity_edges = []
    for i in range(31):
        for j in range(i + 1, 31):
            if is_polar(i, j):
                polarity_edges.append((i, j))
    
    assert len(polarity_edges) == 90, f"Expected 90 polarity edges, got {len(polarity_edges)}"
    
    # Count 4-cycles in a graph
    def count_c4(adj_matrix, edge_list):
        """Count 4-cycles using adjacency matrix"""
        c4_count = 0
        n = adj_matrix.shape[0]
        # For each pair of vertices (u,v), count common neighbors
        # Each 4-cycle is counted 4 times (once per starting vertex)
        # Actually: sum over all pairs (u,v) of C(common_neighbors, 2)
        # Each 4-cycle u-v-w-x-u is counted once for pair (u,w) and once for (v,x)
        # So total = sum * 2 / 4 = sum / 2? Let me think more carefully.
        
        # A 4-cycle is a set {a,b,c,d} with edges ab, bc, cd, da
        # For a fixed pair (a,c), if they share 2+ common neighbors, each pair of common neighbors forms a 4-cycle
        # Each 4-cycle has exactly 2 "diagonal" pairs: (a,c) and (b,d)
        # So: total 4-cycles = (1/2) * sum over all pairs (u,v) of C(common(u,v), 2)
        
        for u in range(n):
            for v in range(u + 1, n):
                # Count common neighbors
                common = np.sum(adj_matrix[u] & adj_matrix[v])
                if common >= 2:
                    c4_count += common * (common - 1) // 2
        
        return c4_count // 2
    
    # Better: use a faster C4 count
    def count_c4_fast(adj_matrix):
        """Count 4-cycles: (1/2) * sum_{u<v} C(deg_common(u,v), 2)"""
        n = adj_matrix.shape[0]
        c4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = int(np.sum(adj_matrix[u] & adj_matrix[v]))
                if common >= 2:
                    c4_count += common * (common - 1) // 2
        return c4_count // 2
    
    # Build adjacency matrix for polarity graph
    adj = np.zeros((31, 31), dtype=np.uint8)
    for u, v in polarity_edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # The polarity graph should be C4-free
    # Verify: count_c4_fast(adj) should be 0
    
    # Now find the best 91st edge to add
    # For each non-edge (u,v), count how many 4-cycles it creates
    # A new edge (u,v) creates a 4-cycle for each pair of common neighbors (w,x) where w~x is NOT an edge... 
    # Actually: edge (u,v) creates a 4-cycle u-w-x-v-u if w~x exists
    # So new 4-cycles = number of edges among common neighbors of u and v
    
    def new_c4s_from_edge(adj, u, v):
        """Count new 4-cycles created by adding edge (u,v)"""
        common = np.where(adj[u] & adj[v])[0]
        count = 0
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                if adj[common[i], common[j]]:
                    count += 1
        return count
    
    # Find the best edge to add
    best_edge = None
    best_c4s = float('inf')
    
    existing_edges = set()
    for u, v in polarity_edges:
        existing_edges.add((min(u,v), max(u,v)))
    
    for u in range(31):
        for v in range(u + 1, 31):
            if (u, v) in existing_edges:
                continue
            c4s = new_c4s_from_edge(adj, u, v)
            if c4s < best_c4s:
                best_c4s = c4s
                best_edge = (u, v)
    
    # Add the best edge
    base_edges = list(polarity_edges)
    if best_edge:
        base_edges.append(best_edge)
    
    # Now try local search: swap edges to reduce C4 count
    def get_edge_set(edges):
        return set((min(u,v), max(u,v)) for u, v in edges)
    
    def build_adj(edges):
        a = np.zeros((31, 31), dtype=np.uint8)
        for u, v in edges:
            a[u, v] = 1
            a[v, u] = 1
        return a
    
    current_edges = base_edges[:]
    current_edge_set = get_edge_set(current_edges)
    current_adj = build_adj(current_edges)
    current_c4 = count_c4_fast(current_adj)
    
    best_edges = current_edges[:]
    best_c4 = current_c4
    
    # Local search: try swapping an edge with a non-edge
    random.seed(42)
    
    def try_swaps(edges, edge_set, adj, c4_count, time_limit):
        """Try random edge swaps to reduce C4 count"""
        best_local_edges = edges[:]
        best_local_set = edge_set.copy()
        best_local_adj = adj.copy()
        best_local_c4 = c4_count
        
        while time.time() < time_limit:
            # Pick a random edge to remove
            edge_idx = random.randint(0, len(best_local_edges) - 1)
            remove_u, remove_v = best_local_edges[edge_idx]
            
            # Find a random non-edge to add
            attempts = 0
            while attempts < 100:
                add_u = random.randint(0, 30)
                add_v = random.randint(0, 30)
                if add_u == add_v:
                    attempts += 1
                    continue
                key = (min(add_u, add_v), max(add_u, add_v))
                if key not in best_local_set:
                    break
                attempts += 1
            else:
                continue
            
            # Compute new C4 count
            new_adj = best_local_adj.copy()
            new_adj[remove_u, remove_v] = 0
            new_adj[remove_v, remove_u] = 0
            new_adj[add_u, add_v] = 1
            new_adj[add_v, add_u] = 1
            
            new_c4 = count_c4_fast(new_adj)
            
            if new_c4 < best_local_c4:
                # Accept the swap
                best_local_edges[edge_idx] = (add_u, add_v)
                best_local_set.discard((min(remove_u, remove_v), max(remove_u, remove_v)))
                best_local_set.add(key)
                best_local_adj = new_adj
                best_local_c4 = new_c4
                best_local_edges = best_local_edges[:]
            
            if time.time() > time_limit:
                break
        
        return best_local_edges, best_local_set, best_local_adj, best_local_c4
    
    # Run local search with time budget
    if time.time() < deadline - 20:
        current_edges, current_edge_set, current_adj, current_c4 = try_swaps(
            current_edges, current_edge_set, current_adj, current_c4, deadline - 20
        )
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_edges = current_edges[:]
    
    # Try multiple restarts with different random seeds if time allows
    if time.time() < deadline - 30:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 20:
                break
            random.seed(seed)
            # Start from the polarity base + best edge
            restart_edges = list(polarity_edges) + [best_edge]
            restart_set = get_edge_set(restart_edges)
            restart_adj = build_adj(restart_edges)
            restart_c4 = count_c4_fast(restart_adj)
            
            restart_edges, restart_set, restart_adj, restart_c4 = try_swaps(
                restart_edges, restart_set, restart_adj, restart_c4, deadline - 20
            )
            
            if restart_c4 < best_c4:
                best_c4 = restart_c4
                best_edges = restart_edges[:]
    
    # Final verification
    final_edge_set = get_edge_set(best_edges)
    assert len(final_edge_set) == 91, f"Expected 91 edges, got {len(final_edge_set)}"
    
    # Convert to required format
    result = [[u, v] for u, v in sorted(final_edge_set)]
    
    # Verify
    assert len(result) == 91
    assert all(0 <= u < 31 and 0 <= v < 31 for u, v in result)
    assert all(u < v for u, v in result)
    
    return result
