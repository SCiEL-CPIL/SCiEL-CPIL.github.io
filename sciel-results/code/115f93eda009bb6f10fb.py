import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    q = 5
    # Build PG(2, q) points: 31 points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # should be 31
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph
    edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edge_set.add((i, j))
    
    # Build adjacency using numpy bitsets for fast operations
    adj_bits = np.zeros((n, n), dtype=np.uint8)
    for (i, j) in edge_set:
        adj_bits[i, j] = 1
        adj_bits[j, i] = 1
    
    edges = [[i, j] for (i, j) in edge_set]
    
    # Count C4s in the base graph
    def count_c4s(adj_b):
        """Count 4-cycles using matrix operations."""
        # For each pair (i,j) that is an edge, count common neighbors
        # Each C4 is counted twice (once for each diagonal pair)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj_b[i, j]:
                    # common neighbors of i and j
                    common = np.sum(adj_b[i] & adj_b[j])
                    count += common
        # Each C4 is counted twice (for each of its two diagonals)
        return count // 2
    
    base_c4s = count_c4s(adj_bits)
    
    # Find non-edges
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj_bits[i, j]:
                non_edges.append((i, j))
    
    # For each non-edge, compute the cost of adding it (new C4s created)
    # New C4s from adding edge (i,j): number of paths i-a-b-j where a,b are distinct
    # i.e., a in N(i), b in N(j), a~b, a!=b
    best_edge = None
    best_cost = float('inf')
    best_total = base_c4s + float('inf')
    
    for (i, j) in non_edges:
        if time.time() > deadline - 10:
            break
        # Count new C4s: pairs (a,b) with a~i, b~j, a~b, a!=b
        # a ranges over neighbors of i, b ranges over neighbors of j
        neigh_i = np.where(adj_bits[i] > 0)[0]
        neigh_j = np.where(adj_bits[j] > 0)[0]
        
        cost = 0
        for a in neigh_i:
            if a == j:
                continue
            for b in neigh_j:
                if a == b:
                    continue
                if adj_bits[a, b]:
                    cost += 1
        
        total = base_c4s + cost
        if total < best_total:
            best_total = total
            best_cost = cost
            best_edge = (i, j)
    
    # Add the best edge
    if best_edge is not None:
        edges.append(list(best_edge))
        i, j = best_edge
        adj_bits[i, j] = 1
        adj_bits[j, i] = 1
    
    # Ensure exactly 91 edges
    if len(edges) < 91:
        existing = set((min(e[0],e[1]), max(e[0],e[1])) for e in edges)
        for i in range(n):
            for j in range(i+1, n):
                if len(edges) >= 91:
                    break
                if (i,j) not in existing:
                    edges.append([i,j])
                    existing.add((i,j))
                    adj_bits[i,j] = 1
                    adj_bits[j,i] = 1
            if len(edges) >= 91:
                break
    elif len(edges) > 91:
        edges = edges[:91]
    
    # Local search: try swapping one edge for another to reduce C4 count
    # Move: remove edge e1, add edge e2, keep total at 91
    # Only try if we have time
    if time.time() < deadline - 5:
        current_c4s = count_c4s(adj_bits)
        best_c4s = current_c4s
        best_move = None
        
        # Precompute list of edges and non-edges
        edge_list = [(min(e[0],e[1]), max(e[0],e[1])) for e in edges]
        edge_set_cur = set(edge_list)
        
        non_edge_list = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj_bits[i, j]:
                    non_edge_list.append((i, j))
        
        # For each edge, for each non-edge, try the swap
        # This is O(E * NE) which is about 91 * 26 = 2366 swaps
        # Each swap requires recomputing C4 count or incremental update
        
        # Use incremental C4 counting:
        # Removing edge (u,v) destroys C4s that use (u,v)
        # Adding edge (a,b) creates C4s that use (a,b)
        
        def c4s_using_edge(adj_b, u, v):
            """Count C4s that use edge (u,v)."""
            # C4s using edge (u,v): u-x-y-v-u where x~u, y~v, x~y, x!=y, x!=v, y!=u
            count = 0
            for x in range(n):
                if x == u or x == v:
                    continue
                if not adj_b[u, x]:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    if not adj_b[v, y]:
                        continue
                    if adj_b[x, y]:
                        count += 1
            # Each C4 using edge (u,v) is counted once here
            return count
        
        for ei, (u, v) in enumerate(edge_list):
            if time.time() > deadline - 5:
                break
            for (a, b) in non_edge_list:
                if (a, b) == (u, v) or (a, b) == (v, u):
                    continue
                # Skip if a or b is u or v (would create multi-edge or self-loop issues)
                # Actually it's fine, just different edges
                
                # Estimate: removing (u,v) destroys c4s_using_edge(u,v) C4s
                # Adding (a,b) creates new C4s
                # But interactions: some C4s might use both edges, etc.
                # For accuracy, just compute delta approximately
                
                # Quick heuristic: skip if the new edge has many common neighbors
                common_ab = np.sum(adj_bits[a] & adj_bits[b])
                if common_ab > 15:
                    continue
                
                # Try the swap
                adj_bits[u, v] = 0
                adj_bits[v, u] = 0
                adj_bits[a, b] = 1
                adj_bits[b, a] = 1
                
                new_c4s = count_c4s(adj_bits)
                
                # Undo
                adj_bits[u, v] = 1
                adj_bits[v, u] = 1
                adj_bits[a, b] = 0
                adj_bits[b, a] = 0
                
                if new_c4s < best_c4s:
                    best_c4s = new_c4s
                    best_move = (ei, (a, b))
        
        # Apply the best move found
        if best_move is not None:
            ei, (a, b) = best_move
            u, v = edge_list[ei]
            # Remove edge ei, add edge (a,b)
            adj_bits[u, v] = 0
            adj_bits[v, u] = 0
            adj_bits[a, b] = 1
            adj_bits[b, a] = 1
            edges[ei] = [a, b]
    
    return edges[:91]
