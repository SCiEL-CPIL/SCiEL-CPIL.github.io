"""Program to construct a 4-colouring of K_n with no monochromatic C6."""
import time
import itertools

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a 4-colouring of K_n with no monochromatic 6-cycle.
    Returns {'n': n, 'colors': [...]} where colors has one entry per pair (i,j), i<j,
    in lexicographic order, each in {0,1,2,3}.
    """
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    n = 17
    num_edges = n * (n - 1) // 2
    
    # Precompute edge list and index mapping
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_list.append((i, j))
            
    # Map (u, v) to index
    edge_idx_map = {}
    for idx, (u, v) in enumerate(edge_list):
        edge_idx_map[(u, v)] = idx
        edge_idx_map[(v, u)] = idx

    # Symmetry breaking: fix the first 3 edges to color 0
    # This reduces the search space significantly
    fixed_edges = {
        (0, 1): 0,
        (0, 2): 0,
        (0, 3): 0
    }
    
    # Initialize colors array
    colors = [0] * num_edges
    for (u, v), c in fixed_edges.items():
        idx = edge_idx_map[(u, v)]
        colors[idx] = c

    # Bitmask adjacency for each color
    # adj_mask[c][u] is a bitmask of vertices connected to u with color c
    adj_mask = [[0] * n for _ in range(4)]
    
    # Initialize with fixed edges
    for (u, v), c in fixed_edges.items():
        adj_mask[c][u] |= (1 << v)
        adj_mask[c][v] |= (1 << u)

    # Precompute list of edges to process, excluding fixed ones
    edges_to_process = []
    for idx, (u, v) in enumerate(edge_list):
        if (u, v) not in fixed_edges and (v, u) not in fixed_edges:
            edges_to_process.append(idx)

    def has_path_len5(u, v, color):
        """
        Check if there is a path of length exactly 5 from u to v in the color subgraph.
        This would form a C6 if we add edge (u,v) with this color.
        We use dynamic programming with bitmasks.
        """
        # dist[1] = neighbors of u (excluding v, as edge (u,v) is not yet added)
        # Actually, if (u,v) is already in adj_mask[color], we are checking if adding it creates a cycle.
        # But in our backtracking, we only call this BEFORE adding the edge.
        # So adj_mask[color] does NOT contain (u,v) yet.
        
        # We want to know if there is a path u -> ... -> v of length 5.
        # Let R[d] be the set of nodes reachable from u in exactly d steps.
        # R[1] = adj_mask[color][u]
        # R[d] = union of adj_mask[color][x] for x in R[d-1]
        
        # Optimization: We can stop early if R[5] intersects {v}.
        # Also, we should ensure the path is simple? 
        # For C6, the cycle must be simple. 
        # If there is a walk of length 5 from u to v, does it imply a simple path?
        # Not necessarily. But if there is a walk, it might contain a cycle.
        # However, for small n=17, the probability of non-simple walks being the only connection is low.
        # A more rigorous check would require tracking visited nodes, which is expensive.
        # Given the constraints, we assume that detecting a walk of length 5 is a sufficient
        # (though not necessary) condition for pruning. 
        # Wait, if we prune when a walk exists, we might prune valid solutions where the only
        # walk of length 5 is non-simple. 
        # However, a non-simple walk of length 5 from u to v implies a shorter cycle or path.
        # If there is a walk of length 5, and no simple path of length 5, then the walk must
        # repeat nodes. This implies a cycle of length < 5 or a path that goes back and forth.
        # If we add (u,v), we form a cycle. If the walk is non-simple, the resulting "cycle"
        # is not a simple C6. 
        # So we should ideally check for SIMPLE paths.
        # But for performance, let's first try the walk check. If it's too aggressive, we'll fail to find a solution.
        # If it's not aggressive enough (allows C6s), we'll return an invalid solution.
        # The walk check is NECESSARY for a simple path. So if no walk exists, no simple path exists.
        # If a walk exists, a simple path MIGHT exist. 
        # So `has_path_len5` returning False is a safe prune (no C6 created).
        # Returning True is a safe prune ONLY IF the walk implies a simple path.
        # Actually, if we return True and prune, we might miss a solution where the only
        # length-5 connection is non-simple.
        # Let's refine: We want to know if adding (u,v) creates a SIMPLE C6.
        # This is equivalent to: Is there a simple path of length 5 from u to v?
        
        # For n=17, the number of simple paths of length 5 is manageable?
        # 17 * 16 * 15 * 14 * 13 * 12 is huge.
        # But we can use the bitmask DP and hope that non-simple walks are rare or that
        # the solution space is large enough that we don't hit this edge case.
        # Actually, a standard BFS finds the SHORTEST path.
        # If the shortest path is < 5, a simple path of length 5 might still exist.
        # If the shortest path is 5, then a simple path of length 5 exists.
        # If the shortest path is > 5, no simple path of length 5 exists? No, shortest path is min.
        
        # Let's stick to the DP for "walk of length 5".
        # This is a superset of "simple path of length 5".
        # If we prune on walk, we are being conservative (safe for validity, risky for completeness).
        # Wait, if we prune a branch where a walk exists but no simple path does, we might lose a valid solution.
        # However, in random graphs or dense graphs, if a walk exists, a simple path usually does too.
        # Let's try this.
        
        r1 = adj_mask[color][u]
        if not r1:
            return False
        
        # We don't need to exclude v from r1 because (u,v) is not in adj_mask yet.
        # But if v is in r1, that's a path of length 1.
        
        r2 = 0
        m = r1
        while m:
            lowbit = m & (-m)
            m ^= lowbit
            node = lowbit.bit_length() - 1
            r2 |= adj_mask[color][node]
        
        if not r2:
            return False
            
        r3 = 0
        m = r2
        while m:
            lowbit = m & (-m)
            m ^= lowbit
            node = lowbit.bit_length() - 1
            r3 |= adj_mask[color][node]
            
        if not r3:
            return False
            
        r4 = 0
        m = r3
        while m:
            lowbit = m & (-m)
            m ^= lowbit
            node = lowbit.bit_length() - 1
            r4 |= adj_mask[color][node]
            
        if not r4:
            return False
            
        r5 = 0
        m = r4
        while m:
            lowbit = m & (-m)
            m ^= lowbit
            node = lowbit.bit_length() - 1
            r5 |= adj_mask[color][node]
            
        return (r5 & (1 << v)) != 0

    def backtrack(pos):
        if time.time() > deadline:
            return False
        
        if pos == len(edges_to_process):
            return True
        
        idx = edges_to_process[pos]
        u, v = edge_list[idx]
        
        for c in range(4):
            if has_path_len5(u, v, c):
                continue
            
            # Assign color c
            colors[idx] = c
            adj_mask[c][u] |= (1 << v)
            adj_mask[c][v] |= (1 << u)
            
            if backtrack(pos + 1):
                return True
            
            # Undo
            adj_mask[c][u] &= ~(1 << v)
            adj_mask[c][v] &= ~(1 << u)
            colors[idx] = 0
            
        return False

    # Start backtracking
    if backtrack(0):
        return {'n': n, 'colors': colors}
    
    # Fallback if n=17 fails (should not happen if R_4(C6) > 17 is true and search is complete)
    # But due to time limits or pruning issues, it might fail.
    # Try n=16?
    n = 16
    num_edges = n * (n - 1) // 2
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_list.append((i, j))
            
    edge_idx_map = {}
    for idx, (u, v) in enumerate(edge_list):
        edge_idx_map[(u, v)] = idx
        edge_idx_map[(v, u)] = idx
        
    fixed_edges = {
        (0, 1): 0,
        (0, 2): 0,
        (0, 3): 0
    }
    
    colors = [0] * num_edges
    for (u, v), c in fixed_edges.items():
        idx = edge_idx_map[(u, v)]
        colors[idx] = c

    adj_mask = [[0] * n for _ in range(4)]
    for (u, v), c in fixed_edges.items():
        adj_mask[c][u] |= (1 << v)
        adj_mask[c][v] |= (1 << u)

    edges_to_process = []
    for idx, (u, v) in enumerate(edge_list):
        if (u, v) not in fixed_edges and (v, u) not in fixed_edges:
            edges_to_process.append(idx)

    def has_path_len5_n16(u, v, color):
        r1 = adj_mask[color][u]
        if not r1:
            return False
        r2 = 0
        m = r1
        while m:
            lowbit = m & (-m)
            m ^= lowbit
            node = lowbit.bit_length() - 1
            r2 |= adj_mask[color][node]
        if not r2:
            return False
        r3 = 0
        m = r2
        while m:
            lowbit = m & (-m)
            m ^= lowbit
            node = lowbit.bit_length() - 1
            r3 |= adj_mask[color][node]
        if not r3:
            return False
        r4 = 0
        m = r3
        while m:
            lowbit = m & (-m)
            m ^= lowbit
            node = lowbit.bit_length() - 1
            r4 |= adj_mask[color][node]
        if not r4:
            return False
        r5 = 0
        m = r4
        while m:
            lowbit = m & (-m)
            m ^= lowbit
            node = lowbit.bit_length() - 1
            r5 |= adj_mask[color][node]
        return (r5 & (1 << v)) != 0

    def backtrack_n16(pos):
        if time.time() > deadline:
            return False
        if pos == len(edges_to_process):
            return True
        idx = edges_to_process[pos]
        u, v = edge_list[idx]
        for c in range(4):
            if has_path_len5_n16(u, v, c):
                continue
            colors[idx] = c
            adj_mask[c][u] |= (1 << v)
            adj_mask[c][v] |= (1 << u)
            if backtrack_n16(pos + 1):
                return True
            adj_mask[c][u] &= ~(1 << v)
            adj_mask[c][v] &= ~(1 << u)
            colors[idx] = 0
        return False

    if backtrack_n16(0):
        return {'n': n, 'colors': colors}

    # Fallback to a known valid small n
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
