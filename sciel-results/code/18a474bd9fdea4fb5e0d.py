"""
Improved program using LUW construction for base graph and Z_2 voltage lift.
Constructs a 5-regular graph of girth 8 (LUW d=3) and lifts it to girth 9.
"""
# EVOLVE-BLOCK-START
import numpy as np

def construct(k, g):
    import collections
    
    # We aim for girth g.
    # Strategy:
    # 1. Construct LUW graph with d = g - 5 (if odd) or g - 4 (if even) to get girth >= g-1 or g.
    # For g=9, we want a base graph of girth 8.
    # LUW with d=3 gives girth >= 3+5=8.
    # 2. Lift the base graph using Z_2 voltages to kill all 8-cycles, resulting in girth >= 9.
    
    q = k
    # Determine base d for LUW
    # We want base girth g_base such that lifting gives girth >= g.
    # If base girth is g-1, and we kill all cycles of length g-1, new girth >= g.
    # LUW d=3 -> girth 8.
    # LUW d=5 -> girth 10. (Too big, n=6250)
    # Let's try d=3 for base.
    
    base_d = 3
    if g < 8:
        # Fallback to direct LUW if g is small
        base_d = max(3, g - 5)
        if base_d % 2 == 0: base_d += 1
        
    # 1. Construct Base LUW Graph
    if q == 4:
        mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def gf_add(x, y): return x ^ y
        def gf_mul(x, y): return mul_table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_mul(x, y): return (x * y) % q

    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    # rules for d=3 are sufficient
    
    start_state = (0, tuple([0] * base_d))
    visited = {start_state: 0}
    queue = collections.deque([start_state])
    base_edges = []
    adj = {}
    
    while queue:
        current = queue.popleft()
        side, vec = current
        cid = visited[current]
        if cid not in adj:
            adj[cid] = []
            
        for first_val in range(q):
            new_vec_list = [0] * base_d
            new_vec_list[0] = first_val
            
            if side == 0:
                for c_idx in range(1, base_d):
                    c_val = c_idx + 1
                    a, b = rules[c_val]
                    val = gf_add(vec[c_idx], gf_mul(new_vec_list[a-1], vec[b-1]))
                    new_vec_list[c_idx] = val
            else:
                for c_idx in range(1, base_d):
                    c_val = c_idx + 1
                    a, b = rules[c_val]
                    val = gf_add(vec[c_idx], gf_mul(vec[a-1], new_vec_list[b-1]))
                    new_vec_list[c_idx] = val
            
            neighbor_state = (1 - side, tuple(new_vec_list))
            if neighbor_state not in visited:
                visited[neighbor_state] = len(visited)
                queue.append(neighbor_state)
            
            nid = visited[neighbor_state]
            if side == 0:
                base_edges.append((cid, nid))
                adj[cid].append(nid)
                if nid not in adj: adj[nid] = []
                adj[nid].append(cid)
                
    n_base = len(visited)
    
    # 2. Find all 8-cycles in the base graph
    # Since the graph is bipartite and regular, we can find 8-cycles by looking at paths of length 4 from a vertex back to itself.
    # To avoid duplicates, we only consider cycles where the first vertex is the smallest index in the cycle.
    
    cycles = []
    # Precompute paths of length 2 and 3?
    # Or just BFS from each vertex to depth 4?
    # N=250, Degree=5.
    # Paths of length 4: 5^4 = 625 per vertex. 250 * 625 = 156,250 checks. Very fast.
    
    # Map edge to index for voltage assignment
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    # Find unique 8-cycles
    # A cycle is a sequence v0, v1, v2, v3, v4, v5, v6, v7, v0
    # We can generate by picking v0, then v1 in N(v0), v2 in N(v1)\{v0}, etc.
    # But simpler: BFS to depth 4.
    
    # To ensure we get each cycle once, we can use a canonical form or just collect all and deduplicate later.
    # Given the small size, collecting all and deduplicating is safe.
    
    all_8_cycles = set()
    
    for v0 in range(n_base):
        # BFS from v0 to depth 4
        # dist 0: {v0}
        # dist 1: N(v0)
        # dist 2: N(N(v0)) \ {v0}
        # dist 3: ...
        # dist 4: ...
        
        # We want paths v0 -> v1 -> v2 -> v3 -> v4 -> v0
        # This is equivalent to finding v4 in N(v0) such that there is a path of length 3 from v0 to v4.
        
        # Let's just enumerate all simple paths of length 4 starting at v0.
        # Since it's a tree locally up to depth 3 (girth 8), there are no cycles of length < 8.
        # So paths of length 4 are simple.
        
        paths_len_4 = []
        # v1
        for v1 in adj[v0]:
            # v2
            for v2 in adj[v1]:
                if v2 == v0: continue
                # v3
                for v3 in adj[v2]:
                    if v3 in (v0, v1): continue
                    # v4
                    for v4 in adj[v3]:
                        if v4 in (v0, v1, v2): continue
                        # Check if v4 is connected to v0
                        if v0 in adj[v4]:
                            # Cycle found: v0-v1-v2-v3-v4-v0
                            # Canonical representation: sort vertices?
                            # Or just store the tuple of edges?
                            # Edge indices:
                            e1 = edge_index[(v0, v1)]
                            e2 = edge_index[(v1, v2)]
                            e3 = edge_index[(v2, v3)]
                            e4 = edge_index[(v3, v4)]
                            e5 = edge_index[(v4, v0)]
                            # The cycle has 5 edges? No, 8-cycle has 8 edges.
                            # Wait. v0->v1->v2->v3->v4->v0 is length 5.
                            # I need length 8.
                            # v0->v1->v2->v3->v4->v5->v6->v7->v0
                            # Depth 4 from v0 reaches v4.
                            # We need v7 to be connected to v0.
                            # So we need paths of length 7 from v0 to v7, and v7 in N(v0).
                            
                            pass # This logic is flawed for 8-cycles.
                            
    # Correct approach for 8-cycles:
    # A cycle of length 8 is v0-v1-v2-v3-v4-v5-v6-v7-v0.
    # v4 is at distance 4 from v0.
    # v7 is at distance 3 from v0 (via v6, v5, v4) and 1 from v0.
    # So we look for v4 at distance 4 from v0.
    # Then from v4, go 3 steps to reach a neighbor of v0.
    
    # Let's implement a helper to get all simple paths of length L from a node.
    def get_paths(start, length):
        if length == 0:
            return [[start]]
        paths = []
        for neighbor in adj[start]:
            sub_paths = get_paths(neighbor, length - 1)
            for p in sub_paths:
                if start not in p[:-1]: # Ensure simple path
                    paths.append([start] + p)
        return paths

    # This recursive approach might be slow if not pruned, but N=250, L=7 is small.
    # However, we can optimize by checking distance.
    
    # Simpler:
    # For each v0:
    #   Find all v4 at distance 4 from v0.
    #   For each v4, find all v7 at distance 3 from v4 such that v7 is a neighbor of v0.
    #   The path v0->...->v4->...->v7->v0 is an 8-cycle.
    
    # Precompute distances and parents?
    # Given small N, we can just use BFS.
    
    all_8_cycles_edges = set()
    
    for v0 in range(n_base):
        # BFS to find nodes at distance 4
        dist = {v0: 0}
        parent = {v0: None}
        q_bfs = collections.deque([v0])
        while q_bfs:
            u = q_bfs.popleft()
            if dist[u] == 4:
                continue
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q_bfs.append(v)
                    
        # Get all nodes at distance 4
        v4_candidates = [v for v, d in dist.items() if d == 4]
        
        for v4 in v4_candidates:
            # We need a path v4 -> v5 -> v6 -> v7 such that v7 is in N(v0) and v7 != v0.
            # And the path v0->...->v4 is simple (guaranteed by BFS tree? No, BFS tree is simple, but there might be other paths).
            # Actually, for girth 8, the path of length 4 is unique?
            # No, there can be multiple paths.
            # But we just need *any* 8-cycle to kill.
            # If we kill all 8-cycles, we are safe.
            
            # Find all simple paths of length 3 from v4 to any neighbor of v0.
            # v5 in N(v4) \ {parent[v4]}
            # v6 in N(v5) \ {v4, parent[v5]}
            # v7 in N(v6) \ {v5, parent[v6]}
            # Check if v7 in N(v0)
            
            p_v4 = parent[v4]
            for v5 in adj[v4]:
                if v5 == p_v4: continue
                p_v5 = parent.get(v5)
                # Note: v5 might not be in the BFS tree if it was visited earlier via a different path?
                # In a tree, it is. In a graph, it might be.
                # But we are looking for *any* path.
                # To keep it simple, let's just enumerate all simple paths of length 3 from v4.
                
                for v6 in adj[v5]:
                    if v6 in (v4, v5): continue
                    for v7 in adj[v6]:
                        if v7 in (v5, v6): continue
                        if v7 == v0: continue
                        if v7 in adj[v0]:
                            # Cycle found: v0...v4-v5-v6-v7-v0
                            # We need the full edge list.
                            # Reconstruct path v0->v4 from parent pointers.
                            path_v0_v4 = []
                            cur = v4
                            while cur != v0:
                                path_v0_v4.append(cur)
                                cur = parent[cur]
                            path_v0_v4.reverse()
                            # path_v0_v4 is [v1, v2, v3, v4]
                            # Full cycle: v0, v1, v2, v3, v4, v5, v6, v7, v0
                            vertices = [v0] + path_v0_v4 + [v5, v6, v7]
                            # Get edge indices
                            edge_set = set()
                            for i in range(8):
                                u, v = vertices[i], vertices[(i+1)%8]
                                edge_set.add(edge_index[(u, v)])
                            # Canonical form: sorted tuple of edge indices
                            all_8_cycles_edges.add(tuple(sorted(edge_set)))
                            
    if not all_8_cycles_edges:
        # No 8-cycles found? Then girth is already > 8.
        # Return the base graph (lifted with 0 voltages or just base if g<=8)
        # But we want girth 9. If base has girth > 8, it might have girth 9 or 10.
        # If base has girth 10, we can just return it? No, we want min n.
        # If base has no 8-cycles, it has girth >= 9.
        # In that case, we can just return the base graph (n=250) if g<=10.
        # But the problem says g=9.
        # If base girth is 10, it satisfies girth >= 9.
        # So we can return the base graph.
        
        # However, LUW d=3 is known to have girth 8. So this case is unlikely.
        return n_base, base_edges

    # 3. Solve Linear System
    # Variables: x_e for e in base_edges
    # Equations: sum(x_e for e in C) = 1 for C in all_8_cycles_edges
    
    num_edges = len(base_edges)
    # Use numpy for Gaussian elimination over Z_2
    # Matrix A of shape (num_cycles, num_edges)
    # Vector b of shape (num_cycles,)
    
    num_cycles = len(all_8_cycles_edges)
    A = np.zeros((num_cycles, num_edges), dtype=np.uint8)
    b = np.ones(num_cycles, dtype=np.uint8)
    
    for i, cycle in enumerate(all_8_cycles_edges):
        for e in cycle:
            A[i, e] = 1
            
    # Gaussian Elimination
    # We want to find x such that A x = b (mod 2)
    
    # Augmented matrix
    M = np.column_stack((A, b))
    rows, cols = M.shape
    rank = 0
    
    for col in range(num_edges):
        # Find pivot
        pivot = -1
        for r in range(rank, rows):
            if M[r, col] == 1:
                pivot = r
                break
        if pivot == -1:
            continue
            
        # Swap
        M[[rank, pivot]] = M[[pivot, rank]]
        
        # Eliminate
        for r in range(rows):
            if r != rank and M[r, col] == 1:
                M[r] ^= M[rank]
                
        rank += 1
        
    # Check for consistency
    for r in range(rank, rows):
        if M[r, num_edges] == 1:
            # Inconsistent
            # Fallback to direct LUW d=5
            pass # Handle fallback
            
    # If consistent, find a solution
    # Set free variables to 0
    x = np.zeros(num_edges, dtype=np.uint8)
    
    # Back substitution
    # This is a simplified version. For a proper solution, we need to track pivots.
    # Given the small size, we can just use a solver or iterate.
    # But we need *a* solution.
    
    # Let's use a simple iterative solver or just pick the first basic solution.
    # Since we just need *one* valid voltage assignment, and the system is linear over Z_2,
    # if it's consistent, a solution exists.
    
    # To get a specific solution, we can perform back substitution on the reduced matrix.
    # However, implementing full back substitution for Z_2 is tedious.
    # Alternative: Randomized search?
    # Or just use `scipy`? No, standard lib + numpy.
    
    # Let's assume consistency and try to find a solution by setting free vars to 0.
    # We need to identify which columns are pivots.
    
    pivot_cols = []
    M2 = np.column_stack((A, b))
    rank2 = 0
    for col in range(num_edges):
        pivot = -1
        for r in range(rank2, rows):
            if M2[r, col] == 1:
                pivot = r
                break
        if pivot == -1:
            continue
        M2[[rank2, pivot]] = M2[[pivot, rank2]]
        for r in range(rows):
            if r != rank2 and M2[r, col] == 1:
                M2[r] ^= M2[rank2]
        pivot_cols.append(col)
        rank2 += 1
        
    # Check consistency again
    for r in range(rank2, rows):
        if M2[r, num_edges] == 1:
            # Inconsistent, fallback
            # Construct LUW d=5
            return _luw_direct(k, g)
            
    # Back substitution
    x_sol = np.zeros(num_edges, dtype=np.uint8)
    for i in range(rank2 - 1, -1, -1):
        col = pivot_cols[i]
        val = M2[i, num_edges]
        for j in range(col + 1, num_edges):
            if M2[i, j] == 1:
                val ^= x_sol[j]
        x_sol[col] = val
        
    # 4. Construct Lifted Graph
    # Vertices: (v, s) where v in 0..n_base-1, s in {0,1}
    # Edges: (v, s) -> (w, s ^ x_e)
    
    n_lifted = 2 * n_base
    lifted_edges = []
    
    for u, v in base_edges:
        e_idx = edge_index[(u, v)]
        voltage = x_sol[e_idx]
        
        # Edge from (u, 0) to (v, voltage)
        lifted_edges.append((u * 2, v * 2 + voltage))
        # Edge from (u, 1) to (v, 1 ^ voltage)
        lifted_edges.append((u * 2 + 1, v * 2 + (1 ^ voltage)))
        
    return n_lifted, lifted_edges

def _luw_direct(k, g):
    # Fallback to original LUW
    import collections
    q = k
    min_d = g - 5
    if min_d < 3: min_d = 3
    d = min_d
    if d % 2 == 0: d += 1
    
    if q == 4:
        mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def gf_add(x, y): return x ^ y
        def gf_mul(x, y): return mul_table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_mul(x, y): return (x * y) % q

    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        if r == 0: rules[c] = (1, c0 - 2)
        elif r == 1: rules[c] = (c0 - 1, 1)
        elif r == 2: rules[c] = (c0, 1)
        else: rules[c] = (1, c0 + 1)

    start_state = (0, tuple([0] * d))
    visited = {start_state: 0}
    queue = collections.deque([start_state])
    edges = []
    
    while queue:
        current = queue.popleft()
        side, vec = current
        for first_val in range(q):
            new_vec_list = [0] * d
            new_vec_list[0] = first_val
            if side == 0:
                for c_idx in range(1, d):
                    c_val = c_idx + 1
                    a, b = rules[c_val]
                    val = gf_add(vec[c_idx], gf_mul(new_vec_list[a-1], vec[b-1]))
                    new_vec_list[c_idx] = val
            else:
                for c_idx in range(1, d):
                    c_val = c_idx + 1
                    a, b = rules[c_val]
                    val = gf_add(vec[c_idx], gf_mul(vec[a-1], new_vec_list[b-1]))
                    new_vec_list[c_idx] = val
            neighbor_state = (1 - side, tuple(new_vec_list))
            if neighbor_state not in visited:
                visited[neighbor_state] = len(visited)
                queue.append(neighbor_state)
            if side == 0:
                edges.append((visited[current], visited[neighbor_state]))
                
    return len(visited), edges

# EVOLVE-BLOCK-END
