import time
import random
import sys

def construct():
    """
    Return a C4-free graph with alpha < 11, maximizing n.
    
    Strategy: 
    1. Build a C4-free base graph (polarity graph of PG(2,5) with 31 vertices).
    2. Use randomized greedy extension with fast bitmask C4 and independence checks.
    3. Incorporate CSP-style domain pruning: when adding a new vertex, 
       precompute which existing vertices it CAN connect to without creating C4,
       then search over subsets of that allowed set.
    4. Use a faster independence number check via branch-and-bound with 
       greedy coloring upper bound.
    """
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    # Build base: polarity graph of PG(2,5) with 31 vertices
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
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
    points = list(set(points))
    n_base = len(points)

    def bilinear(u, v):
        return (u[0] * v[2] + u[1] * v[1] + u[2] * v[0]) % 5

    adj_base = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if bilinear(points[i], points[j]) == 0:
                adj_base[i] |= (1 << j)
                adj_base[j] |= (1 << i)

    # Fast C4 check using bitmasks
    def is_c4_free(n, adj):
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common & (common - 1):  # more than 1 bit set
                    return False
        return True

    # Fast independence number check with branch-and-bound
    def has_independent_set_of_size_11(n, adj):
        """Returns True if there exists an independent set of size >= 11."""
        
        def popcount(x):
            c = 0
            while x:
                x &= x - 1
                c += 1
            return c

        # Greedy coloring for upper bound on clique number in complement
        # (i.e., upper bound on independent set size in original)
        def greedy_color_bound(candidates_mask):
            """Upper bound on independent set size in subgraph induced by candidates_mask."""
            if candidates_mask == 0:
                return 0
            # Greedy coloring of the complement (clique in complement = independent set in original)
            # Actually, we want an upper bound on the size of the largest independent set
            # in the subgraph induced by candidates. We use greedy coloring of the
            # subgraph: chromatic number >= independence number is FALSE.
            # Actually: alpha(G) <= chi(G) is not true in general.
            # But: alpha(G) <= n - chi(complement of G) ... this is getting complicated.
            # Let's use a simpler bound: just count vertices, or use a greedy independent set
            # as a lower bound and the total count as upper bound.
            # For pruning, we'll use: if current_size + popcount(candidates_mask) < 11, prune.
            return popcount(candidates_mask)

        found = [False]

        def backtrack(candidates_mask, current_size):
            if found[0]:
                return
            if current_size >= 11:
                found[0] = True
                return
            if candidates_mask == 0:
                return
            if current_size + popcount(candidates_mask) < 11:
                return

            # Pick a vertex from candidates with highest degree in the candidate subgraph
            # for better pruning
            best_v = -1
            best_deg = -1
            temp = candidates_mask
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                temp ^= lb
                # Degree of v in the candidate subgraph
                deg = popcount(adj[v] & candidates_mask)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v

            if best_v < 0:
                return

            # Try including best_v
            # New candidates: those in candidates that are NOT neighbors of best_v
            new_cands = candidates_mask & ~(adj[best_v] | (1 << best_v))
            backtrack(new_cands, current_size + 1)

            if found[0]:
                return

            # Try excluding best_v
            backtrack(candidates_mask & ~(1 << best_v), current_size)

        backtrack((1 << n) - 1, 0)
        return found[0]

    # Start with base graph
    best_n = n_base
    best_adj = adj_base[:]
    best_edges = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if adj_base[i] & (1 << j):
                best_edges.append([i, j])

    # Try to extend the graph by adding vertices
    max_extend = 43 - n_base

    def try_add_vertex(adj, n, deadline):
        """
        Try to add one new vertex (index n) to the graph with adjacency 'adj' on n vertices.
        Returns (success, new_adj) where new_adj has n+1 entries if successful.
        Uses CSP-style domain pruning: precompute allowed neighbors.
        """
        candidate_n = n + 1
        new_vertex_idx = n

        # CSP-style domain pruning: determine which existing vertices the new vertex
        # CAN be connected to without immediately creating a C4.
        # If new vertex connects to a and b, then N(a) ∩ N(b) must be empty
        # (in the original graph, not counting the new vertex).
        # More precisely: for any two neighbors a, b of the new vertex,
        # adj[a] & adj[b] must be 0 (no common neighbors in the existing graph).
        
        # First, compute for each pair (a,b) whether they can both be neighbors
        # of the new vertex: adj[a] & adj[b] == 0
        allowed_mask = (1 << n) - 1  # initially all vertices allowed
        
        # Precompute incompatibility: if adj[a] & adj[b] != 0, then a and b can't
        # both be neighbors of the new vertex.
        # We'll build a conflict graph on the existing vertices.
        
        # For each vertex a, find which vertices b conflict with a (can't both be neighbors)
        conflict = [0] * n
        for a in range(n):
            for b in range(a + 1, n):
                if adj[a] & adj[b]:
                    conflict[a] |= (1 << b)
                    conflict[b] |= (1 << a)
        
        # Now, a subset S of existing vertices is a valid neighbor set for the new vertex
        # iff for all a, b in S: conflict[a] & (1 << b) == 0, i.e., S is an independent
        # set in the conflict graph.
        
        # We want to find a non-empty (or even empty) independent set S in the conflict graph
        # such that the resulting graph has alpha < 11.
        
        # Try different sizes of S, starting from small
        # First try: empty set (isolated vertex)
        candidate_adj = adj + [0]
        if is_c4_free(candidate_n, candidate_adj):
            if not has_independent_set_of_size_11(candidate_n, candidate_adj):
                return (True, candidate_adj)
        
        # Try single-vertex connections
        for a in range(n):
            if time.time() > deadline:
                return (False, None)
            candidate_adj = adj + [0]
            candidate_adj[a] |= (1 << new_vertex_idx)
            candidate_adj[new_vertex_idx] |= (1 << a)
            if is_c4_free(candidate_n, candidate_adj):
                if not has_independent_set_of_size_11(candidate_n, candidate_adj):
                    return (True, candidate_adj)
        
        # Try two-vertex connections (a, b) where a and b don't conflict
        for a in range(n):
            if time.time() > deadline:
                return (False, None)
            for b in range(a + 1, n):
                if conflict[a] & (1 << b):
                    continue
                candidate_adj = adj + [0]
                candidate_adj[a] |= (1 << new_vertex_idx)
                candidate_adj[b] |= (1 << new_vertex_idx)
                candidate_adj[new_vertex_idx] |= (1 << a) | (1 << b)
                if is_c4_free(candidate_n, candidate_adj):
                    if not has_independent_set_of_size_11(candidate_n, candidate_adj):
                        return (True, candidate_adj)
        
        # Try random subsets of size 3-5 from the allowed set
        # Build list of vertices that can potentially be in a neighbor set
        # (vertices with low conflict degree)
        vertices_by_conflict_deg = sorted(range(n), key=lambda v: popcount(conflict[v]))
        
        for subset_size in range(3, 6):
            if time.time() > deadline:
                return (False, None)
            # Try a few random independent sets in the conflict graph of given size
            for _ in range(20):
                if time.time() > deadline:
                    return (False, None)
                # Greedily build an independent set in conflict graph
                S = []
                remaining = set(vertices_by_conflict_deg[:min(n, 20)])  # focus on low-conflict vertices
                while len(S) < subset_size and remaining:
                    # Pick vertex with lowest conflict degree among remaining
                    best_v = min(remaining, key=lambda v: popcount(conflict[v] & ((1 << n) - 1)))
                    S.append(best_v)
                    remaining.discard(best_v)
                    # Remove all vertices that conflict with best_v
                    to_remove = [u for u in remaining if conflict[best_v] & (1 << u)]
                    for u in to_remove:
                        remaining.discard(u)
                
                if len(S) != subset_size:
                    continue
                
                # Verify S is independent in conflict graph
                valid = True
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        if conflict[S[i]] & (1 << S[j]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                candidate_adj = adj + [0]
                for v in S:
                    candidate_adj[v] |= (1 << new_vertex_idx)
                    candidate_adj[new_vertex_idx] |= (1 << v)
                
                if is_c4_free(candidate_n, candidate_adj):
                    if not has_independent_set_of_size_11(candidate_n, candidate_adj):
                        return (True, candidate_adj)
        
        return (False, None)

    # Main search loop
    for attempt in range(80):
        if time.time() > deadline:
            break

        # Reset to base
        adj = adj_base[:]
        n = n_base

        # Try to add vertices one by one
        for new_idx in range(max_extend):
            if time.time() > deadline:
                break

            success, new_adj = try_add_vertex(adj, n, deadline)
            if success:
                adj = new_adj
                n += 1
            else:
                break

        # Update best if we did better
        if n > best_n:
            best_n = n
            best_adj = adj[:]
            best_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        best_edges.append([i, j])

    # Final verification
    if not is_c4_free(best_n, best_adj):
        # Fall back to base
        best_n = n_base
        best_adj = adj_base[:]
        best_edges = []
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if adj_base[i] & (1 << j):
                    best_edges.append([i, j])

    return {'n': best_n, 'edges': best_edges}
