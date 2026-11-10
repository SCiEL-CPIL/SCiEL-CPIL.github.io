import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n = 16
    FULL = (1 << n) - 1

    # Precompute: for each vertex v, the set of vertices that CANNOT be neighbors
    # due to triangle-freeness given current adjacencies.
    # We'll build adjacency incrementally.

    def is_3_colorable(adj):
        """
        Fast 3-colorability check via backtracking with bitsets.
        Returns True if 3-colorable.
        """
        # Order by degree descending
        degrees = [bin(a).count('1') for a in adj]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)

        colors = [-1] * n

        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            forbidden = 0
            neighbors = adj[v]
            while neighbors:
                lsb = neighbors & -neighbors
                u = lsb.bit_length() - 1
                if colors[u] != -1:
                    forbidden |= (1 << colors[u])
                neighbors -= lsb

            for c in range(3):
                if not (forbidden & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False

        return backtrack(0)

    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Strategy: Backtracking over vertices 0..15.
    # For each vertex v, we choose its neighbor set among {v+1, ..., 15} (only forward edges to avoid symmetry).
    # Constraint: triangle-free means if v is connected to u and w, then u and w must NOT be connected.
    #   i.e., N(v) must be an independent set in the subgraph induced by {v+1,...,15}.
    # We process vertices in order 0,1,...,15.
    # When assigning neighbors to vertex v, those neighbors are from {v+1,...,15}.
    # The triangle-free constraint: for any two neighbors u,w of v (u<w, both > v),
    #   the edge (u,w) must NOT exist in the final graph.
    #   Since u,w > v, their edges haven't been assigned yet (we go in order).
    #   So we just need to remember: for each pair (u,w) with u<w, whether they are "forbidden"
    #   to be connected (because some earlier vertex is connected to both).

    # Better formulation:
    # Process vertices 0..15. For vertex v, choose a subset S_v ⊆ {v+1,...,15} as its neighbors.
    # Triangle-free: for any u < v < w, if (u,v) and (v,w) are edges, then (u,w) must NOT be an edge.
    #   Equivalently: when we assign S_v, for any u in S_v (u > v) and any previously-assigned
    #   vertex w < v that is connected to v, we must have u NOT in S_w... wait, that's backwards.
    
    # Let me re-think. Process vertices 0, 1, ..., 15.
    # When we're at vertex v, we decide which vertices in {v+1,...,15} are neighbors of v.
    # Let S_v = set of neighbors of v that are > v.
    # Triangle-free constraint: if u ∈ S_v (so (v,u) is an edge, u > v), and w < v with (w,v) edge (i.e., v ∈ S_w),
    #   then (w,u) must NOT be an edge. Since w < v < u, the edge (w,u) would be in S_w.
    #   So: u ∉ S_w for all w < v where v ∈ S_w.
    #   Equivalently: S_v ∩ (∪_{w < v, v ∈ S_w} S_w) = ∅? No...
    #   For each w < v such that v ∈ S_w (i.e., w is connected to v), we need S_v ∩ S_w = ∅.
    #   Wait no. The triangle is (w, v, u). Edges (w,v) and (v,u) exist. Triangle-free means (w,u) absent.
    #   (w,u) is in S_w (since w < u). So u ∉ S_w.
    #   So the constraint is: for all w < v with v ∈ S_w, we have S_v ∩ S_w = ∅? 
    #   No, it's: for all w < v with v ∈ S_w, and for all u ∈ S_v, we need u ∉ S_w.
    #   I.e., S_v ∩ S_w = ∅ for all w < v where v ∈ S_w.
    
    # Actually simpler: maintain a bitmask "forbidden[v]" = set of vertices > v that CANNOT be neighbors of v
    #   due to triangle-freeness constraints from earlier vertices.
    #   When we assign S_v, we must have S_v ⊆ (allowed_v \ forbidden_v).
    #   After assigning S_v, for each u ∈ S_v, and for each w < v where v ∈ S_w,
    #   we need to ensure u ∉ S_w. But S_w is already fixed!
    #   So actually, the constraint must be checked BEFORE we fix S_w.
    
    # Revised approach:
    # Process v = 0, 1, ..., 15.
    # Maintain: for each pair (i,j) with i<j, a boolean "edge_ij" (or "forbidden_ij").
    # When processing vertex v, we choose S_v ⊆ {v+1,...,15} such that:
    #   (a) For all u ∈ S_v: edge(v,u) is not forbidden (i.e., no triangle formed with earlier vertices).
    #       Specifically: for all w < v where edge(w,v) exists, we need edge(w,u) = False for all u ∈ S_v.
    #       So S_v must be disjoint from S_w for all w < v where edge(w,v) is true.
    #   (b) For all u1, u2 ∈ S_v: edge(u1,u2) must be False (triangle v-u1-u2).
    #       This means S_v must be an independent set in the "future" graph, but since future edges
    #       aren't determined yet, we just need to FORBID edge(u1,u2) for all u1,u2 ∈ S_v.
    
    # So the algorithm:
    # - Maintain a 2D boolean array or bitmask "forbidden" where forbidden[i] is a bitmask of j > i
    #   that CANNOT be connected to i (due to triangle constraints from already-processed vertices).
    # - Also maintain "edge" bitmasks for already-determined edges.
    # - When processing vertex v:
    #   * Compute allowed = {v+1,...,15} minus forbidden[v].
    #   * Also, S_v must be an independent set in the sense that no two elements of S_v
    #     are connected by an already-forbidden edge... actually no, we're CHOOSING S_v.
    #     The constraint is: for u1, u2 ∈ S_v, we must FORBID edge(u1,u2) going forward.
    #     But if edge(u1,u2) was already determined (can't happen since u1,u2 > v and we process in order,
    #     so edge(u1,u2) hasn't been decided yet). Good.
    #   * Additionally: for w < v where edge(w,v) exists, we need u ∉ S_w for all u ∈ S_v.
    #     But S_w is already fixed! So this means: u must not be in S_w.
    #     Since S_w is fixed, this is a constraint on S_v: S_v ∩ S_w = ∅ for all w < v with edge(w,v).
    #     This can be precomputed as part of forbidden[v].
    
    # Let me just implement it cleanly:

    # State: 
    #   adj[i] for i < current_v: bitmask of neighbors (only bits for j > i matter for forward edges)
    #   forbidden[i] for i < current_v: bitmask of j > i that CANNOT be neighbors of i
    #   (forbidden is updated as we go)
    
    # When processing vertex v:
    #   * candidates = FULL & ~((1 << (v+1)) - 1)  # vertices > v
    #     candidates &= ~forbidden[v]
    #   * S_v must be a subset of candidates
    #   * S_v must be an independent set in the "forbidden" sense: for u1, u2 ∈ S_v,
    #     we need to check that adding edge(v,u1) and edge(v,u2) doesn't create a triangle
    #     with any already-existing edge. Since u1, u2 > v, the only triangle would be (v, u1, u2)
    #     which requires edge(u1, u2). But edge(u1, u2) hasn't been decided yet.
    #     So we just need to FORBID edge(u1, u2) for all pairs in S_v.
    #   * Also: for w < v where edge(w, v) exists (i.e., v is in adj[w]), we need
    #     S_v ∩ adj[w] = ∅. (Because triangle w-v-u would form if u ∈ S_v ∩ adj[w].)
    #     Wait, adj[w] contains neighbors of w. If v ∈ adj[w] (edge w-v exists) and u ∈ adj[w] (edge w-u exists),
    #     and u ∈ S_v (edge v-u exists), then triangle w-v-u. So yes, S_v ∩ adj[w] = ∅.
    
    # This is getting complex. Let me use a simpler but still exhaustive approach:
    # Use the fact that n=16 is small. The number of possible neighbor sets for vertex v
    # is 2^(15-v). Total search space is product of 2^(15-v) for v=0..15 = 2^(120) which is too big.
    # But with pruning, it should be manageable.

    # Actually, let me try a different structural approach:
    # Use the known result that the maximum edges in a triangle-free 4-chromatic graph on 16 vertices
    # is likely close to the Turan bound of 64. 
    # 
    # New structural strategy: Use a bipartite "backbone" (K_{8,8} minus some edges) plus
    # a small 4-chromatic core (Grötzsch or similar) embedded in one partition, and use
    # exact backtracking to determine which edges to remove from K_{8,8} to embed the core
    # while maintaining triangle-freeness.
    #
    # Even better: Direct exhaustive search with strong pruning.
    # Process vertices 0..15. For each, try all subsets of remaining vertices as neighbors.
    # Prune aggressively.

    # Let me implement the direct backtracking:

    best_edges = [0]
    best_adj = [None]

    def solve():
        n = 16
        adj = [0] * n  # adj[i] bitmask of neighbors
        # forbidden[i] = bitmask of j > i that cannot be adjacent to i
        forbidden = [0] * n
        
        # Process vertices 0..15
        def backtrack(v):
            if time.time() > deadline:
                return
            if v == n:
                # Check 3-colorability
                if not is_3_colorable(adj):
                    ec = edge_count(adj)
                    if ec > best_edges[0]:
                        best_edges[0] = ec
                        best_adj[0] = adj[:]
                return
            
            # Determine candidates for neighbors of v
            # Candidates are vertices > v that are not forbidden
            candidates_mask = 0
            for u in range(v + 1, n):
                if not (forbidden[v] >> u) & 1:
                    # Check: for all w < v where edge(w,v) exists, u must not be in adj[w]
                    # (to avoid triangle w-v-u)
                    ok = True
                    for w in range(v):
                        if (adj[w] >> v) & 1:
                            if (adj[w] >> u) & 1:
                                ok = False
                                break
                    if ok:
                        candidates_mask |= (1 << u)
            
            # Now we need to choose S_v ⊆ candidates_mask such that:
            # 1. S_v is "independent" in the sense that for any u1, u2 in S_v,
            #    we will forbid edge(u1, u2). This is always satisfiable (we just update forbidden).
            # 2. We want to maximize total edges, so we prefer larger S_v.
            
            # Enumerate all subsets of candidates_mask, largest first (for maximization)
            # But we also need to consider that choosing a smaller S_v might allow more edges later.
            
            # For efficiency, enumerate subsets in order of decreasing size.
            # candidates_mask has at most (n - v - 1) bits set.
            cands = []
            mask = candidates_mask
            while mask:
                lsb = mask & -mask
                cands.append(lsb.bit_length() - 1)
                mask -= lsb
            
            k = len(cands)
            
            # Generate all subsets, sorted by size descending
            # For k up to 15, this is 2^15 = 32768 subsets max for v=0.
            # But with pruning, most will be cut short.
            
            # Actually, for the first few vertices, this could be expensive.
            # Let's use a smarter approach: try the full candidate set first,
            # then subsets of size k-1, etc.
            
            # For each subset S, check if it's valid (no two elements create issues).
            # The only constraint within S is that we'll forbid edges between elements of S.
            # That's always fine. So any subset of candidates is valid!
            
            # So we just need to enumerate subsets of candidates_mask.
            # To maximize edges, try larger subsets first.
            
            # Enumerate subsets by iterating from full set down
            full_cand = candidates_mask
            
            # We'll iterate over all submasks of full_cand in order of decreasing popcount
            # For efficiency, group by size
            
            # Actually, let's just iterate all submasks. For v=0, that's up to 2^15=32768.
            # For v=1, up to 2^14=16384. Etc. Total is manageable with pruning.
            
            # Iterate submasks of full_cand
            sub = full_cand
            while True:
                # Try adding edges from v to all vertices in sub
                new_adj = adj[:]
                new_forbidden = forbidden[:]
                
                # Add edges
                new_adj[v] |= sub
                for u in range(v + 1, n):
                    if (sub >> u) & 1:
                        new_adj[u] |= (1 << v)
                
                # Update forbidden: for each pair u1, u2 in sub, forbid edge(u1, u2)
                # (triangle v-u1-u2)
                valid = True
                for i in range(len(cands)):
                    u1 = cands[i]
                    if not (sub >> u1) & 1:
                        continue
                    for j in range(i + 1, len(cands)):
                        u2 = cands[j]
                        if not (sub >> u2) & 1:
                            continue
                        # Forbid edge(u1, u2)
                        if (new_forbidden[u1] >> u2) & 1:
                            # Already forbidden, that's fine
                            pass
                        else:
                            # Check if edge(u1,u2) is already set (shouldn't be since u1,u2 > v)
                            if (new_adj[u1] >> u2) & 1:
                                valid = False
                                break
                            new_forbidden[u1] |= (1 << u2)
                    if not valid:
                        break
                
                if valid:
                    # Also need to check: for w < v where edge(w,v) exists,
                    # we already ensured sub ∩ adj[w] = ∅ in the candidates computation.
                    
                    adj[:] = new_adj
                    forbidden[:] = new_forbidden
                    
                    backtrack(v + 1)
                    
                    # Restore
                    adj[:] = new_adj  # already restored by recursion
                    # Actually, we modified in place. Let's use copies properly.
                    # This is getting messy. Let me restructure.
                
                if sub == 0:
                    break
                sub = (sub - 1) & full_cand
            
            return

    # The in-place modification is problematic. Let me restructure to pass state explicitly.
    
    def solve2():
        n = 16
        best_ec = [0]
        best_state = [None]
        
        def backtrack(v, adj, forbidden):
            if time.time() > deadline:
                return
            if v == n:
                if not is_3_colorable(adj):
                    ec = edge_count(adj)
                    if ec > best_ec[0]:
                        best_ec[0] = ec
                        best_state[0] = adj[:]
                return
            
            # Determine candidates
            candidates_mask = 0
            for u in range(v + 1, n):
                if (forbidden[v] >> u) & 1:
                    continue
                ok = True
                for w in range(v):
                    if (adj[w] >> v) & 1:
                        if (adj[w] >> u) & 1:
                            ok = False
                            break
                if ok:
                    candidates_mask |= (1 << u)
            
            # Get list of candidate vertices
            cands = []
            mask = candidates_mask
            while mask:
                lsb = mask & -mask
                cands.append(lsb.bit_length() - 1)
                mask -= lsb
            
            k = len(cands)
            
            # Try all subsets of cands, from largest to smallest
            # For efficiency, iterate submasks in order of decreasing popcount
            # But for now, just iterate all submasks of candidates_mask
            
            # To prioritize larger subsets, we can iterate from full down
            # submask enumeration: sub, (sub-1)&full, ...
            # This doesn't guarantee order by size, but we can sort.
            
            # For k <= 16, 2^k <= 65536. For early vertices this is fine.
            # Generate all submasks
            all_subs = []
            sub = candidates_mask
            while True:
                all_subs.append(sub)
                if sub == 0:
                    break
                sub = (sub - 1) & candidates_mask
            
            # Sort by popcount descending
            all_subs.sort(key=lambda x: bin(x).count('1'), reverse=True)
            
            for sub in all_subs:
                if time.time() > deadline:
                    return
                
                # Check validity and build new state
                new_adj = adj[:]
                new_forbidden = forbidden[:]
                
                # Add edges from v to vertices in sub
                new_adj[v] |= sub
                for u in cands:
                    if (sub >> u) & 1:
                        new_adj[u] |= (1 << v)
                
                # Forbid edges between pairs in sub
                valid = True
                sub_cands = [c for c in cands if (sub >> c) & 1]
                for i in range(len(sub_cands)):
                    u1 = sub_cands[i]
                    for j in range(i + 1, len(sub_cands)):
                        u2 = sub_cands[j]
                        if (new_adj[u1] >> u2) & 1:
                            valid = False
                            break
                        new_forbidden[u1] |= (1 << u2)
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Pruning: check if remaining potential edges can beat current best
                # Current edges so far
                current_ec = sum(bin(a).count('1') for a in new_adj[:v+1]) // 2 + \
                             sum(bin(new_adj[v]).count('1') for _ in [0])  # rough
                # Actually let's just compute properly
                current_ec = 0
                for i in range(v + 1):
                    current_ec += bin(new_adj[i]).count('1')
                current_ec //= 2
                
                # Maximum possible remaining edges (optimistic)
                remaining_vertices = list(range(v + 1, n))
                max_remaining = 0
                for i in range(len(remaining_vertices)):
                    for j in range(i + 1, len(remaining_vertices)):
                        max_remaining += 1
                # Also edges between [0..v] and [v+1..n-1] that aren't forbidden
                for i in range(v + 1):
                    for j in range(v + 1, n):
                        if not (new_forbidden[i] >> j) & 1:
                            max_remaining += 1
                
                if current_ec + max_remaining <= best_ec[0]:
                    continue
                
                backtrack(v + 1, new_adj, new_forbidden)
        
        backtrack(0, [0] * n, [0] * n)
        return best_ec[0], best_state[0]
    
    # Run the solver
    ec, adj_result = solve2()
    
    if adj_result is None:
        # Fallback: Grötzsch graph
        adj = [0] * n
        # C5: 0-1-2-3-4-0
        for i in range(5):
            u, v = i, (i + 1) % 5
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Copies 5-9
        for i in range(5):
            u = 5 + i
            v1 = (i - 1) % 5
            v2 = (i + 1) % 5
            adj[u] |= (1 << v1)
            adj[v1] |= (1 << u)
            adj[u] |= (1 << v2)
            adj[v2] |= (1 << u)
        # Top 10
        for i in range(5):
            adj[10] |= (1 << (5 + i))
            adj[5 + i] |= (1 << 10)
        adj_result = adj
        ec = edge_count(adj)
    
    return get_edge_list(adj_result)

# EVOLVE-BLOCK-END
