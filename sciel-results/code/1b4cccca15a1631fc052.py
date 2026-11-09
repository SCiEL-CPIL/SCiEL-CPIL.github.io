import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n = 20
    range_n = range(n)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(all_edges)
    
    # Precompute edge index for fast lookup
    edge_idx = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
    
    # For each pair (u,v), precompute which other vertices are "between" them
    # i.e., for pair (u,v), the vertices w that would be common neighbors
    # Actually, common neighbors of u and v are vertices w adjacent to both.
    # When we add/remove edge (u,v), it affects common-neighbor counts of pairs (u,w) and (v,w) for all w.
    
    def count_common_neighbors(adj, u, v):
        common = adj[u] & adj[v]
        return common.bit_count()
    
    def is_k33_free(adj):
        for u in range_n:
            for v in range(u + 1, n):
                if count_common_neighbors(adj, u, v) >= 3:
                    return False
        return True
    
    def get_violations(adj):
        """Return list of violating pairs (u,v) where common neighbors >= 3."""
        violations = []
        for u in range_n:
            for v in range(u + 1, n):
                if count_common_neighbors(adj, u, v) >= 3:
                    violations.append((u, v))
        return violations
    
    def edge_removal_impact(adj, u, v):
        """
        If we remove edge (u,v), how many violations does this fix?
        Removing edge (u,v) means u is no longer a neighbor of v and vice versa.
        This affects common-neighbor counts for pairs (u,w) and (v,w) for all w.
        For pair (u,w): v was a common neighbor if w~v. After removal, v is no longer in N(u), so v is no longer common.
        Actually, common neighbors of (u,w) are vertices in N(u)∩N(w). Removing edge (u,v) removes v from N(u), so if v was in N(w), it's no longer common.
        Similarly for (v,w).
        """
        fixed = 0
        # For each w != u, v:
        for w in range_n:
            if w == u or w == v:
                continue
            # Check pair (u, w): was v a common neighbor? v in N(w) and v in N(u) (yes, since edge exists)
            if (adj[w] >> v) & 1:
                # v was in N(u) and N(w), so it was a common neighbor of (u,w)
                # After removal, v no longer in N(u), so common count decreases by 1
                # This fixes a violation if the count was exactly 3
                common_uw = (adj[u] & adj[w])
                cnt = common_uw.bit_count()
                # v is in this common set since v in adj[u] and v in adj[w]
                # After removal, cnt decreases by 1
                if cnt == 3:
                    fixed += 1
            # Check pair (v, w): was u a common neighbor?
            if (adj[w] >> u) & 1:
                common_vw = (adj[v] & adj[w])
                cnt = common_vw.bit_count()
                if cnt == 3:
                    fixed += 1
        return fixed
    
    def prune_to_k33_free(adj, edges):
        """
        Starting from a graph (possibly with K3,3 violations), iteratively remove edges
        to eliminate all violations. Greedy: remove the edge whose removal fixes the most violations.
        """
        adj = adj[:]
        edge_set = set((min(e), max(e)) for e in edges)
        
        while True:
            violations = get_violations(adj)
            if not violations:
                break
            
            # Find the edge whose removal fixes the most violations
            best_edge = None
            best_fix = -1
            
            # Candidate edges: those incident to at least one violating pair
            candidate_edges = set()
            for u, v in violations:
                for w in range_n:
                    if w != u and (u, w) in edge_set or (w, u) in edge_set:
                        candidate_edges.add((min(u, w), max(u, w)))
                    if w != v and (v, w) in edge_set or (w, v) in edge_set:
                        candidate_edges.add((min(v, w), max(v, w)))
            
            for (x, y) in candidate_edges:
                fix = edge_removal_impact(adj, x, y)
                if fix > best_fix:
                    best_fix = fix
                    best_edge = (x, y)
            
            if best_edge is None or best_fix <= 0:
                # No single edge removal fixes a violation? This shouldn't happen if violations exist.
                # As a fallback, remove an arbitrary edge from a violating pair's neighborhood
                u, v = violations[0]
                # Remove an edge incident to u or v
                for w in range_n:
                    if w != u and ((u, w) in edge_set or (w, u) in edge_set):
                        best_edge = (min(u, w), max(u, w))
                        break
                    if w != v and ((v, w) in edge_set or (w, v) in edge_set):
                        best_edge = (min(v, w), max(v, w))
                        break
                if best_edge is None:
                    break
            
            # Remove the best edge
            x, y = best_edge
            adj[x] &= ~(1 << y)
            adj[y] &= ~(1 << x)
            edge_set.discard((x, y))
            edge_set.discard((y, x))
        
        return adj, list(edge_set)
    
    def try_add_edges(adj, edge_set):
        """Try to add any missing edges that keep the graph K3,3-free."""
        improved = True
        while improved:
            improved = False
            for (u, v) in all_edges:
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                # Check if adding (u,v) keeps K3,3-free
                ok = True
                for w in range_n:
                    if w == u or w == v:
                        continue
                    # Check pair (u, w)
                    common_uw = adj[u] & adj[w]
                    cnt = common_uw.bit_count()
                    if (adj[w] >> v) & 1:
                        cnt += 1
                    if cnt >= 3:
                        ok = False
                        break
                    if not ok:
                        break
                    # Check pair (v, w)
                    common_vw = adj[v] & adj[w]
                    cnt2 = common_vw.bit_count()
                    if (adj[w] >> u) & 1:
                        cnt2 += 1
                    if cnt2 >= 3:
                        ok = False
                        break
                if ok:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    improved = True
        return adj, edge_set
    
    def swap_search(adj, edge_set, max_swaps=100):
        """
        Try to improve by swapping: remove edge (a,b), add edge (c,d) if the result is K3,3-free
        and has the same or more edges. Actually, we want to maintain edge count while reducing violations,
        but since we start K3,3-free, we just try to add edges.
        
        Better: try removing one edge and adding another to potentially enable more edges later.
        For simplicity, just do local search: try adding any valid edge.
        """
        # Just call try_add_edges
        adj, edge_set = try_add_edges(adj, edge_set)
        return adj, edge_set
    
    best_edges = []
    best_count = 0
    
    # Strategy: Start from dense random graphs, prune to K3,3-free, then fill in.
    # Also try starting from the complete graph and pruning.
    
    # Phase 1: Start from complete graph, prune
    adj_full = [0] * n
    for i in range_n:
        for j in range(i+1, n):
            adj_full[i] |= (1 << j)
            adj_full[j] |= (1 << i)
    
    edges_full = all_edges[:]
    adj_pruned, edges_pruned = prune_to_k33_free(adj_full, edges_full)
    adj_pruned, edges_pruned_set = try_add_edges(adj_pruned, set((min(e), max(e)) for e in edges_pruned))
    count = len(edges_pruned_set)
    if count > best_count:
        best_count = count
        best_edges = [[u, v] for (u, v) in edges_pruned_set]
    
    # Phase 2: Multiple random dense starts, prune, fill
    while time.time() < deadline - 10:
        # Create a random dense graph
        adj = [0] * n
        edges = []
        edge_set = set()
        
        # Start with a random subset of edges, biased toward dense
        density = random.uniform(0.5, 0.9)
        for (u, v) in all_edges:
            if random.random() < density:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                edge_set.add((u, v))
        
        # Prune to K3,3-free
        adj_p, edges_p = prune_to_k33_free(adj, edges)
        
        # Fill in
        adj_p, edge_set_p = try_add_edges(adj_p, set((min(e), max(e)) for e in edges_p))
        count = len(edge_set_p)
        
        if count > best_count:
            best_count = count
            best_edges = [[u, v] for (u, v) in edge_set_p]
    
    # Final: try to add more edges to the best solution
    if best_edges:
        adj_best = [0] * n
        edge_set_best = set()
        for (u, v) in best_edges:
            adj_best[u] |= (1 << v)
            adj_best[v] |= (1 << u)
            edge_set_best.add((u, v))
        adj_best, edge_set_best = try_add_edges(adj_best, edge_set_best)
        if len(edge_set_best) > best_count:
            best_count = len(edge_set_best)
            best_edges = [[u, v] for (u, v) in edge_set_best]
    
    return best_edges
