# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct():
    n = 15
    all_edges = list(itertools.combinations(range(n), 2))
    m = len(all_edges)  # 105
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    time_budget = 55.0

    # Precompute neighbor sets for each vertex
    neighbors = [set() for _ in range(n)]
    for (u, v) in all_edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Build conflict graph: vertices = potential edges (0..m-1)
    # Two potential edges (u,v) and (u',v') conflict if they cannot both be in
    # any valid graph without creating a forbidden 3-path.
    #
    # Forbidden: path s-u-v-t (3 edges) where deg(s) == deg(t) in the FULL graph.
    # When we add edges incrementally, we can only detect conflicts that are
    # independent of other edges. The key structural conflict:
    #
    # If we have edges (a,b) and (a,c) sharing vertex a, and edges (d,b) and (d,c)
    # sharing vertex d, then path d-b-a-c or d-c-a-b exists. This creates a
    # forbidden 3-path if deg(d) == deg(c) or deg(d) == deg(b).
    #
    # More precisely: for ANY 4-clique of edges forming a "ladder" or "path of
    # length 3", we need endpoints to have different degrees.
    #
    # For a static conflict (independent of other edges), the strongest guarantee:
    # Two edges (u,v) and (u',v') CONFLICT if there exists a vertex w such that
    # w is adjacent to both u and v' (or v and u'), creating a 3-path where the
    # two "free" endpoints would need to have different degrees, but the shared
    # structure forces a specific degree relationship.
    #
    # Actually, the cleanest static conflict: consider the 3-path s-u-v-t.
    # Edges involved: (s,u), (u,v), (v,t). If we include ALL THREE, then s and t
    # must have different degrees. But degrees depend on the whole graph.
    #
    # Static conflict that doesn't depend on degrees: if edges (s,u), (u,v), (v,t)
    # are all included AND s==t (impossible in simple graph for distinct s,u,v,t)
    # or if the structure is forced.
    #
    # Let me think differently. The REAL static conflict:
    # If we have a triangle (a,b), (b,c), (c,a), then for any vertex d connected
    # to a: path d-a-b-c is a 3-path with endpoints d and c. So deg(d) != deg(c).
    # Similarly d-a-c-b gives deg(d) != deg(b).
    #
    # But this depends on degrees, not a pure static conflict.
    #
    # The truly static conflict: if including edges (e1, e2) creates a 3-path
    # where the two endpoints are the SAME vertex, which is impossible.
    #
    # Let me reconsider. For a MAXIMUM CLIQUE approach, I need: if edges e1 and
    # e2 are both in the graph, they must NOT create a forbidden 3-path REGARDLESS
    # of other edges. This is hard to satisfy statically.
    #
    # Alternative: use a relaxed conflict where two edges conflict if they are
    # "close" in the path structure, then use branch-and-bound with a dynamic
    # validity check.
    #
    # Better approach: Build conflict as: two edges (u,v) and (a,b) conflict if
    # they share a vertex AND the other two vertices are also connected in a way
    # that creates a forced 3-path. Specifically:
    #
    # Edge (u,v) and edge (v,w) share vertex v. If we also have edges (u,x) and
    # (w,x) for some x, then path x-u-v-w is forbidden if deg(x)==deg(w).
    #
    # For a STATIC conflict independent of degrees:
    # Two edges (u,v) and (u,w) (sharing vertex u) CONFLICT with edges (x,v) and
    # (x,w) (sharing vertex x) because together they form a 3-path x-v-u-w with
    # endpoints x and w that would need different degrees, but if x==w it's
    # impossible (not in simple graph).
    #
    # The key insight for static conflicts: if two edges share BOTH endpoints
    # (i.e., they're the same edge - not possible) or if their union creates a
    # K4 or similar where degree constraints are overdetermined.
    #
    # Let me use a different strategy: the conflict is that edges (u,v) and (a,b)
    # are incompatible if there's a 3-path using them as the "middle" edges where
    # the endpoints would be forced to have the same degree by the structure.
    #
    # Actually, the simplest non-trivial static conflict:
    # If edges (a,b) and (a,c) are both present, and edges (d,b) and (d,c) are
    # both present, then path d-b-a-c exists. The endpoints are d and c.
    # For this to be a CONFLICT (always forbidden), we'd need deg(d) == deg(c)
    # always, which isn't true.
    #
    # I'll use a HYBRID approach: build a sparse conflict graph where edges
    # conflict if they form a "2-path" (share a vertex) with specific structure,
    # then use branch-and-bound with DYNAMIC validity checking.

    # Build conflict graph conservatively: two potential edges conflict if
    # they share a vertex AND the other endpoints are adjacent (creating a
    # triangle, which creates many 3-paths).
    #
    # Actually, let's be more aggressive: two edges (u,v) and (a,b) conflict if
    # they share a vertex. This is very conservative but ensures we don't miss
    # conflicts. No wait, that's too conservative - sharing a vertex is fine.
    #
    # Let me think about what ACTUALLY creates a forbidden 3-path:
    # A 3-path s-u-v-t uses edges (s,u), (u,v), (v,t). This is forbidden iff
    # deg(s) == deg(t).
    #
    # For a STATIC conflict between two edges e1=(a,b) and e2=(c,d):
    # They conflict if INCLUDING BOTH makes it IMPOSSIBLE to avoid a forbidden
    # 3-path regardless of other edges.
    #
    # The only way this happens: if the 3-path is forced to have s==t (impossible)
    # or if the degree constraint is unsatisfiable.
    #
    # With 15 vertices and degrees 0-14, it's hard to force a static conflict
    # between just two edges.
    #
    # NEW APPROACH: Use branch-and-bound where at each step we check if adding
    # an edge creates a forbidden 3-path GIVEN the current partial solution.
    # This is dynamic, not static. We use the conflict graph as a pruning aid.

    # For efficiency, let's precompute for each pair of edges whether they can
    # coexist in ANY valid graph. Two edges (u,v) and (a,b) can coexist unless
    # they create an immediate structural impossibility.
    #
    # The simplest impossibility: if u==a and v==b (same edge) - not a conflict.
    # If the four endpoints form a structure where a 3-path is forced with
    # identical endpoints - impossible in simple graph.
    #
    # I'll build the conflict graph as: edge e1 and e2 conflict if they are
    # the same edge (trivial) or if including both creates a K4 where degree
    # constraints are contradictory.
    #
    # For K4 on vertices {a,b,c,d}: all 6 edges present.
    # 3-paths: a-b-c-d (endpoints a,d), a-b-d-c (a,c), a-c-b-d (a,d), etc.
    # In K4, all 4 vertices have degree 3, so ALL 3-paths are forbidden!
    # Therefore, NO two edges that are part of a common K4 can coexist...
    # wait, that means we can't have a K4 at all.
    #
    # More precisely: if we have a 4-clique, every 3-path within it is forbidden
    # because all 4 vertices have the same degree (within the clique, but degree
    # in the full graph may differ).
    #
    # So the static conflict: two edges conflict if they are in a common
    # 4-set of vertices where all 6 edges could be present. But we don't know
    # if all 6 will be present.
    #
    # Let me use: two edges (u,v) and (a,b) conflict if {u,v,a,b} has 4 distinct
    # vertices and the 4 edges (u,a),(u,b),(v,a),(v,b) are all potential edges
    # (they always are). This doesn't help.
    #
    # FINAL APPROACH: Use branch-and-bound with dynamic validity checking.
    # The "conflict graph" for pruning: two edges are in conflict if they share
    # a vertex and the other two vertices are adjacent. This creates a triangle
    # with a pendant, which creates 3-paths.
    #
    # Actually, I'll just do a smart branch-and-bound with dynamic checking.

    # Precompute: for each edge (u,v), the set of "partner" edges that with (u,v)
    # create a 3-path. Edge (u,v) participates in 3-paths as the middle edge:
    # s-u-v-t where s in N(u)\{v}, t in N(v)\{u,s}.
    #
    # For branch-and-bound, I'll order edges and use bitmasks for efficiency.

    # Let's use a different strategy: greedy with smart ordering + local search,
    # but enhanced with the insight that we should prioritize edges that create
    # FEWER 3-paths.

    # Edge centrality: number of 3-paths an edge participates in as middle edge.
    edge_centrality = []
    for (u, v) in all_edges:
        # 3-paths s-u-v-t: s in N(u)\{v}, t in N(v)\{u,s}
        s_candidates = [x for x in neighbors[u] if x != v]
        count = 0
        for s in s_candidates:
            t_candidates = [x for x in neighbors[v] if x != u and x != s]
            count += len(t_candidates)
        edge_centrality.append(count)

    # Also compute: for each edge, how many OTHER edges it would "conflict" with
    # if added (i.e., how many 3-paths it creates with existing edges)
    # This is dynamic, so we compute it during search.

    # Sort edges by centrality (ascending) - add less-central edges first
    edge_order = sorted(range(m), key=lambda i: edge_centrality[i])

    # For the branch-and-bound, we'll use a recursive search with pruning.
    # State: current set of edges (as bitmask), current validity.
    # We add edges in a specific order and check validity incrementally.

    # For efficiency with 105 edges, we need to be smart.
    # Let's use a heuristic: build the graph greedily, then do local search
    # with edge swaps and additions.

    # Greedy construction with degree-awareness:
    def build_greedy(order, seed):
        rng = random.Random(seed)
        adj = [0] * n
        degs = [0] * n
        in_graph = [False] * m

        def bit(i):
            return 1 << i

        def is_valid_add(u, v):
            # Check if adding edge (u,v) creates a forbidden 3-path
            # given current adj and degs (BEFORE adding the edge)
            # New edges involving u or v might create 3-paths.
            # 3-paths where (u,v) is the middle edge: s-u-v-t
            s_candidates = [x for x in range(n) if (adj[u] >> x) & 1 and x != v]
            for s in s_candidates:
                t_candidates = [x for x in range(n) if (adj[v] >> x) & 1 and x != u and x != s]
                for t in t_candidates:
                    if degs[s] == degs[t]:
                        return False
            # 3-paths where (u,v) is an end edge: u-v-x-y where (v,x) exists
            # This creates path u-v-x-y with endpoints u,y
            for x in range(n):
                if (adj[v] >> x) & 1 and x != u:
                    for y in range(n):
                        if (adj[x] >> y) & 1 and y != v and y != u:
                            if degs[u] == degs[y]:
                                return False
            # 3-paths where (u,v) is an end edge: y-x-v-u where (x,v) exists
            for x in range(n):
                if (adj[u] >> x) & 1 and x != v:
                    for y in range(n):
                        if (adj[x] >> y) & 1 and y != u and y != v:
                            if degs[y] == degs[v]:
                                return False
            # 3-paths where (u,v) is NOT involved but the degree change creates
            # a new forbidden 3-path: edges (s,w) and (t,w) where deg changes
            # This is complex; we'll check the full validity after adding.
            return True

        def full_check():
            # Check all 3-paths in the current graph
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        continue
                    S_mask = adj[u] & ~(1 << v)
                    T_mask = adj[v] & ~(1 << u)
                    # For each degree d, check if there's s in S with deg d
                    # and t in T with deg d, s != t
                    deg_groups = {}
                    for x in range(n):
                        d = degs[x]
                        if d not in deg_groups:
                            deg_groups[d] = 0
                        deg_groups[d] |= (1 << x)
                    for d, mask in deg_groups.items():
                        A = S_mask & mask
                        B = T_mask & mask
                        if A and B:
                            if not (A == B and (A & (A - 1)) == 0):
                                return False
            return True

        edges_in_graph = []
        # Try adding edges in the given order
        for eidx in order:
            (u, v) = all_edges[eidx]
            if in_graph[eidx]:
                continue
            if is_valid_add(u, v):
                # Tentatively add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
                in_graph[eidx] = True
                edges_in_graph.append(eidx)
                if not full_check():
                    # Rollback
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    degs[u] -= 1
                    degs[v] -= 1
                    in_graph[eidx] = False
                    edges_in_graph.pop()
                    continue

        return edges_in_graph, adj, degs

    # Local search: try to add more edges, or swap edges
    def local_search(edges_in_graph, adj, degs, max_iters=500):
        in_graph = [False] * m
        for eidx in edges_in_graph:
            in_graph[eidx] = True

        def bit(i):
            return 1 << i

        def full_check_adj(adj_check, degs_check):
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj_check[u] >> v) & 1):
                        continue
                    S_mask = adj_check[u] & ~(1 << v)
                    T_mask = adj_check[v] & ~(1 << u)
                    deg_groups = {}
                    for x in range(n):
                        d = degs_check[x]
                        if d not in deg_groups:
                            deg_groups[d] = 0
                        deg_groups[d] |= (1 << x)
                    for d, mask in deg_groups.items():
                        A = S_mask & mask
                        B = T_mask & mask
                        if A and B:
                            if not (A == B and (A & (A - 1)) == 0):
                                return False
            return True

        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            # Try adding each missing edge
            for eidx in range(m):
                if in_graph[eidx]:
                    continue
                (u, v) = all_edges[eidx]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
                if full_check_adj(adj, degs):
                    in_graph[eidx] = True
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    degs[u] -= 1
                    degs[v] -= 1

        return edges_in_graph, adj, degs, in_graph

    # Run multiple trials with different orderings
    best_edges = []
    best_count = 0

    random.seed(12345)

    for trial in range(50):
        # Create a randomized ordering
        order = list(edge_order)
        random.shuffle(order)
        # Mix: some in centrality order, some random
        if trial % 3 == 0:
            order = edge_order[:]  # strict centrality order
        elif trial % 3 == 1:
            order = list(range(m))
            random.shuffle(order)
        else:
            # Hybrid
            order = list(edge_order)
            random.shuffle(order[:m//2])

        edges_in_graph, adj, degs = build_greedy(order, trial)
        edges_in_graph, adj, degs, in_graph = local_search(edges_in_graph, adj, degs)

        if len(edges_in_graph) > best_count:
            best_count = len(edges_in_graph)
            best_edges = edges_in_graph[:]

        if best_count >= 20:  # If we're doing well, stop early
            break

    # Additional local search on best solution with different heuristics
    if best_count > 0:
        in_graph = [False] * m
        for eidx in best_edges:
            in_graph[eidx] = True
        adj = [0] * n
        degs = [0] * n
        for eidx in best_edges:
            (u, v) = all_edges[eidx]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1

        # Try edge swaps: remove one edge, add another
        improved = True
        while improved and best_count < 25:
            improved = False
            for eidx in range(m):
                if in_graph[eidx]:
                    continue
                (u, v) = all_edges[eidx]
                # Try removing each existing edge and adding this one
                for eidx2 in range(m):
                    if not in_graph[eidx2]:
                        continue
                    (u2, v2) = all_edges[eidx2]
                    # Remove eidx2
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
                    degs[u2] -= 1
                    degs[v2] -= 1
                    in_graph[eidx2] = False
                    # Add eidx
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degs[u] += 1
                    degs[v] += 1
                    in_graph[eidx] = True

                    if full_check_adj(adj, degs):
                        best_edges = [i for i in range(m) if in_graph[i]]
                        best_count = len(best_edges)
                        improved = True
                        # Continue searching with this new state
                        break
                    else:
                        # Rollback
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        degs[u] -= 1
                        degs[v] -= 1
                        in_graph[eidx] = False
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        degs[u2] += 1
                        degs[v2] += 1
                        in_graph[eidx2] = True

                    if improved:
                        break
                if improved:
                    break

    # Build result
    result = []
    for eidx in best_edges:
        (u, v) = all_edges[eidx]
        result.append([int(u), int(v)])

    return result
# EVOLVE-BLOCK-END
