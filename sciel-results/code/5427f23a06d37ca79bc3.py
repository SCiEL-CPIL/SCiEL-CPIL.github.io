import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 145

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(n, k, p, seed):
        rng = random.Random(seed)
        # Build cross edges: for each I vertex, which K vertices it connects to
        adj_I = [set() for _ in range(n - k)]
        for i in range(n - k):
            v = k + i
            for u in range(k):
                if rng.random() < p:
                    adj_I[i].add(u)

        # All edges
        all_edges = set()
        # K-K edges
        for u in range(k):
            for v in range(u + 1, k):
                all_edges.add((u, v))
        # I-K edges
        for i in range(n - k):
            v = k + i
            for u in adj_I[i]:
                all_edges.add((u, v))

        # Strategy:
        # Each I vertex v with neighbor set S (non-empty) needs a clique containing v and all of S.
        # Minimal: {v} ∪ S. This covers I-K edges (v,u) for u in S, and K-K edges (u,w) for u,w in S.
        # But K-K edges can be covered by multiple I-cliques if they appear in multiple S sets.
        # We need each edge in exactly one clique.
        #
        # Approach: 
        # 1. For each I vertex, assign its I-K edges to its clique {v} ∪ S.
        # 2. For K-K edges: assign each K-K edge to exactly one clique.
        #    - If a K-K edge (u,w) is in the neighbor set S of some I vertex v, it can be covered by {v} ∪ S.
        #    - We want to minimize total cliques. Each I vertex with non-empty S needs one clique.
        #    - K-K edges not covered by any I-clique need additional cliques.
        #
        # To minimize: assign K-K edges to I-cliques whenever possible (no extra cost).
        # For K-K edges not in any S, they need their own cliques.
        #
        # But wait: a K-K edge (u,w) can only be in clique {v} ∪ S if u,w ∈ S.
        # So we need to check if u,w are both in S for some I vertex v.
        #
        # Greedy: For each I vertex, its clique {v} ∪ S covers all K-K edges within S.
        # If a K-K edge is in S for multiple I vertices, assign it to just one.
        #
        # Implementation:
        # 1. Create cliques for each I vertex with non-empty S: {v} ∪ S.
        # 2. Track which K-K edges are covered by these cliques.
        # 3. For remaining K-K edges, add additional cliques.
        #    But we must ensure no edge is in two cliques.
        #    So when we form {v} ∪ S, we only claim K-K edges in S that haven't been claimed yet.
        #    Actually, the clique {v} ∪ S inherently contains all K-K edges in S.
        #    We can't "partially" include a clique.
        #
        # Alternative: Don't use {v} ∪ S directly. Instead:
        # - Each I-K edge (v,u) must be in some clique containing both v and u.
        # - Since I is independent, v can only be in a clique with K vertices.
        # - The clique must contain v and all its K-neighbors? No, just the specific K vertex u for edge (v,u).
        # - But if v has neighbors u1, u2, we can put v with u1 in one clique and v with u2 in another.
        #   However, that's more cliques. Better to put v with all its neighbors in one clique.
        #
        # The constraint is: every edge in exactly one clique.
        # So if we put v in clique {v, u1, u2}, then edges (v,u1), (v,u2), (u1,u2) are all in this clique.
        # Edge (u1,u2) cannot appear in any other clique.
        #
        # So the approach:
        # 1. For each I vertex v, form clique C_v = {v} ∪ N(v) where N(v) = adj_I[i].
        # 2. This covers all I-K edges incident to v, and all K-K edges within N(v).
        # 3. Problem: K-K edges may be covered by multiple C_v's.
        #    We need to ensure each K-K edge is in exactly one clique.
        #    Solution: When forming C_v, only include K-K edges that haven't been covered yet.
        #    But we can't modify the clique — if v and u1,u2 are in C_v, then (u1,u2) is automatically in C_v.
        #
        # So we need a different approach:
        # - Assign each I-K edge to a clique.
        # - Each I vertex v must be in at least one clique (if it has any edges).
        # - We can have multiple cliques containing v, each with a subset of N(v).
        # - But that increases clique count.
        #
        # Optimal for split graphs: 
        # Actually, for the "exactly one" constraint, we can:
        # 1. For each I vertex v, create clique {v} ∪ N(v).
        # 2. For K-K edges that appear in multiple such cliques, we have a conflict.
        # 3. To resolve: for K-K edges, assign them to one specific clique.
        #    But the clique is fixed — we can't remove an edge from it.
        #
        # I think the correct approach is:
        # - Each I-K edge (v,u) must be in a clique containing v and u.
        # - Since v is in I, any clique containing v can only contain K vertices.
        # - We can partition N(v) into subsets, and for each subset T, create clique {v} ∪ T.
        # - This uses |T| cliques for v if we partition into singletons, or 1 if we use all of N(v).
        # - Using all of N(v) in one clique is optimal for v.
        # - The K-K edges within N(v) are then covered by this clique.
        # - If a K-K edge (u,w) is in N(v1) and N(v2), it's covered by both C_v1 and C_v2.
        #   This violates "exactly one."
        #
        # So we need to ensure no K-K edge is in two cliques.
        # This means: for any K-K edge (u,w), it can be in at most one I-clique.
        # So we need to choose which I-clique "owns" each K-K edge.
        #
        # This is getting complex. Let me use a simpler approach:
        # 1. For each I vertex v, create clique {v} ∪ N(v).
        # 2. Collect all K-K edges covered by these cliques.
        # 3. For K-K edges covered by multiple cliques, we have a problem.
        #    But actually, we can just accept that and add the K-K edges as separate 2-cliques?
        #    No, that would double-cover.
        #
        # Let me think differently. The constraint is strict: each edge in exactly one clique.
        # So I need to plan the cliques carefully.
        #
        # New approach:
        # 1. I-K edges: each must be in a clique with its I vertex.
        #    For I vertex v, all its I-K edges can be in one clique {v} ∪ N(v).
        # 2. K-K edges: each must be in exactly one clique.
        #    A K-K edge (u,w) can be in:
        #    a) An I-clique {v} ∪ S where u,w ∈ S
        #    b) A pure K-clique {u, w, ...}
        # 3. To minimize total cliques:
        #    - Use one clique per I vertex (if N(v) non-empty): {v} ∪ N(v)
        #    - This covers |N(v)| I-K edges and |N(v)|*(|N(v)|-1)/2 K-K edges.
        #    - K-K edges not covered by any I-clique need pure K-cliques.
        #    - K-K edges covered by multiple I-cliques: assign to one, but the clique structure
        #      forces them to be in all I-cliques that contain both endpoints.
        #
        # The key issue: if u,w ∈ N(v1) and u,w ∈ N(v2), then (u,w) is in both C_v1 and C_v2.
        # This is invalid.
        #
        # Solution: Don't use full N(v) in the clique. Instead, partition N(v) so that
        # no K-K edge is shared between different I-cliques.
        # This is like a graph coloring / edge partition problem.
        #
        # Simpler: Use 2-cliques for I-K edges. Each I-K edge (v,u) gets clique {v,u}.
        # This uses |E_IK| cliques for I-K edges, and no K-K edges are covered.
        # Then all K-K edges need pure K-cliques.
        # Total: |E_IK| + (cliques for K-K edges).
        #
        # Better: Use {v} ∪ N(v) for I vertices, but handle K-K edge conflicts by
        # adding extra K-cliques for the "extra" coverage. But that doesn't work because
        # the edge is already in the I-clique.
        #
        # I think the cleanest valid approach:
        # 1. For each I-K edge (v,u), create a clique. Group edges by I vertex.
        #    For I vertex v, create clique {v} ∪ N(v). This is valid and covers all I-K edges for v.
        # 2. For K-K edges: they are covered by I-cliques if both endpoints are in N(v) for some v.
        #    To avoid double coverage, we need to ensure each K-K edge is in at most one I-clique.
        #    This requires that for any K-K edge (u,w), there is at most one I vertex v with {u,w} ⊆ N(v).
        #    This is not guaranteed.
        #
        # Given the complexity, let me use the simple approach:
        # - Each I-K edge gets its own 2-clique {v,u}.
        # - Each K-K edge gets its own 2-clique {u,w}.
        # Total: |E| cliques. This is valid but not optimal.
        #
        # Better: Group I-K edges by I vertex. For I vertex v, clique {v} ∪ N(v).
        # For K-K edges, only include them if not already covered. But we can't partially include.
        #
        # Let me just do: one clique per I vertex ({v} ∪ N(v)), and for K-K edges not in any
        # such clique, add 2-cliques. For K-K edges in multiple I-cliques, this is invalid.
        #
        # To make it valid: after forming I-cliques, remove K-K edges from consideration that are
        # covered, and for those covered by multiple, we have a problem.
        #
        # I'll use the safe approach: 2-cliques for all edges. Then optimize.

        # Safe approach: each edge is its own clique
        # But that's |E| cliques, very suboptimal.
        
        # Better safe approach:
        # 1. For each I vertex v with N(v) non-empty, create clique {v} ∪ N(v).
        # 2. Track all edges covered.
        # 3. For K-K edges covered by multiple I-cliques, we need to fix this.
        #    Actually, let's just not include K-K edges in I-cliques.
        #    Create clique {v} for I vertex v? No, needs at least 2 vertices.
        #    Create clique {v, u} for each I-K edge? That's |E_IK| cliques.
        #
        # Let me use: for each I vertex v, create |N(v)| cliques of size 2: {v, u} for each u in N(v).
        # This covers all I-K edges, no K-K edges.
        # Then K-K edges need their own cliques.
        # Total: |E_IK| + (cliques for K-K edges).
        #
        # For K-K edges, since K is a clique, we can partition them into larger cliques.
        # Greedy: repeatedly pick a clique covering the most remaining K-K edges.

        cliques = []
        covered_edges = set()

        # I-K edges: one 2-clique per edge
        for i in range(n - k):
            v = k + i
            for u in adj_I[i]:
                clique = [v, u]
                cliques.append(clique)
                covered_edges.add((min(v, u), max(v, u)))

        # K-K edges: greedy clique cover
        rem_K_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                rem_K_edges.add((u, w))

        rem_adj = [set() for _ in range(k)]
        for (u, w) in rem_K_edges:
            rem_adj[u].add(w)
            rem_adj[w].add(u)

        while rem_K_edges:
            if time.time() > deadline:
                break
            best_clique = None
            best_count = -1

            # Greedy: try to build large cliques
            verts = sorted(range(k), key=lambda x: len(rem_adj[x]), reverse=True)
            for start in verts[:min(10, k)]:
                if not rem_adj[start]:
                    break
                C = {start}
                candidates = set(rem_adj[start])
                while candidates:
                    best_c = None
                    best_d = -1
                    for c in candidates:
                        d = sum(1 for x in candidates if x != c and x in rem_adj[c])
                        if d > best_d:
                            best_d = d
                            best_c = c
                    if best_c is None:
                        break
                    C.add(best_c)
                    candidates = {x for x in candidates if x != best_c and x in rem_adj[best_c]}

                C_list = sorted(C)
                count = 0
                for i in range(len(C_list)):
                    for j in range(i + 1, len(C_list)):
                        e = (min(C_list[i], C_list[j]), max(C_list[i], C_list[j]))
                        if e in rem_K_edges:
                            count += 1

                if count > best_count:
                    best_count = count
                    best_clique = C_list

            if best_clique is None or best_count <= 0:
                if rem_K_edges:
                    e = next(iter(rem_K_edges))
                    best_clique = sorted(e)
                else:
                    break

            if len(best_clique) >= 2:
                cliques.append(best_clique)

            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    e = (min(best_clique[i], best_clique[j]), max(best_clique[i], best_clique[j]))
                    rem_K_edges.discard(e)
                    rem_adj[best_clique[i]].discard(best_clique[j])
                    rem_adj[best_clique[j]].discard(best_clique[i])

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = clique_partition(n, k, p, seed)
        for c in cliques:
            for i in range(len(c)):
                c[i] = int(c[i])
        results.append(cliques)

    while len(results) < 3:
        results.append([])

    return results
