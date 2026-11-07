"""Rewritten program with structured greedy split-graph clique partition."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import heapq
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed, deadline):
        """
        Solve clique partition for a split graph with clique C={0..k-1} and independent set I={k..n-1}.
        
        Strategy:
        1. Build adjacency.
        2. Greedily cover edges by selecting large cliques (each containing at least one independent vertex).
        3. For each independent vertex, find the largest clique in its neighborhood within C.
        4. Use a priority queue to pick cliques that cover the most uncovered edges.
        5. Finally, cover any remaining edges among C with a single clique if possible, or small cliques.
        """
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                adj[u].add(v)
                adj[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if (u, v) in set(instance(n, k, p, seed)):  # This is inefficient; let's build edge set first
                    pass

        # Better: build edge set once
        edge_set = set()
        for u in range(k):
            for v in range(u + 1, k):
                edge_set.add((u, v))
        rng = Random(seed)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edge_set.add((u, v))

        # Rebuild adjacency from edge_set
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        # For each independent vertex v, find the largest clique in adj[v] ∩ C
        # Since C is a clique, any subset of adj[v] ∩ C is a clique.
        # The largest such clique is just adj[v] ∩ C itself.
        # But we want to cover edges efficiently.
        
        # Strategy: 
        # - Each independent vertex v must be in at least one clique with its neighbors in C.
        # - The clique {v} ∪ (adj[v] ∩ C) covers all edges from v to C.
        # - But this clique might be very large, and we might be able to do better by splitting.
        # - However, since I is independent, each clique can contain at most one vertex from I.
        # - So we need at least one clique per independent vertex that has at least one neighbor in C.
        # - Additionally, edges within C must be covered.
        
        # Let's think about it differently:
        # - The edges within C form a complete graph K_k.
        # - The edges between C and I form a bipartite graph.
        # - A clique in the split graph is either:
        #   (a) A subset of C (a clique in K_k), or
        #   (b) {v} ∪ S where v ∈ I and S ⊆ adj[v] ∩ C (S is a clique in C, which it always is).
        
        # To minimize the number of cliques:
        # - We need to cover all edges of K_k with cliques that are subsets of C, OR with cliques of type (b) that include some vertices of C.
        # - We need to cover all bipartite edges with cliques of type (b).
        
        # Key insight: Any clique of type (b) = {v} ∪ S covers:
        #   - All edges from v to S (bipartite edges)
        #   - All edges within S (edges of K_k)
        
        # So we can use cliques of type (b) to cover both bipartite edges and some K_k edges.
        
        # Greedy approach:
        # 1. For each v ∈ I, compute S_v = adj[v] ∩ C.
        # 2. We need to choose for each v a subset S_v' ⊆ S_v, and the clique will be {v} ∪ S_v'.
        # 3. The union of all S_v' must cover all vertices of C that have at least one neighbor in I (to cover bipartite edges).
        #    Actually, each bipartite edge (u, v) must be in exactly one clique, so for each v, S_v' must include all u ∈ S_v.
        #    Wait, no: each edge must be in EXACTLY one clique. So if we put (u, v) in clique {v} ∪ S_v', then we cannot put it in any other clique.
        #    Since each v appears in only one clique (because I is independent, two vertices of I cannot be in the same clique),
        #    all bipartite edges incident to v must be covered by the single clique containing v.
        #    Therefore, S_v' must equal S_v.
        
        # So the cliques of type (b) are fixed: for each v ∈ I with S_v non-empty, we must have clique {v} ∪ S_v.
        # These cliques cover:
        #   - All bipartite edges.
        #   - Some edges of K_k: specifically, for each edge (u, w) in K_k, it is covered if there exists v ∈ I such that u, w ∈ S_v.
        
        # Remaining edges: edges of K_k that are not covered by any {v} ∪ S_v.
        # These must be covered by cliques that are subsets of C.
        
        # So the algorithm is:
        # 1. For each v ∈ I with S_v non-empty, add clique {v} ∪ S_v.
        # 2. Compute the set of edges in K_k that are covered by these cliques.
        # 3. For the remaining edges in K_k, cover them with cliques that are subsets of C.
        #    Since C is a clique, the entire C is a clique, but it might include vertices that are already "used up" in the sense that
        #    some of their edges are already covered. But we can still use large cliques.
        #    Actually, we just need to cover the remaining edges. The greedy approach: repeatedly pick the largest clique
        #    (subset of C) that covers the most uncovered edges, add it, and remove those edges.
        
        # Let's implement this.
        
        cliques = []
        
        # Step 1: Add cliques for each independent vertex
        covered_edges = set()
        for v in range(k, n):
            S_v = sorted(adj[v] & set(range(k)))
            if S_v:
                clique = [v] + S_v
                cliques.append(clique)
                # Mark all edges in this clique as covered
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, w = clique[i], clique[j]
                        if u < w:
                            covered_edges.add((u, w))
                        else:
                            covered_edges.add((w, u))
        
        # Step 2: Find remaining edges in K_k
        remaining_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_edges:
                    remaining_edges.append((u, w))
        
        # Step 3: Cover remaining edges with cliques in C
        # Greedy: repeatedly pick the largest clique in C that covers the most remaining edges.
        # Since C is a clique, any subset is a clique. The largest clique is C itself, but it might cover
        # edges that are already covered. We want to cover only remaining edges.
        # Actually, we can just take the largest subset of C that has at least one remaining edge.
        # Simpler: repeatedly pick two vertices with a remaining edge, and extend to a maximal clique.
        # But since C is a clique, the maximal clique is just the set of all vertices in C that are incident to remaining edges.
        
        # Let's do: while remaining_edges:
        #   Pick the vertex u in C with the most remaining edges.
        #   Let S = {u} ∪ {w : (u, w) is a remaining edge}
        #   Actually, since C is a clique, we can take S = {all vertices in C that are incident to at least one remaining edge}.
        #   But that might be too large. Let's just take a greedy maximal clique.
        
        # Simpler greedy: 
        #   While remaining_edges:
        #     Pick (u, w) from remaining_edges.
        #     Let S = {u, w} ∪ {x in C : x is adjacent to both u and w in the remaining-edge graph}
        #     But since C is a clique, x is adjacent to u and w iff (u, x) and (w, x) are in remaining_edges or covered.
        #     This is getting complicated.
        
        # Even simpler: since C is a clique, we can cover all remaining edges with a single clique: the set of all vertices in C that are incident to at least one remaining edge.
        # But that clique might include edges that are already covered, which is fine as long as we don't double-cover.
        # Wait, the rule is: every edge must lie in EXACTLY one listed clique.
        # So we cannot include an edge in a clique if it's already covered.
        
        # So we need to cover the remaining edges with cliques that contain ONLY remaining edges (within C).
        # Since C is a clique, any subset of C is a clique, and the edges within that subset are exactly the pairs in the subset.
        # So we need to partition the remaining edges into cliques, where each clique is a set of vertices such that all pairs in the set are remaining edges.
        # This is equivalent to: the remaining edges form a graph on C, and we need to cover its edges with cliques.
        # But the remaining-edge graph is a subgraph of K_k, so it's an arbitrary graph on k vertices.
        
        # Greedy clique cover:
        remaining_set = set(remaining_edges)
        while remaining_set:
            # Pick the vertex with the highest degree in the remaining graph
            deg = {}
            for (u, w) in remaining_set:
                deg[u] = deg.get(u, 0) + 1
                deg[w] = deg.get(w, 0) + 1
            if not deg:
                break
            u = max(deg, key=lambda x: deg[x])
            # Find all vertices w such that (u, w) is in remaining_set
            neighbors = set()
            for (a, b) in remaining_set:
                if a == u:
                    neighbors.add(b)
                elif b == u:
                    neighbors.add(a)
            # The clique is {u} ∪ neighbors, but we need to ensure that all pairs in neighbors are also in remaining_set.
            # Since the remaining graph might not be a clique, we need to find a maximal clique containing u.
            # Greedy: start with {u}, then add vertices in neighbors that are connected to all current members.
            clique_set = {u}
            candidates = list(neighbors)
            for w in candidates:
                # Check if w is connected to all in clique_set
                ok = True
                for x in clique_set:
                    if x < w:
                        if (x, w) not in remaining_set:
                            ok = False
                            break
                    else:
                        if (w, x) not in remaining_set:
                            ok = False
                            break
                if ok:
                    clique_set.add(w)
            clique_list = sorted(clique_set)
            cliques.append(clique_list)
            # Remove all edges in this clique from remaining_set
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    if a < b:
                        remaining_set.discard((a, b))
                    else:
                        remaining_set.discard((b, a))
        
        return cliques

    def validate(cliques, n, k, p, seed):
        """Validate that the clique partition is correct."""
        edge_set = set()
        for u in range(k):
            for v in range(u + 1, k):
                edge_set.add((u, v))
        rng = Random(seed)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edge_set.add((u, v))
        
        # Check each clique is a valid clique with at least 2 vertices
        covered = set()
        for clique in cliques:
            if len(clique) < 2:
                return False
            # Check all pairs are edges
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    if u < v:
                        e = (u, v)
                    else:
                        e = (v, u)
                    if e not in edge_set:
                        return False
                    if e in covered:
                        return False
                    covered.add(e)
        
        # Check all edges are covered
        if covered != edge_set:
            return False
        
        return True

    results = []
    t0 = time.time()
    deadline = t0 + 110  # 10-second safety margin
    
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback: return seed-like result (2-cliques) for remaining instances
            for remaining_spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))[len(results):]:
                n, k, p, seed = remaining_spec
                edges = instance(n, k, p, seed)
                results.append([list(e) for e in edges])
            break
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed, deadline)
        # Validate
        if not validate(cliques, n, k, p, seed):
            # Fallback
            edges = instance(n, k, p, seed)
            cliques = [list(e) for e in edges]
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
