def construct():
    import time
    from random import Random
    from itertools import combinations

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _max_matching_line_graph(n, edges):
        """Find maximum matching in the line graph of a graph with n vertices and given edges.
        Returns a list of matched edge-pairs (each pair is a tuple of two edges sharing a vertex)."""
        if not edges:
            return []
        # Build line graph adjacency: two edges are adjacent if they share a vertex
        edge_index = {e: i for i, e in enumerate(edges)}
        m = len(edges)
        adj = [set() for _ in range(m)]
        # Group edges by vertex for efficient line graph construction
        vertex_edges = {}
        for i, (u, v) in enumerate(edges):
            vertex_edges.setdefault(u, []).append(i)
            vertex_edges.setdefault(v, []).append(i)
        for v, elist in vertex_edges.items():
            for a in range(len(elist)):
                for b in range(a + 1, len(elist)):
                    adj[elist[a]].add(elist[b])
                    adj[elist[b]].add(elist[a])
        
        # Find maximum matching using augmenting paths (Hopcroft-Karp style for general graph is complex,
        # but for small graphs we can use simple augmenting path search)
        match = [-1] * m  # match[i] = j means edge i is matched with edge j
        def try_augment(u, visited):
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    if match[v] == -1 or try_augment(match[v], visited):
                        match[u] = v
                        match[v] = u
                        return True
            return False
        
        result = []
        for i in range(m):
            if match[i] == -1:
                if try_augment(i, set()):
                    pass
        # Collect matched pairs
        seen = set()
        for i in range(m):
            if match[i] != -1 and i < match[i] and i not in seen:
                result.append((edges[i], edges[match[i]]))
                seen.add(i)
                seen.add(match[i])
        return result

    def _bron_kerbosch_edge_disjoint(N, adjN, available_edges, deadline, max_cliques=1000):
        """Find edge-disjoint maximal cliques in a graph to cover all edges.
        Returns list of cliques (each a set of vertices) that are edge-disjoint and cover all edges."""
        cliques = []
        remaining = set(available_edges)
        
        while remaining:
            if time.time() > deadline:
                # Fallback: cover each remaining edge as a 2-clique
                for e in remaining:
                    cliques.append(frozenset([e[0], e[1]]))
                break
            if len(cliques) >= max_cliques:
                for e in remaining:
                    cliques.append(frozenset([e[0], e[1]]))
                break
            
            # Find a maximal clique in the subgraph induced by remaining edges
            # Start from a vertex with highest degree in remaining
            deg = {v: 0 for v in N}
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            # Pick vertex with max degree
            if not deg or max(deg.values()) == 0:
                # No edges left
                break
            
            start_v = max(deg, key=lambda x: deg[x])
            
            # Bron-Kerbosch to find maximal clique containing start_v
            # Using recursive approach
            R = {start_v}
            P = set(N - {start_v})
            X = set()
            
            # Filter P to only vertices adjacent to all in R
            def filter_adjacent(P_set, R_set):
                result = set()
                for v in P_set:
                    if all(v in adjN[u] for u in R_set):
                        result.add(v)
                return result
            
            P = filter_adjacent(P, R)
            
            # Recursive Bron-Kerbosch
            def bk(R_set, P_set, X_set):
                if not P_set and not X_set:
                    if len(R_set) >= 2:
                        cliques.append(frozenset(R_set))
                    return
                # Pivot selection: choose vertex in P_set ∪ X_set with max degree in P_set
                all_v = P_set | X_set
                if all_v:
                    pivot = max(all_v, key=lambda v: len(adjN[v] & P_set))
                    candidates = P_set - adjN[pivot]
                else:
                    candidates = P_set
                
                for v in list(candidates):
                    bk(R_set | {v}, P_set & adjN[v], X_set & adjN[v])
                    P_set.discard(v)
                    X_set.add(v)
            
            # But we need edge-disjointness, so let's use a simpler greedy approach
            # that ensures each clique uses only remaining edges
            S = {start_v}
            # Greedily add vertices that form cliques with current S using only remaining edges
            changed = True
            while changed:
                changed = False
                for v in sorted(N - S):
                    # Check if adding v keeps S∪{v} as a clique using only remaining edges
                    ok = True
                    for u in S:
                        if v < u:
                            e = (v, u)
                        else:
                            e = (u, v)
                        if e not in remaining:
                            ok = False
                            break
                    if ok and v not in S:
                        S.add(v)
                        changed = True
            
            if len(S) < 2:
                # Just take one edge
                e = min(remaining)
                S = {e[0], e[1]}
            
            cliques.append(frozenset(S))
            # Remove all edges within S from remaining
            for u, v in combinations(sorted(S), 2):
                remaining.discard((u, v))
        
        return cliques

    def _solve(n, k, p, seed, deadline):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[v].add(u)

        # Core edges
        core_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                core_edges.add((u, v))

        cliques = []

        # Group I-vertices by identical neighbor set
        groups = {}
        for i in range(k, n):
            key = frozenset(adj[i])
            groups.setdefault(key, []).append(i)

        # Process each group: pack core vertices into edge-disjoint cliques
        for neighbors, members in sorted(groups.items(), key=lambda x: -len(x[0])):
            N = neighbors
            if not N:
                continue
            
            # Build adjacency within N
            adjN = {v: set() for v in N}
            for v in N:
                for w in N:
                    if v < w and (v, w) in core_edges:
                        adjN[v].add(w)
                        adjN[w].add(v)

            # Find edge-disjoint maximal cliques to cover all core edges within N
            # These cliques will each be combined with all members
            remaining = set((u, v) for u in N for v in N if u < v and (u, v) in core_edges)
            
            while remaining:
                if time.time() > deadline:
                    # Fallback: cover each remaining edge as a 2-clique
                    for e in remaining:
                        cliques.append(sorted(members) + sorted([e[0], e[1]]))
                    break
                
                # Greedy maximal clique using only remaining edges
                deg = {v: 0 for v in N}
                for u, v in remaining:
                    deg[u] += 1
                    deg[v] += 1
                
                if max(deg.values()) == 0:
                    break
                
                start_v = max(deg, key=lambda x: deg[x])
                S = {start_v}
                
                changed = True
                while changed:
                    changed = False
                    for v in sorted(N - S):
                        ok = True
                        for u in S:
                            if v < u:
                                e = (v, u)
                            else:
                                e = (u, v)
                            if e not in remaining:
                                ok = False
                                break
                        if ok and v not in S:
                            S.add(v)
                            changed = True
                
                if len(S) < 2:
                    e = min(remaining)
                    S = {e[0], e[1]}
                
                cliques.append(sorted(members) + sorted(S))
                
                # Remove all edges within S
                for u, v in combinations(sorted(S), 2):
                    remaining.discard((u, v))

        # Cover any remaining core edges with pure core cliques
        remaining_edges = set(core_edges)
        # Remove edges already covered by I-vertex cliques
        for clique in cliques:
            if len(clique) > 0:
                # Check if this clique contains only core vertices
                all_core = all(v < k for v in clique)
                if all_core:
                    for u, v in combinations(sorted(clique), 2):
                        remaining_edges.discard((u, v))

        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    cliques.append(sorted([e[0], e[1]]))
                break
            
            # Greedy maximal clique using only remaining edges
            deg = {v: 0 for v in range(k)}
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            if max(deg.values()) == 0:
                break
            
            start_v = max(deg, key=lambda x: deg[x])
            S = {start_v}
            
            changed = True
            while changed:
                changed = False
                for v in range(k):
                    if v in S:
                        continue
                    ok = True
                    for u in S:
                        if v < u:
                            e = (v, u)
                        else:
                            e = (u, v)
                        if e not in remaining_edges:
                            ok = False
                            break
                    if ok and v not in S:
                        S.add(v)
                        changed = True
            
            if len(S) < 2:
                e = min(remaining_edges)
                S = {e[0], e[1]}
            
            cliques.append(sorted(S))
            
            for u, v in combinations(sorted(S), 2):
                remaining_edges.discard((u, v))

        return cliques

    results = []
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(_solve(*spec, deadline))

    final_results = []
    for cliques in results:
        clean_cliques = []
        for clique in cliques:
            clean_cliques.append([int(v) for v in clique])
        final_results.append(clean_cliques)

    return final_results
