import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(R, P, X, graph, n):
        if not P and not X:
            if len(R) >= 2:
                yield frozenset(R)
            return
        if not P:
            return
        pivot_candidates = list(P) + list(X)
        if not pivot_candidates:
            return
        max_inter = -1
        pivot = pivot_candidates[0]
        for p in pivot_candidates:
            inter = len(P & graph[p])
            if inter > max_inter:
                max_inter = inter
                pivot = p
        for v in list(P - graph[pivot]):
            new_R = R | {v}
            new_P = P & graph[v]
            new_X = X & graph[v]
            yield from bron_kerbosch(new_R, new_P, new_X, graph, n)
            P.remove(v)
            X.add(v)

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        cliques = list(bron_kerbosch(frozenset(), set(range(n)), set(), adj, n))
        valid_cliques = [c for c in cliques if len(c) >= 2]
        if not valid_cliques:
            return []

        C = len(valid_cliques)

        # vertex_cliques[v] = list of clique indices containing v
        vertex_cliques = defaultdict(list)
        for i, c in enumerate(valid_cliques):
            for v in c:
                vertex_cliques[v].append(i)

        # clique_vertices[i] = set of vertices in clique i (for fast intersection checks)
        clique_vertices = [set(c) for c in valid_cliques]

        best_set = None
        best_size = float('inf')

        rng_solver = random.Random(seed + 777)

        # Strategy: multiple greedy restarts with different orderings,
        # then local search with removals and 1-for-1 swaps.
        
        # Pre-compute for each vertex the list of cliques it covers
        # and for each clique the list of vertices in it.

        def greedy_cover(ordering=None):
            """Greedy set cover. If ordering is given, use it as a priority for ties."""
            hit = [False] * C
            current = set()
            hit_count = [0] * C  # not needed, just use hit array
            
            # We'll maintain a priority: for each vertex, how many unhit cliques it covers.
            # Recompute each step for simplicity (n is small).
            
            while True:
                unhit = [i for i in range(C) if not hit[i]]
                if not unhit:
                    break
                
                best_v = -1
                best_hits = 0
                best_tie_break = float('inf')
                
                for v in range(n):
                    if v in current:
                        continue
                    hits = 0
                    for ci in vertex_cliques[v]:
                        if not hit[ci]:
                            hits += 1
                    if hits > best_hits:
                        best_hits = hits
                        best_v = v
                        # For tie-breaking, use ordering if provided
                        if ordering is not None:
                            best_tie_break = ordering.index(v) if v in ordering else n
                    elif hits == best_hits and hits > 0:
                        # Tie: use ordering
                        if ordering is not None:
                            tb = ordering.index(v) if v in ordering else n
                            if tb < best_tie_break:
                                best_v = v
                                best_tie_break = tb
                        else:
                            # Random tie-break
                            if rng_solver.random() < 0.5:
                                best_v = v
                
                if best_v == -1:
                    # Should not happen if there are unhit cliques
                    break
                if best_hits == 0:
                    break
                    
                current.add(best_v)
                for ci in vertex_cliques[best_v]:
                    hit[ci] = True
                    
            return current

        def local_search_remove(current):
            """Remove redundant vertices."""
            improved = True
            while improved:
                improved = False
                vs = list(current)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        return current
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        c = clique_vertices[ci]
                        found_other = False
                        for u in c:
                            if u != v and u in current:
                                found_other = True
                                break
                        if not found_other:
                            can_remove = False
                            break
                    if can_remove:
                        current.remove(v)
                        improved = True
            return current

        def local_search_swap(current):
            """Try 1-for-1 swaps to enable further removals."""
            improved = True
            while improved:
                improved = False
                vs = list(current)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        return current
                    # Try swapping v with some u not in current
                    for u in range(n):
                        if u in current:
                            continue
                        new_set = set(current)
                        new_set.remove(v)
                        new_set.add(u)
                        # Check if new_set still covers all cliques
                        valid = True
                        for ci in range(C):
                            c = clique_vertices[ci]
                            if not (c & new_set):
                                valid = False
                                break
                        if valid:
                            # Now try to remove more
                            new_set = local_search_remove(new_set)
                            if len(new_set) < len(current):
                                current = new_set
                                improved = True
                                break
            return current

        # Run multiple restarts
        num_restarts = 300
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Different strategies for each restart
            if r % 3 == 0:
                # Standard greedy with random tie-breaking
                current = greedy_cover()
            elif r % 3 == 1:
                # Greedy with a random ordering as tie-break
                ordering = list(range(n))
                rng_solver.shuffle(ordering)
                current = greedy_cover(ordering)
            else:
                # Greedy with vertex-degree-based ordering (high degree first as tie-break)
                deg = [len(adj[v]) for v in range(n)]
                ordering = sorted(range(n), key=lambda v: (-deg[v], v))
                current = greedy_cover(ordering)
            
            current = local_search_remove(current)
            
            if len(current) < best_size:
                best_size = len(current)
                best_set = set(current)
                # Try swap local search on the best so far
                if time.time() < deadline - 5:
                    current2 = local_search_swap(set(current))
                    if len(current2) < best_size:
                        best_size = len(current2)
                        best_set = set(current2)

        # If we haven't found anything, fallback
        if best_set is None:
            # Just take all vertices that are in any clique
            all_vertices = set()
            for c in valid_cliques:
                all_vertices.update(c)
            best_set = all_vertices

        return sorted(best_set)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
        res = solve_instance(n, p, seed)
        results.append(res)

    return results
