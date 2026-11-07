"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                neighbors[u].add(v)
                neighbors[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        # Build edge sets
        E_CC = set()
        for u in range(k):
            for w in range(u + 1, k):
                E_CC.add((u, w))

        E_CI = {}
        for v in range(k, n):
            E_CI[v] = set(neighbors[v] & set(range(k)))

        # Tomita's maximum clique algorithm with pivoting
        def max_clique(vertices, adj_sets, deadline=None):
            """Find a maximum clique in the subgraph induced by vertices."""
            if not vertices:
                return []
            
            best = []
            
            def expand(C, P):
                nonlocal best
                if deadline and time.time() > deadline:
                    return
                if len(C) > len(best):
                    best = list(C)
                if not P:
                    return
                
                # Pivoting heuristic
                # Choose pivot u in P that maximizes |P ∩ N(u)|
                pivot_candidates = list(P)
                if not pivot_candidates:
                    return
                
                best_pivot = pivot_candidates[0]
                max_common = -1
                for u in pivot_candidates:
                    common = len(P & adj_sets[u])
                    if common > max_common:
                        max_common = common
                        best_pivot = u
                
                # Order candidates: try vertices NOT in N(best_pivot) first
                non_neighbors = P - adj_sets[best_pivot]
                neighbors_of_pivot = P & adj_sets[best_pivot]
                
                # Sort by degree within P (greedy heuristic)
                order = sorted(non_neighbors, key=lambda x: len(adj_sets[x] & P), reverse=True) + \
                        sorted(neighbors_of_pivot, key=lambda x: len(adj_sets[x] & P), reverse=True)
                
                for v in order:
                    if deadline and time.time() > deadline:
                        return
                    # If v is in neighbors_of_pivot, we can stop (optimization)
                    if v in neighbors_of_pivot and len(C) + len(P) <= len(best):
                        break
                    
                    new_P = P & adj_sets[v]
                    if len(C) + len(new_P) <= len(best):
                        continue
                    
                    C.append(v)
                    expand(C, new_P)
                    C.pop()
                    P.discard(v)
            
            expand([], set(vertices))
            return best

        # Strategy 1: Process I-vertices, greedily extract max clique from available CC edges
        def strategy_greedy_cc(order):
            available_cc = set(E_CC)
            cliques = []
            assigned_ci = {}

            # Build adjacency for C vertices within available_cc
            adj_c = [set() for _ in range(k)]
            for u in range(k):
                for w in range(u + 1, k):
                    if (u, w) in available_cc:
                        adj_c[u].add(w)
                        adj_c[w].add(u)

            for v in order:
                if not E_CI[v]:
                    continue
                neighbors_v = list(E_CI[v])

                # Find maximal clique in available_cc restricted to neighbors_v
                # Use Tomita's algorithm on the subgraph induced by neighbors_v
                if neighbors_v:
                    # Create adjacency sets for the subgraph
                    sub_adj = {u: set() for u in neighbors_v}
                    for u in neighbors_v:
                        for w in neighbors_v:
                            if u != w and (min(u, w), max(u, w)) in available_cc:
                                sub_adj[u].add(w)
                    
                    clique = max_clique(neighbors_v, sub_adj)
                    
                    if len(clique) >= 2:
                        cliques.append(sorted(clique + [v]))
                        for i in range(len(clique)):
                            for j in range(i + 1, len(clique)):
                                available_cc.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))
                        assigned_ci[v] = set(clique)
                        # Update adj_c
                        for u in clique:
                            adj_c[u] = set()
                        for u in clique:
                            for w in clique:
                                if u != w and (min(u, w), max(u, w)) in available_cc:
                                    adj_c[u].add(w)
                    elif len(clique) == 1:
                        cliques.append(sorted([clique[0], v]))
                        assigned_ci[v] = set(clique)

            # Cover remaining CI edges
            for v in E_CI:
                assigned = assigned_ci.get(v, set())
                for u in E_CI[v]:
                    if u not in assigned:
                        cliques.append(sorted([u, v]))

            # Cover remaining CC edges
            for edge in available_cc:
                cliques.append(sorted(list(edge)))

            return cliques

        # Strategy 2: Process I-vertices, cover ALL N(v) with multiple cliques
        def strategy_multi_clique(order):
            available_cc = set(E_CC)
            cliques = []
            assigned_ci = {}

            for v in order:
                if not E_CI[v]:
                    continue
                remaining = set(E_CI[v])
                v_cliques = []

                while remaining:
                    # Find maximal clique in remaining
                    sub_adj = {u: set() for u in remaining}
                    for u in remaining:
                        for w in remaining:
                            if u != w and (min(u, w), max(u, w)) in available_cc:
                                sub_adj[u].add(w)
                    
                    clique = max_clique(list(remaining), sub_adj)
                    
                    if len(clique) >= 2:
                        v_cliques.append(sorted(clique + [v]))
                        for i in range(len(clique)):
                            for j in range(i + 1, len(clique)):
                                available_cc.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))
                        remaining -= set(clique)
                    elif len(clique) == 1:
                        v_cliques.append(sorted([clique[0], v]))
                        remaining.discard(clique[0])
                    else:
                        break

                cliques.extend(v_cliques)
                assigned_ci[v] = set(E_CI[v])  # All covered

            # Cover remaining CC edges
            for edge in available_cc:
                cliques.append(sorted(list(edge)))

            return cliques

        # Strategy 3: Direct edge-based greedy
        def strategy_edge_greedy():
            all_edges = set()
            for u in range(k):
                for w in range(u + 1, k):
                    all_edges.add((u, w))
            for v in range(k, n):
                for u in E_CI[v]:
                    all_edges.add((min(u, v), max(u, v)))

            rem_set = set(all_edges)
            cliques = []

            adj = [set() for _ in range(n)]
            for u in range(k):
                for w in range(u + 1, k):
                    adj[u].add(w)
                    adj[w].add(u)
            for v in range(k, n):
                for u in E_CI[v]:
                    adj[u].add(v)
                    adj[v].add(u)

            while rem_set:
                # Pick an edge with maximum potential for extension
                best_edge = None
                best_score = -1
                for e in list(rem_set)[:50]:
                    u, w = e
                    common = adj[u] & adj[w]
                    score = sum(1 for x in common if x != u and x != w and (min(u, x), max(u, x)) in rem_set and (min(w, x), max(w, x)) in rem_set)
                    if score > best_score:
                        best_score = score
                        best_edge = e

                if best_edge is None:
                    best_edge = next(iter(rem_set))

                u, w = best_edge
                clique_vertices = {u, w}

                candidates = list(adj[u] & adj[w])
                candidates = [x for x in candidates if x != u and x != w]
                candidates.sort(key=lambda x: sum(1 for y in clique_vertices if x in adj[y]), reverse=True)

                for x in candidates:
                    if all((min(x, y), max(x, y)) in rem_set for y in clique_vertices):
                        clique_vertices.add(x)

                for u1, u2 in combinations(sorted(clique_vertices), 2):
                    rem_set.discard((u1, u2))

                cliques.append(sorted(clique_vertices))

            return cliques

        # Generate different orderings
        I_vertices = list(range(k, n))
        order1 = sorted(I_vertices, key=lambda v: len(E_CI[v]), reverse=True)
        order2 = sorted(I_vertices, key=lambda v: len(E_CI[v]))
        rng2 = random.Random(seed + 1000)
        order3 = I_vertices[:]
        rng2.shuffle(order3)

        # Try all strategies and keep the best
        best_cliques = None
        best_count = float('inf')
        deadline = time.time() + 100.0

        strategies = [
            (strategy_greedy_cc, order1),
            (strategy_greedy_cc, order2),
            (strategy_greedy_cc, order3),
            (strategy_multi_clique, order1),
            (strategy_multi_clique, order2),
            (strategy_multi_clique, order3),
            (strategy_edge_greedy, None),
        ]

        for func, order in strategies:
            if time.time() > deadline:
                break
            try:
                if order is not None:
                    cliques = func(order)
                else:
                    cliques = func()
                
                # Validate
                count = [0] * len(all_edges) if 'all_edges' in dir() else None
                eid = {}
                all_e = set()
                for u in range(k):
                    for w in range(u + 1, k):
                        all_e.add((u, w))
                for v in range(k, n):
                    for u in E_CI[v]:
                        all_e.add((min(u, v), max(u, v)))
                
                eid = {e: i for i, e in enumerate(all_e)}
                count = [0] * len(all_e)
                valid = True
                for clique in cliques:
                    L = len(clique)
                    for a in range(L):
                        ua = clique[a]
                        for b in range(a + 1, L):
                            vb = clique[b]
                            edge = (min(ua, vb), max(ua, vb))
                            idx = eid.get(edge)
                            if idx is None:
                                valid = False
                                break
                            count[idx] += 1
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    for c in count:
                        if c != 1:
                            valid = False
                            break
                
                if valid and len(cliques) < best_count:
                    best_count = len(cliques)
                    best_cliques = cliques
            except:
                continue

        if best_cliques is None:
            # Fallback
            cliques = []
            all_e = set()
            for u in range(k):
                for w in range(u + 1, k):
                    all_e.add((u, w))
            for v in range(k, n):
                for u in E_CI[v]:
                    all_e.add((min(u, v), max(u, v)))
            for e in all_e:
                cliques.append(sorted(list(e)))
            best_cliques = cliques

        return best_cliques

    result = []
    start_time = time.time()
    deadline = start_time + 115.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        cliques = solve(*spec)
        result.append(cliques)

    # Ensure we have exactly 3 results
    while len(result) < 3:
        spec = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))[len(result)]
        cliques = []
        n, k, p, seed = spec
        rng = random.Random(seed)
        for u in range(k):
            for w in range(u + 1, k):
                cliques.append([u, w])
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cliques.append([u, v])
        result.append(cliques)

    return result[:3]
# EVOLVE-BLOCK-END
