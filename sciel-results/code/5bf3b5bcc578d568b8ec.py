"""Clique partition of split graphs: structural optimization via subset selection + greedy coloring."""
import time
from random import Random
from itertools import combinations

def construct():
    _T0 = time.time()
    _DEADLINE = 130.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)

        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # Build adjacency for C vertices and N_C for I vertices
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in edge_set)

        I_verts = list(range(k, n))
        m = len(I_verts)

        # C-C edges
        cc_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cc_edge_idx = {}
        for idx, e in enumerate(cc_edges):
            cc_edge_idx[e] = idx
        m_cc = len(cc_edges)

        # For each I-vertex, the set of C-C edges within its neighborhood
        E_N = []
        d_v = []
        for i, v in enumerate(I_verts):
            neigh = sorted(N_C[v])
            d_v.append(len(neigh))
            s = set()
            for a, b in combinations(neigh, 2):
                e = (a, b) if a < b else (b, a)
                if e in cc_edge_idx:
                    s.add(cc_edge_idx[e])
            E_N.append(s)

        # Greedy coloring of a graph given as adjacency list on vertices 0..k-1
        # Only considers edges in the given edge set
        def _greedy_color(adj, edge_subset):
            """Color the graph (V, edge_subset) greedily. Returns list of color classes."""
            # Build active vertex set
            active = set()
            for e in edge_subset:
                a, b = cc_edges[e]
                active.add(a)
                active.add(b)
            if not active:
                return []
            
            # Build complement adjacency among active vertices
            # Two vertices can be in same color class if NOT connected by an edge in edge_subset
            adj_set = {}
            for v in active:
                adj_set[v] = set()
            for e in edge_subset:
                a, b = cc_edges[e]
                adj_set[a].add(b)
                adj_set[b].add(a)
            
            # Greedy: order by degree descending, assign to first available color
            order = sorted(active, key=lambda v: -len(adj_set[v]))
            color_of = {}
            colors = []  # list of sets
            
            for v in order:
                # Find first color where v has no neighbor
                placed = False
                for ci, cset in enumerate(colors):
                    if all(v not in adj_set.get(w, set()) for w in cset):
                        cset.add(v)
                        color_of[v] = ci
                        placed = True
                        break
                if not placed:
                    colors.append({v})
                    color_of[v] = len(colors) - 1
            
            return [sorted(c) for c in colors if len(c) >= 2]

        def _theta(edge_subset):
            """Min clique cover of remaining C-C edges (greedy)."""
            if not edge_subset:
                return 0, []
            # Check if complete graph on some vertex set
            active = set()
            for e in edge_subset:
                a, b = cc_edges[e]
                active.add(a)
                active.add(b)
            n_act = len(active)
            if len(edge_subset) == n_act * (n_act - 1) // 2:
                return 1, [sorted(active)]
            
            # Greedy coloring of complement
            # Complement edge exists if NOT in edge_subset
            comp_adj = {v: set() for v in active}
            for a in active:
                for b in active:
                    if a < b:
                        e = (a, b)
                        if e not in edge_subset:
                            comp_adj[a].add(b)
                            comp_adj[b].add(a)
            
            # Greedy coloring of complement = clique cover of original
            order = sorted(active, key=lambda v: -len(comp_adj[v]))
            colors = []
            for v in order:
                placed = False
                for ci, cset in enumerate(colors):
                    if all(v in comp_adj.get(w, set()) or v == w for w in cset):
                        # v is connected to all in cset in complement => independent in original
                        cset.add(v)
                        placed = True
                        break
                if not placed:
                    colors.append({v})
            
            result = [sorted(c) for c in colors if len(c) >= 2]
            return len(result), result

        def _eval(S_set):
            """Evaluate total clique count for a given set S of big-clique vertices."""
            n_big = len(S_set)
            n_indiv = sum(d_v[i] for i in range(m) if i not in S_set and d_v[i] > 0)
            
            # Compute consumed C-C edges
            consumed = set()
            for i in S_set:
                consumed |= E_N[i]
            remaining = set(range(m_cc)) - consumed
            # Filter to only actual edges
            remaining = {e for e in remaining if e < m_cc}
            
            theta, ccliques = _theta(remaining)
            return n_big + n_indiv + theta, ccliques

        # Local search with better initialization
        best_val = float('inf')
        best_S = frozenset()
        best_ccliques = []

        def _update(S_set):
            nonlocal best_val, best_S, best_ccliques
            val, ccliques = _eval(S_set)
            if val < best_val:
                best_val = val
                best_S = S_set
                best_ccliques = ccliques
            return val

        # Multiple initial solutions
        inits = []
        # All vertices with d>=2
        S_all = frozenset(i for i in range(m) if d_v[i] >= 2)
        inits.append(S_all)
        # Empty
        inits.append(frozenset())
        # Greedy by benefit: start empty, add vertex with max marginal benefit
        S_g = frozenset()
        for i in sorted(range(m), key=lambda x: d_v[x], reverse=True):
            if d_v[i] < 2:
                continue
            S_try = S_g | frozenset([i])
            inits.append(S_try)
            S_g = S_try
        inits.append(S_g)

        visited = set()
        for S_init in inits:
            if _time_up():
                break
            if S_init in visited:
                continue
            visited.add(S_init)
            S_cur = S_init
            _update(S_cur)
            
            improved = True
            while improved and not _time_up():
                improved = False
                # Try adding one vertex
                best_add = None
                best_add_val = best_val
                for i in range(m):
                    if i in S_cur or d_v[i] < 2:
                        continue
                    S_new = S_cur | frozenset([i])
                    if S_new in visited:
                        continue
                    val = _eval(S_new)[0]
                    if val < best_add_val:
                        best_add_val = val
                        best_add = (i, S_new)
                if best_add is not None:
                    i, S_new = best_add
                    _update(S_new)
                    S_cur = S_new
                    visited.add(S_new)
                    improved = True
                    continue
                
                # Try removing one vertex
                best_rem = None
                best_rem_val = best_val
                for i in list(S_cur):
                    S_new = S_cur - frozenset([i])
                    if S_new in visited:
                        continue
                    val = _eval(S_new)[0]
                    if val < best_rem_val:
                        best_rem_val = val
                        best_rem = (i, S_new)
                if best_rem is not None:
                    i, S_new = best_rem
                    _update(S_new)
                    S_cur = S_new
                    visited.add(S_new)
                    improved = True
                    continue
                
                # Try swap
                best_swap = None
                best_swap_val = best_val
                for i_add in range(m):
                    if i_add in S_cur or d_v[i_add] < 2:
                        continue
                    for i_rem in list(S_cur):
                        S_new = (S_cur - frozenset([i_rem])) | frozenset([i_add])
                        if S_new in visited:
                            continue
                        val = _eval(S_new)[0]
                        if val < best_swap_val:
                            best_swap_val = val
                            best_swap = (i_add, i_rem, S_new)
                if best_swap is not None:
                    i_add, i_rem, S_new = best_swap
                    _update(S_new)
                    S_cur = S_new
                    visited.add(S_new)
                    improved = True

        # Build final answer
        S_final = best_S
        cliques = []
        for i in range(m):
            v = I_verts[i]
            if i in S_final:
                cliques.append([v] + sorted(N_C[v]))
            else:
                for u in sorted(N_C[v]):
                    cliques.append([v, u])
        cliques.extend(best_ccliques)

        if not cliques:
            cliques = [list(range(k))]

        # Validate
        edge_count = {}
        for clique in cliques:
            if len(clique) < 2:
                continue
            for a, b in combinations(clique, 2):
                e = (a, b) if a < b else (b, a)
                edge_count[e] = edge_count.get(e, 0) + 1

        missing = [e for e in edge_set if edge_count.get(e, 0) == 0]
        over = [e for e, c in edge_count.items() if c > 1]

        if missing or over:
            cliques = [list(range(k))]
            for (u, v) in edge_set:
                if u >= k or v >= k:
                    cliques.append([u, v])

        # Ensure all cliques have at least 2 vertices
        cliques = [c for c in cliques if len(c) >= 2]
        if not cliques:
            cliques = [list(range(k))]

        # Convert to ensure Python ints
        cliques = [[int(v) for v in c] for c in cliques]

        results.append(cliques)

    return results
