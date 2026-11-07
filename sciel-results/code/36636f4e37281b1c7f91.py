import time
from random import Random
from collections import deque
from scipy.optimize import linear_sum_assignment
import numpy as np

def construct():
    _T0 = time.time()
    _DEADLINE = 140.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _dsatur_color_bipartite(c_adj, v_adj, num_c, num_v):
        """Color bipartite graph: C vertices 0..num_c-1, V vertices num_c..num_c+num_v-1.
        Returns color list (length num_c + num_v) using DSATUR greedy."""
        total = num_c + num_v
        adj = [set() for _ in range(total)]
        for c in range(num_c):
            for v in c_adj[c]:
                adj[c].add(num_c + v)
                adj[num_c + v].add(c)
        for v in range(num_v):
            for c in v_adj[v]:
                adj[num_c + v].add(c)
                adj[c].add(num_c + v)

        colors = [-1] * total
        sat_colors = [set() for _ in range(total)]
        degree = [len(adj[i]) for i in range(total)]
        uncolored = set(range(total))

        for _ in range(total):
            best_node = None
            best_key = None
            for node in uncolored:
                key = (len(sat_colors[node]), degree[node])
                if best_key is None or key > best_key:
                    best_key = key
                    best_node = node
            uncolored.discard(best_node)

            c = 0
            while c in sat_colors[best_node]:
                c += 1
            colors[best_node] = c

            for nb in adj[best_node]:
                if nb in uncolored and c not in sat_colors[nb]:
                    sat_colors[nb].add(c)

        return colors

    def _solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # N_C[v] = set of C-vertices adjacent to I-vertex v
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in edge_set)

        cross_edges_norm = []
        for (a, b) in edge_set:
            if a < k and b >= k:
                cross_edges_norm.append((a, b))
            elif b < k and a >= k:
                cross_edges_norm.append((b, a))

        # Baseline: 1 C-clique + per-cross-edge 2-cliques
        base_cliques = [list(range(k))]
        for (u, v) in cross_edges_norm:
            base_cliques.append(sorted([u, v]))
        best_cliques = base_cliques
        best_count = len(base_cliques)

        # All C-C edges
        all_cc_edges = [(u, w) for u in range(k) for w in range(u + 1, k)]
        total_cc = len(all_cc_edges)

        if total_cc == 0:
            return best_cliques, best_count

        # NEW APPROACH: Minimum-weight matching on edge-compatibility graph
        # Each edge in the split graph can be "merged" with compatible edges to form larger cliques.
        # We model this as: for each I-vertex v, we want to select a subset of C-vertices to include
        # in the clique {v} ∪ S_v, where S_v ⊆ N_C[v].
        # 
        # Key insight: The problem decomposes into assigning each C-C edge to exactly one I-vertex
        # that is adjacent to both endpoints. This is a set cover problem, but we can use matching
        # to find good solutions.
        #
        # Alternative: Use local search / simulated annealing to optimize the assignment.
        
        # Precompute for each v: which C-C edges it can cover
        v_cover = {}
        for v in range(k, n):
            nc = sorted(N_C[v])
            if len(nc) >= 2:
                v_cover[v] = set((nc[i], nc[j]) for i in range(len(nc)) for j in range(i + 1, len(nc)))
            else:
                v_cover[v] = set()

        # Greedy set cover to get initial solution
        uncovered = set(all_cc_edges)
        selected_v = []
        V_v = {v: set() for v in range(k, n)}

        while uncovered and not _time_up():
            best_v = None
            best_cnt = 0
            for v in range(k, n):
                if v in selected_v:
                    continue
                cnt = len(uncovered & v_cover[v])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_v is None or best_cnt == 0:
                break
            selected_v.append(best_v)
            uncovered -= v_cover[best_v]

        if uncovered:
            return best_cliques, best_count

        # Initialize V_v[v] with cross-edge requirements
        for (u, v) in cross_edges_norm:
            if v in selected_v:
                V_v[v].add(u)

        # Assign remaining C-C edges
        cc_covered = set()
        for v in selected_v:
            nc_v = sorted(V_v[v])
            for i in range(len(nc_v)):
                for j in range(i + 1, len(nc_v)):
                    cc_covered.add((nc_v[i], nc_v[j]))

        remaining_cc = set(all_cc_edges) - cc_covered
        for (u, w) in sorted(remaining_cc):
            if _time_up():
                break
            assigned = False
            for v in selected_v:
                if u in N_C[v] and w in N_C[v]:
                    V_v[v].add(u)
                    V_v[v].add(w)
                    assigned = True
                    break
            if not assigned:
                return best_cliques, best_count

        # Build initial cliques
        cliques = []
        for v in selected_v:
            if V_v[v]:
                clique = sorted([v] + list(V_v[v]))
                cliques.append(clique)

        # Verify validity
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edge_set:
                        return best_cliques, best_count

        # Verify all edges covered exactly once
        edge_cover_count = {}
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    edge_cover_count[(a, b)] = edge_cover_count.get((a, b), 0) + 1

        for e in edge_set:
            if edge_cover_count.get(e, 0) != 1:
                return best_cliques, best_count

        if len(cliques) < best_count:
            best_cliques = cliques
            best_count = len(cliques)

        # LOCAL SEARCH: Try to improve by moving C-vertices between I-vertex cliques
        # or by adding/removing I-vertices from the selection
        import random as _rnd
        _rng = _rnd.Random(seed + 999)
        
        for iteration in range(50):
            if _time_up():
                break
            
            # Try random perturbation: move a C-vertex from one I-clique to another
            if not selected_v or len(selected_v) < 2:
                break
            
            v1_idx = _rng.randrange(len(selected_v))
            v2_idx = _rng.randrange(len(selected_v) - 1)
            if v2_idx >= v1_idx:
                v2_idx += 1
            
            v1 = selected_v[v1_idx]
            v2 = selected_v[v2_idx]
            
            # Try moving a C-vertex from v1 to v2 (if it's in N_C[v2])
            candidates = [u for u in V_v[v1] if u in N_C[v2] and u not in V_v[v2]]
            if not candidates:
                continue
            
            u = _rng.choice(candidates)
            
            # Check if removing u from v1 still covers all C-C edges
            old_V_v1 = set(V_v[v1])
            V_v[v1].discard(u)
            
            # Check coverage
            valid = True
            for (a, b) in all_cc_edges:
                covered = False
                for v in selected_v:
                    if a in V_v[v] and b in V_v[v]:
                        covered = True
                        break
                if not covered:
                    valid = False
                    break
            
            if valid:
                V_v[v2].add(u)
                # Check if this reduces total clique count (it won't directly, but may enable future merges)
                # For now, just accept if valid
                pass
            else:
                V_v[v1] = old_V_v1
        
        return best_cliques, best_count

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append(sorted([u, v]))
            results.append(cliques)
            continue
        n, k, p, seed = spec
        cliques, _ = _solve_instance(n, k, p, seed)
        results.append(cliques)
    return results
