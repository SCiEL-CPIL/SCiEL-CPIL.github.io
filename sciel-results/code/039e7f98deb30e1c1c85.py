"""Clique partition of split graphs via ILP for Type-B cliques + Max-Clique cover."""
import time
from random import Random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def construct():
    _T0 = time.time()
    _DEADLINE = 140.0  # safety margin before 150s kill

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _cc_edge(a, b):
        return (a, b) if a < b else (b, a)

    def _max_clique(adj, n):
        """Find a maximum clique in a graph with n vertices and adjacency list adj.
        Uses Bron-Kerbosch with pivot.
        Returns the clique as a set of vertices.
        """
        best_clique = set()
        
        def bronk(R, P, X):
            nonlocal best_clique
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = set(R)
                return
            
            # Pruning: if we can't beat best_clique, return
            if len(R) + len(P) <= len(best_clique):
                return
            
            # Choose pivot u from P U X
            # Heuristic: pick u with max degree in P
            pivot = None
            max_deg = -1
            for u in P:
                deg = len(adj[u] & P)
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            # If P is empty, check best and return
            if not P:
                if len(R) > len(best_clique):
                    best_clique = set(R)
                return

            # Iterate over P \ N(pivot)
            candidates = list(P - adj[pivot]) if pivot is not None else list(P)
            for v in candidates:
                # Add v to R
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bronk(new_R, new_P, new_X)
                # Move v from P to X
                P.discard(v)
                X.add(v)

        bronk(frozenset(), frozenset(range(n)), frozenset())
        return best_clique

    def _cover_remaining_cc(remaining, k):
        """Cover the remaining C-C edges with C-only cliques.
        Uses Maximum Clique extraction.
        """
        if not remaining:
            return []
        
        cliques = []
        remaining = set(remaining)
        
        # We need to rebuild adjacency for the current remaining graph
        # Vertices are 0..k-1
        while remaining:
            # Build adjacency for current remaining edges
            adj = [set() for _ in range(k)]
            active = set()
            for (a, b) in remaining:
                adj[a].add(b)
                adj[b].add(a)
                active.add(a)
                active.add(b)
            
            if not active:
                break

            # Map active vertices to 0..m-1 for the BK algorithm
            active_list = sorted(active)
            idx_map = {v: i for i, v in enumerate(active_list)}
            m = len(active_list)
            local_adj = [set() for _ in range(m)]
            
            for (a, b) in remaining:
                if a in idx_map and b in idx_map:
                    ia = idx_map[a]
                    ib = idx_map[b]
                    local_adj[ia].add(ib)
                    local_adj[ib].add(ia)
            
            max_cl = _max_clique(local_adj, m)
            
            if not max_cl:
                # Should not happen if remaining is not empty
                break
            
            # Map back to original vertex ids
            clique_vertices = {active_list[i] for i in max_cl}
            
            # Remove edges within this clique from remaining
            cl_sorted = sorted(clique_vertices)
            for i in range(len(cl_sorted)):
                for j in range(i + 1, len(cl_sorted)):
                    remaining.discard(_cc_edge(cl_sorted[i], cl_sorted[j]))
            
            cliques.append(cl_sorted)
            
        return cliques

    def _solve_type_b_ilp(n, k, edge_set):
        """Solve ILP to maximize number of Type-B cliques."""
        # Identify C-C edges
        cc_edges = []
        cc_edge_idx = {}
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) in edge_set:
                    cc_edges.append((u, v))
                    cc_edge_idx[(u, v)] = len(cc_edges) - 1
        
        if not cc_edges:
            return []
            
        # Identify I-vertices
        i_vertices = list(range(k, n))
        
        # Build adjacency for I-vertices to C
        # adj_c[i] = set of C vertices connected to I vertex i
        adj_c = {w: set() for w in i_vertices}
        for w in i_vertices:
            for u in range(k):
                if (u, w) in edge_set:
                    adj_c[w].add(u)
        
        # Variables x[e, w] for e in cc_edges, w in i_vertices
        # Only if w is adjacent to both endpoints of e
        valid_pairs = []
        for idx_e, (u, v) in enumerate(cc_edges):
            for w in i_vertices:
                if u in adj_c[w] and v in adj_c[w]:
                    valid_pairs.append((idx_e, w))
                    
        if not valid_pairs:
            return []
            
        num_vars = len(valid_pairs)
        var_to_pair = {i: pair for i, pair in enumerate(valid_pairs)}
        
        # Objective: Maximize sum x -> Minimize -sum x
        c = np.full(num_vars, -1.0)
        
        # Constraints:
        # 1. For each C-C edge e, sum_w x[e, w] <= 1
        # 2. For each I-vertex w and C-vertex u in adj_c[w], sum_{e incident to u} x[e, w] <= 1
        
        constraints = []
        
        # Constraint 1: Edge constraints
        # Group variables by edge index
        vars_by_edge = {}
        for i, (e_idx, w) in enumerate(valid_pairs):
            if e_idx not in vars_by_edge:
                vars_by_edge[e_idx] = []
            vars_by_edge[e_idx].append(i)
            
        for e_idx, var_indices in vars_by_edge.items():
            if not var_indices:
                continue
            row = np.zeros(num_vars)
            for i in var_indices:
                row[i] = 1.0
            constraints.append(LinearConstraint(row, -np.inf, 1.0))
            
        # Constraint 2: Vertex-Vertex constraints (Matching constraint for each w)
        # For each w, and each u in adj_c[w], sum of x[e, w] for e incident to u <= 1
        # Note: e is incident to u if u is one of its endpoints.
        
        # Precompute for each w, and for each u, the list of variables (e_idx, w) where e is incident to u
        vars_by_w_u = {}
        for i, (e_idx, w) in enumerate(valid_pairs):
            u, v = cc_edges[e_idx]
            if u in adj_c[w]:
                key = (w, u)
                if key not in vars_by_w_u:
                    vars_by_w_u[key] = []
                vars_by_w_u[key].append(i)
            if v in adj_c[w]:
                key = (w, v)
                if key not in vars_by_w_u:
                    vars_by_w_u[key] = []
                vars_by_w_u[key].append(i)
                
        for (w, u), var_indices in vars_by_w_u.items():
            if not var_indices:
                continue
            row = np.zeros(num_vars)
            for i in var_indices:
                row[i] = 1.0
            constraints.append(LinearConstraint(row, -np.inf, 1.0))
            
        bounds = Bounds(lb=0, ub=1)
        
        # Solve
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(num_vars))
        
        if not res.success:
            return []
            
        type_b_cliques = []
        for i, x in enumerate(res.x):
            if x > 0.5:
                e_idx, w = var_to_pair[i]
                u, v = cc_edges[e_idx]
                type_b_cliques.append([u, v, w])
                
        return type_b_cliques

    def _local_improve(cliques, edge_set, k, n):
        """Try to merge adjacent 2-cliques into larger cliques."""
        # Build a map from edge to clique index
        edge_to_clique = {}
        for i, cl in enumerate(cliques):
            if len(cl) == 2:
                e = _cc_edge(cl[0], cl[1])
                edge_to_clique[e] = i
        
        # Try to find pairs of 2-cliques that can be merged into a 3-clique
        # Two 2-cliques [a,b] and [a,c] can be merged if (b,c) is an edge
        # Two 2-cliques [a,b] and [c,d] can be merged if they share a vertex and form a triangle
        merged = [True] * len(cliques)
        new_cliques = []
        
        # Try merging 2-cliques that share a vertex
        for i in range(len(cliques)):
            if merged[i] or len(cliques[i]) != 2:
                continue
            a, b = cliques[i]
            for j in range(i + 1, len(cliques)):
                if merged[j] or len(cliques[j]) != 2:
                    continue
                c, d = cliques[j]
                # Check if they share a vertex
                if a == c or a == d:
                    # They share 'a', check if b and the other form a triangle
                    other = d if a == c else c
                    if _cc_edge(b, other) in edge_set:
                        # Can merge into [a, b, other]
                        new_cliques.append([a, b, other])
                        merged[i] = False
                        merged[j] = False
                        break
                elif b == c or b == d:
                    # They share 'b', check if a and the other form a triangle
                    other = d if b == c else c
                    if _cc_edge(a, other) in edge_set:
                        # Can merge into [b, a, other]
                        new_cliques.append([b, a, other])
                        merged[i] = False
                        merged[j] = False
                        break
        
        # Also try to absorb 2-cliques into existing larger cliques
        for i in range(len(cliques)):
            if merged[i] or len(cliques[i]) != 2:
                continue
            a, b = cliques[i]
            for j, cl in enumerate(cliques):
                if not merged[j] or j == i or len(cl) < 3:
                    continue
                # Check if a can be added to cl
                if a not in cl:
                    can_add_a = all(_cc_edge(a, v) in edge_set for v in cl)
                    if can_add_a:
                        new_cliques.append(cl + [a])
                        merged[i] = False
                        break
                # Check if b can be added to cl
                if b not in cl:
                    can_add_b = all(_cc_edge(b, v) in edge_set for v in cl)
                    if can_add_b:
                        new_cliques.append(cl + [b])
                        merged[i] = False
                        break
        
        result = []
        for i, cl in enumerate(cliques):
            if merged[i]:
                result.append(cl)
        result.extend(new_cliques)
        return result

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add(_cc_edge(a, b))

        # all C-C edges
        all_cc = set(_cc_edge(u, v) for u in range(k) for v in range(u + 1, k))
        
        # Solve ILP for Type-B cliques
        type_b_cliques = _solve_type_b_ilp(n, k, edge_set)
        
        cliques = list(type_b_cliques)
        
        # Determine covered edges
        covered_cc = set()
        covered_cross = set()
        
        for cl in type_b_cliques:
            if len(cl) == 3:
                u, v, w = cl
                covered_cc.add(_cc_edge(u, v))
                covered_cross.add(_cc_edge(u, w))
                covered_cross.add(_cc_edge(v, w))
                
        # Cover remaining C-C edges with C-only cliques
        remaining_cc = all_cc - covered_cc
        c_only = _cover_remaining_cc(remaining_cc, k)
        for cl in c_only:
            cliques.append(cl)
            
        # Cover remaining cross-edges with 2-cliques
        for w in range(k, n):
            for u in range(k):
                if _cc_edge(u, w) in edge_set and _cc_edge(u, w) not in covered_cross:
                    cliques.append([u, w])
                    covered_cross.add(_cc_edge(u, w))
        
        # Local improvement: try to merge 2-cliques into larger cliques
        cliques = _local_improve(cliques, edge_set, k, n)
        
        return cliques

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
        results.append(solve_one(*spec))

    # convert to plain python int lists
    out = []
    for clist in results:
        nl = []
        for c in clist:
            # Ensure int and list
            nl.append([int(x) for x in c])
        out.append(nl)
    return out
