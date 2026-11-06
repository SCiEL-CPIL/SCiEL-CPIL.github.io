"""Stronger solver: Bron-Kerbosch for maximal cliques, then minimum hitting set via
greedy + local search + exact ILP polish (scipy HiGHS) with a time deadline."""
import time
import random
import sys
from collections import defaultdict

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # seconds; keep a safety margin before 150s kill

    def remaining_time():
        return DEADLINE - (time.time() - t0)

    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch_max_cliques(adj, n):
        """Return list of maximal cliques (each as a frozenset)."""
        max_cliques = []
        # Use the pivot version
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            # Choose pivot u from P ∪ X with max |P ∩ N(u)|
            union = P | X
            if union:
                # Pick pivot with max degree into P
                best_u = max(union, key=lambda u: len(P & adj[u]))
            else:
                return
            candidates = P - adj[best_u]
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        bk(set(), set(range(n)), set())
        return max_cliques

    def greedy_hit_set(cliques):
        """Greedy: repeatedly pick vertex covering most uncovered cliques."""
        # Map vertex -> list of clique indices it hits
        vertex_to_cliques = defaultdict(set)
        for i, cl in enumerate(cliques):
            for v in cl:
                vertex_to_cliques[v].add(i)
        
        uncovered = set(range(len(cliques)))
        chosen = set()
        
        while uncovered:
            best_v = None
            best_count = -1
            for v in vertex_to_cliques:
                if v in chosen:
                    continue
                cnt = len(vertex_to_cliques[v] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            if best_v is None:
                break
            chosen.add(best_v)
            uncovered -= vertex_to_cliques[best_v]
        
        return chosen

    def is_valid(chosen, cliques):
        for cl in cliques:
            if not cl & chosen:
                return False
        return True

    def local_search(chosen, cliques, vertex_to_cliques, n, time_limit):
        """Try to reduce |chosen| via removals, then swaps, then 2-swap."""
        chosen = set(chosen)
        start_time = time.time()
        
        # Precompute clique hits for speed
        def build_hit_map(sel):
            """Map clique_idx -> count of chosen vertices in it"""
            hm = [0] * len(cliques)
            for i, cl in enumerate(cliques):
                hm[i] = len(cl & sel)
            return hm
        
        # Phase 1: try removing vertices
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            hm = build_hit_map(chosen)
            for v in list(chosen):
                # Check if removing v still covers all cliques
                ok = True
                for i, cl in enumerate(cliques):
                    if hm[i] == 1 and v in cl:
                        ok = False
                        break
                if ok:
                    chosen.discard(v)
                    improved = True
        
        # Phase 2: try swapping one chosen vertex for an unchosen one
        all_vertices = set()
        for cl in cliques:
            all_vertices |= cl
        
        for _ in range(20):
            if (time.time() - start_time) >= time_limit:
                break
            improved = False
            hm = build_hit_map(chosen)
            for v in list(chosen):
                # Find cliques that only v covers (among chosen)
                critical_cliques = [i for i, cl in enumerate(cliques) if hm[i] == 1 and v in cl]
                
                if not critical_cliques:
                    continue
                
                # For each candidate replacement w not in chosen
                for w in all_vertices:
                    if w in chosen:
                        continue
                    # Check w covers all critical cliques
                    covers_all = all(w in cliques[i] for i in critical_cliques)
                    if not covers_all:
                        continue
                    # Try the swap
                    new_chosen = (chosen - {v}) | {w}
                    if is_valid(new_chosen, cliques):
                        chosen = new_chosen
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        # Phase 3: try 2-removal (remove two vertices that are both redundant)
        for _ in range(10):
            if (time.time() - start_time) >= time_limit:
                break
            improved = False
            hm = build_hit_map(chosen)
            chosen_list = sorted(chosen)
            for i in range(len(chosen_list)):
                v1 = chosen_list[i]
                for j in range(i + 1, len(chosen_list)):
                    v2 = chosen_list[j]
                    # Check if removing both still covers all cliques
                    ok = True
                    for k, cl in enumerate(cliques):
                        if hm[k] <= 2:
                            cnt = 0
                            if v1 in cl:
                                cnt += 1
                            if v2 in cl:
                                cnt += 1
                            if hm[k] - cnt < 1:
                                ok = False
                                break
                    if ok:
                        chosen.discard(v1)
                        chosen.discard(v2)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        # Phase 4: final removal pass
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            hm = build_hit_map(chosen)
            for v in list(chosen):
                ok = True
                for i, cl in enumerate(cliques):
                    if hm[i] == 1 and v in cl:
                        ok = False
                        break
                if ok:
                    chosen.discard(v)
                    improved = True
        
        return chosen

    def ilp_polish(chosen, cliques, n, time_limit):
        """Solve the hitting set exactly via ILP (scipy HiGHS)."""
        if not HAVE_SCIPY:
            return chosen
        if time.time() - time.time() > time_limit:  # placeholder, actual check below
            return chosen
        
        num_cliques = len(cliques)
        num_vertices = n
        
        # Objective: minimize sum of x_v
        c = [1.0] * num_vertices
        
        # Constraints: for each clique, sum_{v in clique} x_v >= 1
        # A is num_cliques x num_vertices
        A_rows = []
        for cl in cliques:
            row = [0.0] * num_vertices
            for v in cl:
                row[v] = 1.0
            A_rows.append(row)
        
        # LinearConstraint: A @ x >= 1 for each row
        # scipy milp uses lb, ub per constraint
        lb = [1.0] * num_cliques
        ub = [float('inf')] * num_cliques
        
        # Bounds: 0 <= x_v <= 1
        var_lb = [0.0] * num_vertices
        var_ub = [1.0] * num_vertices
        
        # Integrality: all integer
        integrality = [1] * num_vertices
        
        try:
            result = milp(
                c=c,
                constraints=LinearConstraint(A_rows, lb=lb, ub=ub),
                integrality=integrality,
                bounds=Bounds(var_lb, var_ub),
                options={"time_limit": max(0.1, time_limit), "mip_rel_gap": 0.0}
            )
            if result.status == 0:  # optimal
                sol = set()
                for v in range(num_vertices):
                    if result.x[v] > 0.5:
                        sol.add(v)
                # Verify validity
                if is_valid(sol, cliques):
                    return sol
        except Exception:
            pass
        
        return chosen

    def solve_instance(n, p, seed, time_budget):
        edges = instance(n, p, seed)
        adj = build_adj(n, edges)
        
        # Find maximal cliques
        cliques = bron_kerbosch_max_cliques(adj, n)
        
        if not cliques:
            return []
        
        # Build vertex-to-clique mapping
        vertex_to_cliques = defaultdict(set)
        for i, cl in enumerate(cliques):
            for v in cl:
                vertex_to_cliques[v].add(i)
        
        # Greedy initial solution
        chosen = greedy_hit_set(cliques)
        
        # Local search with most of the time
        elapsed = time.time() - t0
        ls_time = max(0.1, time_budget * 0.4)
        chosen = local_search(chosen, cliques, vertex_to_cliques, n, ls_time)
        
        # ILP polish with remaining time
        elapsed = time.time() - t0
        ilp_time = max(0.5, time_budget - elapsed - 1.0)
        chosen = ilp_polish(chosen, cliques, n, ilp_time)
        
        # Final quick local search on ILP result
        elapsed = time.time() - t0
        final_ls_time = max(0.05, time_budget - elapsed - 0.5)
        if final_ls_time > 0.1:
            chosen = local_search(chosen, cliques, vertex_to_cliques, n, final_ls_time)
        
        return sorted(chosen)

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        if time.time() - t0 > DEADLINE - 10:
            # Not enough time; return what we have plus empty for rest
            results.append([])
            continue
        # Allocate time: give more to later (larger) instances
        remaining = DEADLINE - (time.time() - t0)
        if idx == 0:
            budget = min(remaining * 0.25, 30.0)
        elif idx == 1:
            budget = min(remaining * 0.35, 40.0)
        else:
            budget = min(remaining - 5.0, 60.0)
        budget = max(budget, 5.0)
        sol = solve_instance(n, p, seed, budget)
        results.append(sol)
    
    return results
