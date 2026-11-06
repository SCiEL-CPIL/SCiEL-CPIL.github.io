import time
import math
import random as _random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        """Generate edge list using the same RNG as the seed program."""
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting.
        """
        cliques = []
        
        def bk(R, P, X):
            if time.time() - start_time > DEADLINE:
                return
            
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u in P ∪ X to maximize |P ∩ adj[u]|
            best_u = None
            best_count = -1
            for u in P:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            for u in X:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            
            if best_u is None:
                candidates = list(P)
            else:
                candidates = list(P - adj[best_u])
            
            for v in candidates:
                if time.time() - start_time > DEADLINE:
                    return
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.discard(v)
                P.discard(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Enumerate maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        m = len(cliques)
        
        # Build vertex_cliques: map vertex index to set of clique indices it belongs to
        vertex_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(ci)
        
        # Identify active vertices (those in at least one clique)
        active_vertices = [v for v in range(n) if vertex_cliques[v]]
        
        if not active_vertices:
            return []
        
        # Preprocessing: Dominance Pruning
        active_vertices_sorted = sorted(active_vertices, key=lambda v: -len(vertex_cliques[v]))
        
        keep_vertices = set()
        for v in active_vertices_sorted:
            dominated = False
            for u in keep_vertices:
                if vertex_cliques[v].issubset(vertex_cliques[u]):
                    dominated = True
                    break
            if not dominated:
                keep_vertices.add(v)
                
        candidates = list(keep_vertices)
        
        if not candidates:
            return []

        # Precompute coverage for each candidate
        candidate_covers = [vertex_cliques[v] for v in candidates]
        num_candidates = len(candidates)
        
        # Precompute for each clique, which candidates cover it
        clique_candidates = [set() for _ in range(m)]
        for i in range(num_candidates):
            for ci in candidate_covers[i]:
                clique_candidates[ci].add(i)
        
        # Try ILP first if available
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
            
            # x[i] = 1 if we select candidate i
            # Minimize sum(x)
            # Subject to: for each clique ci, sum(x[i] for i in clique_candidates[ci]) >= 1
            
            A = np.zeros((m, num_candidates))
            for ci in range(m):
                for i in clique_candidates[ci]:
                    A[ci, i] = 1.0
            
            c = np.ones(num_candidates)
            
            constraints = [
                LinearConstraint(A, lb=np.ones(m), ub=np.inf)
            ]
            
            bounds = Bounds(lb=0, ub=1)
            
            result = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(num_candidates))
            
            if result.success and result.x is not None:
                sol_indices = set(int(i) for i in range(num_candidates) if result.x[i] > 0.5)
                # Verify the solution
                covered = set()
                for i in sol_indices:
                    covered |= candidate_covers[i]
                if covered == set(range(m)):
                    best_vertices = set(candidates[i] for i in sol_indices)
                    return sorted(best_vertices)
        except Exception:
            pass
        
        # Fallback: Greedy + Branch and Bound
        
        # Initial greedy upper bound
        def greedy_cover(uncovered_set, rng=None):
            selected = set()
            unc = set(uncovered_set)
            while unc:
                best_idx = -1
                best_cov = 0
                best_cov_list = []
                for i in range(num_candidates):
                    if i in selected:
                        continue
                    cov = len(unc & candidate_covers[i])
                    if cov > best_cov:
                        best_cov = cov
                        best_cov_list = [i]
                    elif cov == best_cov and best_cov > 0:
                        best_cov_list.append(i)
                if not best_cov_list:
                    break
                if rng and len(best_cov_list) > 1:
                    best_idx = rng.choice(best_cov_list)
                else:
                    best_idx = best_cov_list[0]
                selected.add(best_idx)
                unc -= candidate_covers[best_idx]
            return selected

        init_sol = greedy_cover(set(range(m)))
        best_size = len(init_sol)
        best_solution = set(init_sol)
        
        if best_size == 0:
            return []
        
        # Try a few random restarts
        rng = _random.Random(42)
        for _ in range(5):
            sol = greedy_cover(set(range(m)), rng)
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = set(sol)
        
        # Convert to actual vertex indices
        best_vertices = set(candidates[i] for i in best_solution)
        
        # Branch and Bound
        def compute_lower_bound(selected, uncovered):
            if not uncovered:
                return 0
            max_cov = 0
            for i in range(num_candidates):
                if i in selected:
                    continue
                cov = len(uncovered & candidate_covers[i])
                if cov > max_cov:
                    max_cov = cov
            if max_cov == 0:
                return float('inf')
            return math.ceil(len(uncovered) / max_cov)
        
        def branch_and_bound(selected, uncovered, depth=0):
            nonlocal best_size, best_vertices
            
            if not uncovered:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_vertices = set(candidates[i] for i in selected)
                return
            
            if len(selected) >= best_size:
                return
            
            if time.time() - start_time > DEADLINE:
                return
            
            lb = len(selected) + compute_lower_bound(selected, uncovered)
            if lb >= best_size:
                return
            
            # Choose a clique to branch on
            target_ci = -1
            min_avail = float('inf')
            
            for ci in uncovered:
                avail = len(clique_candidates[ci] - selected)
                if avail == 0:
                    return
                if avail < min_avail:
                    min_avail = avail
                    target_ci = ci
                    if min_avail == 1:
                        break
            
            if target_ci == -1:
                return
            
            avail_candidates = clique_candidates[target_ci] - selected
            avail_list = sorted(avail_candidates, key=lambda i: (-len(uncovered & candidate_covers[i]), i))
            
            for i in avail_list:
                new_selected = selected | {i}
                new_uncovered = uncovered - candidate_covers[i]
                branch_and_bound(new_selected, new_uncovered, depth + 1)
        
        branch_and_bound(set(), set(range(m)))
        
        # Local search polish
        if best_size > 1:
            sol_set = set(best_solution)
            improved = True
            while improved:
                improved = False
                for v_idx in list(sol_set):
                    trial_set = sol_set - {v_idx}
                    covered = set()
                    for i in trial_set:
                        covered |= candidate_covers[i]
                    if covered == set(range(m)):
                        sol_set = trial_set
                        best_size = len(sol_set)
                        best_solution = set(sol_set)
                        improved = True
                        break
                if time.time() - start_time > DEADLINE - 10:
                    break
        
        best_vertices = set(candidates[i] for i in best_solution)
        return sorted(best_vertices)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)
        
    return results
