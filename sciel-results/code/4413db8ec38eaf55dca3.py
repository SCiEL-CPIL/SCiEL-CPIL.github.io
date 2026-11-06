import time
import math
import random as _random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

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
        # If C_u is a subset of C_v, u is dominated by v (v covers all cliques u covers)
        # So we can safely remove u from the candidate set.
        
        # Sort by number of cliques covered, descending
        active_vertices_sorted = sorted(active_vertices, key=lambda v: -len(vertex_cliques[v]))
        
        keep_vertices = set()
        for v in active_vertices_sorted:
            # Check if v is dominated by any already kept vertex
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
        # candidate_covers[i] = set of clique indices covered by candidate i
        candidate_covers = [vertex_cliques[v] for v in candidates]
        num_candidates = len(candidates)
        
        # Precompute for each clique, which candidates cover it
        # clique_candidates[ci] = set of candidate indices that cover clique ci
        clique_candidates = [set() for _ in range(m)]
        for i in range(num_candidates):
            for ci in candidate_covers[i]:
                clique_candidates[ci].add(i)
        
        # Initial greedy upper bound with multiple random restarts
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

        # Get initial solution with deterministic greedy
        init_sol = greedy_cover(set(range(m)))
        best_size = len(init_sol)
        best_solution = set(init_sol)
        
        if best_size == 0:
            return []
        
        # Try a few random restarts to potentially get a better initial bound
        rng = _random.Random(42)
        for _ in range(5):
            sol = greedy_cover(set(range(m)), rng)
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = set(sol)
        
        # Convert best_solution to actual vertex indices
        best_vertices = set(candidates[i] for i in best_solution)
        
        # Branch and Bound with improved heuristics
        def compute_lower_bound(selected, uncovered):
            """Compute a lower bound using greedy set cover approximation"""
            if not uncovered:
                return 0
            
            # Simple bound: ceil(|uncovered| / max_coverage)
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
            
            # Lower bound
            lb = len(selected) + compute_lower_bound(selected, uncovered)
            if lb >= best_size:
                return
            
            # Choose a clique to branch on
            # Heuristic: Choose the uncovered clique with the fewest available candidates
            # Break ties by choosing the one that is hardest to cover
            target_ci = -1
            min_avail = float('inf')
            best_hardness = -1
            
            for ci in uncovered:
                avail = len(clique_candidates[ci] - selected)
                if avail == 0:
                    return  # Infeasible
                if avail < min_avail:
                    min_avail = avail
                    target_ci = ci
                    # Hardness: number of cliques that share these same candidates
                    # (heuristic: just use avail as tiebreaker for now)
                    if min_avail == 1:
                        break
            
            if target_ci == -1:
                return
            
            # Get candidates that can cover target_ci, not yet selected
            avail_candidates = clique_candidates[target_ci] - selected
            
            # Order by coverage in descending order, then by index for determinism
            avail_list = sorted(avail_candidates, key=lambda i: (-len(uncovered & candidate_covers[i]), i))
            
            for i in avail_list:
                new_selected = selected | {i}
                new_uncovered = uncovered - candidate_covers[i]
                branch_and_bound(new_selected, new_uncovered, depth + 1)
        
        branch_and_bound(set(), set(range(m)))
        
        # Local search: try swapping one vertex in the solution with one not in it
        # This can sometimes find a smaller solution that B&B missed due to time limits
        if best_size > 1:
            sol_set = set(best_solution)
            improved = True
            while improved:
                improved = False
                # Try removing each vertex and see if we can still cover all cliques
                for v_idx in list(sol_set):
                    trial_set = sol_set - {v_idx}
                    # Check if trial_set still covers all cliques
                    covered = set()
                    for i in trial_set:
                        covered |= candidate_covers[i]
                    if covered == set(range(m)):
                        sol_set = trial_set
                        best_size = len(sol_set)
                        best_solution = set(sol_set)
                        improved = True
                        break
                # Try adding a vertex and removing two others (2-exchange)
                if not improved and best_size >= 3:
                    for v_idx in list(sol_set):
                        trial_set = sol_set - {v_idx}
                        for u_idx in list(trial_set):
                            trial_set2 = trial_set - {u_idx}
                            # Find a vertex w not in trial_set2 that covers all cliques
                            # that were only covered by v_idx or u_idx
                            need_cover = set()
                            for ci in range(m):
                                if ci not in candidate_covers[v_idx] and ci not in candidate_covers[u_idx]:
                                    continue
                                # Check if ci is covered by trial_set2
                                covered_by_trial = False
                                for i in trial_set2:
                                    if ci in candidate_covers[i]:
                                        covered_by_trial = True
                                        break
                                if not covered_by_trial:
                                    need_cover.add(ci)
                            
                            if not need_cover:
                                sol_set = trial_set2
                                best_size = len(sol_set)
                                best_solution = set(sol_set)
                                improved = True
                                break
                            else:
                                # Find a single vertex that covers all of need_cover
                                for w_idx in range(num_candidates):
                                    if w_idx in trial_set2:
                                        continue
                                    if need_cover.issubset(candidate_covers[w_idx]):
                                        new_sol = trial_set2 | {w_idx}
                                        if len(new_sol) < best_size:
                                            sol_set = new_sol
                                            best_size = len(sol_set)
                                            best_solution = set(sol_set)
                                            improved = True
                                            break
                        if improved:
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
