"""Combined strategy: Bron-Kerbosch enumeration + LP/Greedy initialization + Branch-and-Bound optimization."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s kill

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_pivot(R, P, X, adj, cliques):
        """Bron-Kerbosch with pivot to enumerate all maximal cliques."""
        if time.time() > DEADLINE:
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X with max |P ∩ N(u)|
        candidates = list(P) + list(X)
        if not candidates:
            return
        
        # Pick pivot with maximum neighborhood intersection with P
        best_u = None
        best_count = -1
        for u in candidates:
            count = len(P & adj[u])
            if count > best_count:
                best_count = count
                best_u = u
        
        # Only consider P \ N(u)
        for v in list(P - adj[best_u]):
            bron_kerbosch_pivot(R | {v}, P & adj[v], X & adj[v], adj, cliques)
            P.remove(v)
            X.add(v)

    def enumerate_maximal_cliques(n, edges):
        """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Optimization: Isolated vertices cannot be in any clique of size >= 2.
        # We can restrict the initial P to non-isolated vertices.
        non_isolated = {u for u in range(n) if adj[u]}
        if not non_isolated:
            return []
            
        cliques = []
        bron_kerbosch_pivot(frozenset(), set(non_isolated), set(), adj, cliques)
        return cliques

    def greedy_hitting_set(cliques, n):
        """Greedy: repeatedly pick vertex covering most uncovered cliques."""
        if not cliques:
            return set()
            
        # Map clique index to set of vertices
        # We need to track which cliques are covered
        uncovered_indices = set(range(len(cliques)))
        result = set()
        
        # Pre-compute vertex to clique indices mapping for speed
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        while uncovered_indices:
            best_v = -1
            best_count = 0
            
            # Find vertex covering most uncovered cliques
            for v in range(n):
                if v in result:
                    continue
                count = 0
                for idx in vertex_to_cliques[v]:
                    if idx in uncovered_indices:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count == 0:
                break
                
            result.add(best_v)
            # Remove all cliques containing best_v
            for idx in vertex_to_cliques[best_v]:
                if idx in uncovered_indices:
                    uncovered_indices.remove(idx)
        
        return result

    def lp_rounding_hitting_set(cliques, n):
        """Solve LP relaxation for hitting set, round with greedy completion."""
        if not cliques:
            return set()
            
        try:
            from scipy.optimize import linprog
            m = len(cliques)
            # Objective: minimize sum x_v
            c_obj = np.ones(n)
            # Constraints: for each clique C, sum_{v in C} x_v >= 1
            # linprog minimizes c^T x subject to A_ub x <= b_ub
            # So we use -A x <= -1
            A_ub = np.zeros((m, n))
            for i, clique in enumerate(cliques):
                for v in clique:
                    A_ub[i, v] = 1.0
            A_ub = -A_ub
            b_ub = -np.ones(m)
            bounds = [(0, 1)] * n  # x_v in [0, 1]
            
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                x = res.x
                # Round: take vertices with x_v > 0.5
                selected = set(v for v in range(n) if x[v] > 0.5)
                
                # Check coverage
                uncovered = [c for c in cliques if not any(v in selected for v in c)]
                if not uncovered:
                    return selected
                else:
                    # Complete with greedy
                    extra = greedy_hitting_set(uncovered, n)
                    selected.update(extra)
                    return selected
        except Exception:
            pass
        return greedy_hitting_set(cliques, n)

    def local_search_minimize(hitting_set, cliques, n):
        """Try to remove vertices from the hitting set if possible."""
        hs = set(hitting_set)
        if not hs:
            return hs
            
        # Count how many times each vertex is needed to cover cliques
        # A vertex v is redundant if every clique containing v also contains another vertex in hs.
        
        # Pre-compute cliques containing each vertex
        vertex_in_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                if v in hs:
                    vertex_in_cliques[v].append(i)

        changed = True
        while changed:
            changed = False
            # Try removing each vertex
            for v in list(hs):
                # Check if removing v leaves any clique uncovered
                can_remove = True
                for idx in vertex_in_cliques[v]:
                    c = cliques[idx]
                    # Is there another vertex in c that is in hs (excluding v)?
                    if not any(u in hs for u in c if u != v):
                        can_remove = False
                        break
                if can_remove:
                    hs.remove(v)
                    changed = True
                    # Update vertex_in_cliques for remaining vertices? 
                    # For simplicity and speed, we restart the loop. 
                    # Given small n, this is acceptable.
                    break
        return hs

    def branch_and_bound_hitting_set(cliques, n, initial_upper_bound, time_limit=10.0):
        """Branch and bound for minimum hitting set."""
        start = time.time()
        best_solution = set(initial_upper_bound)
        best_size = len(best_solution)
        
        if best_size == 0:
            return best_solution

        # Sort cliques by size (smallest first)
        sorted_cliques = sorted(cliques, key=lambda c: len(c))
        
        # Pre-compute: for each clique, which vertices are in it
        # And for each vertex, which cliques it hits
        
        def bnb(idx, current_set, remaining_cliques):
            nonlocal best_solution, best_size
            if time.time() - start > time_limit:
                return
            if not remaining_cliques:
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_solution = set(current_set)
                return
            
            # Lower bound: current size + minimum additional vertices needed
            # Simple bound: 1 for each remaining clique? No, they might overlap.
            # Better bound: Max number of disjoint cliques in remaining?
            # Or just: if len(current_set) + 1 >= best_size, we can't improve if we must pick at least 1.
            # Actually, we must pick at least 1 vertex for the first remaining clique.
            if len(current_set) + 1 >= best_size:
                # Even if we pick 1 vertex that covers ALL remaining (unlikely), 
                # we won't beat best_size if len(current) + 1 >= best_size.
                # Wait, if len(current) + 1 == best_size, we might find an equal solution, 
                # but we want strictly smaller. So prune.
                return

            # Pick the smallest remaining clique
            clique = remaining_cliques[0]
            
            # Try each vertex in this clique
            # Heuristic: try vertices with highest frequency in remaining cliques first
            freq = {}
            for c in remaining_cliques:
                for v in c:
                    freq[v] = freq.get(v, 0) + 1
            
            candidates = sorted(clique, key=lambda v: freq.get(v, 0), reverse=True)
            
            for v in candidates:
                if time.time() - start > time_limit:
                    return
                if v in current_set:
                    continue
                new_set = current_set | {v}
                # Remove all cliques containing v
                new_remaining = [c for c in remaining_cliques[1:] if v not in c]
                bnb(idx + 1, new_set, new_remaining)

        bnb(0, set(), sorted_cliques)
        return best_solution

    def find_min_hitting_set(cliques, n):
        """Orchestrate the finding of the minimum hitting set."""
        if not cliques:
            return []
        
        # 1. Greedy
        sol_greedy = greedy_hitting_set(cliques, n)
        
        # 2. LP Rounding
        sol_lp = lp_rounding_hitting_set(cliques, n)
        
        # Pick the smaller of the two
        if len(sol_lp) < len(sol_greedy):
            current_best = sol_lp
        else:
            current_best = sol_greedy
            
        # 3. Local Search to minimize current_best
        current_best = local_search_minimize(current_best, cliques, n)
        
        # 4. Branch and Bound to try to improve
        # Allocate remaining time to BnB
        elapsed = time.time() - start_time
        remaining_time = max(1.0, DEADLINE - elapsed - 5.0) # 5s buffer for other instances
        
        # Only run BnB if we have enough time and the problem isn't trivially small
        if remaining_time > 2.0 and len(current_best) > 0:
            sol_bnb = branch_and_bound_hitting_set(cliques, n, current_best, time_limit=remaining_time)
            if len(sol_bnb) < len(current_best):
                current_best = sol_bnb
            # Apply local search again on BnB result
            current_best = local_search_minimize(current_best, cliques, n)
            
        return sorted(current_best)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # Fallback: return all vertices (valid but large)
            edges = generate_edges(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
        
        edges = generate_edges(n, p, seed)
        cliques = enumerate_maximal_cliques(n, edges)
        
        if not cliques:
            results.append([])
            continue
            
        sol = find_min_hitting_set(cliques, n)
        results.append(sol)

    return results
# EVOLVE-BLOCK-END
