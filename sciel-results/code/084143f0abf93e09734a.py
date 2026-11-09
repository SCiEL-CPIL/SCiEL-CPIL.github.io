"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_pivot(R, P, X, adj, cliques, depth=0):
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
            bron_kerbosch_pivot(R | {v}, P & adj[v], X & adj[v], adj, cliques, depth + 1)
            P.remove(v)
            X.add(v)

    def enumerate_maximal_cliques(n, edges):
        """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        cliques = []
        bron_kerbosch_pivot(frozenset(), set(range(n)), set(), adj, cliques)
        return cliques

    def fast_greedy_hitting_set(cliques, n):
        """Fast greedy: precompute coverage, use efficient updates."""
        if not cliques:
            return []
        
        m = len(cliques)
        # Precompute: for each vertex, which cliques it covers
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].append(i)
        
        uncovered = [True] * m
        result = set()
        
        while True:
            # Check if all covered
            if not any(uncovered):
                break
            
            # Find vertex covering most uncovered cliques
            best_v = -1
            best_count = 0
            for v in range(n):
                if v in result:
                    continue
                count = sum(1 for i in vertex_to_cliques[v] if uncovered[i])
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count == 0:
                break
            
            result.add(best_v)
            for i in vertex_to_cliques[best_v]:
                uncovered[i] = False
        
        return sorted(result)

    def local_search_improve(solution, cliques, n, max_iterations=100):
        """Try to improve solution via removal and swap operations."""
        if not solution:
            return solution
        
        sol_set = set(solution)
        
        def is_valid(s):
            return all(any(v in s for v in clique) for clique in cliques)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try removing each vertex
            for v in list(sol_set):
                candidate = sol_set - {v}
                if is_valid(candidate):
                    sol_set = candidate
                    improved = True
                    break
            
            # Try swapping: remove v, add w
            for v in list(sol_set):
                for w in range(n):
                    if w in sol_set:
                        continue
                    candidate = (sol_set - {v}) | {w}
                    if len(candidate) < len(sol_set) and is_valid(candidate):
                        sol_set = candidate
                        improved = True
                        break
                if improved:
                    break
        
        return sorted(sol_set)

    def lp_rounding_hitting_set(cliques, n):
        """Solve LP relaxation for hitting set, round with greedy completion."""
        try:
            from scipy.optimize import linprog
            m = len(cliques)
            if m == 0:
                return []
            # Objective: minimize sum x_v
            c_obj = np.ones(n)
            # Constraints: A_ub @ x <= b_ub (we need >= 1, so use -A @ x <= -1)
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
                # Try multiple rounding thresholds
                best_solution = None
                for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    selected = set(v for v in range(n) if x[v] > threshold)
                    if all(any(v in selected for v in clique) for clique in cliques):
                        if best_solution is None or len(selected) < len(best_solution):
                            best_solution = set(selected)
                
                if best_solution is not None:
                    return sorted(best_solution)
                
                # Fallback: use greedy on uncovered
                selected = set(v for v in range(n) if x[v] > 0.5)
                uncovered_cliques = [clique for clique in cliques if not any(v in selected for v in clique)]
                if uncovered_cliques:
                    extra = fast_greedy_hitting_set(uncovered_cliques, n)
                    selected.update(extra)
                return sorted(selected)
        except Exception:
            pass
        return fast_greedy_hitting_set(cliques, n)

    def branch_and_bound_hitting_set(cliques, n, time_limit=5.0):
        """Branch and bound for minimum hitting set with better bounds."""
        start = time.time()
        best_solution = None
        best_size = float('inf')
        
        # Sort cliques by size (smallest first)
        sorted_cliques = sorted(cliques, key=lambda c: len(c))
        m = len(sorted_cliques)
        
        if m == 0:
            return []
        
        # Precompute vertex frequencies
        vertex_freq = {v: 0 for v in range(n)}
        for clique in sorted_cliques:
            for v in clique:
                vertex_freq[v] += 1
        
        def get_lower_bound(remaining_cliques, current_size):
            """Compute a lower bound on additional vertices needed."""
            if not remaining_cliques:
                return current_size
            
            # Count how many cliques each vertex can still cover
            # Simple bound: need at least ceil(uncovered_cliques / max_coverage) more
            max_cov = 0
            for v in range(n):
                cov = sum(1 for c in remaining_cliques if v in c)
                if cov > max_cov:
                    max_cov = cov
            
            if max_cov == 0:
                return current_size + len(remaining_cliques)
            
            import math
            return current_size + math.ceil(len(remaining_cliques) / max_cov)
        
        def bnb(idx, current_set, remaining_cliques):
            nonlocal best_solution, best_size
            if time.time() - start > time_limit:
                return
            
            if not remaining_cliques:
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_solution = set(current_set)
                return
            
            # Lower bound check
            lb = get_lower_bound(remaining_cliques, len(current_set))
            if lb >= best_size:
                return
            
            # Pick the smallest remaining clique
            clique = remaining_cliques[0]
            
            # Try vertices in this clique, ordered by frequency (most frequent first)
            vertices_in_clique = sorted(clique, key=lambda v: -vertex_freq[v])
            
            for v in vertices_in_clique:
                if time.time() - start > time_limit:
                    return
                new_set = current_set | {v}
                # Remove all cliques containing v
                new_remaining = [c for c in remaining_cliques[1:] if v not in c]
                bnb(idx + 1, new_set, new_remaining)
        
        bnb(0, set(), sorted_cliques)
        
        if best_solution is not None:
            return sorted(best_solution)
        return fast_greedy_hitting_set(cliques, n)

    def randomized_greedy(cliques, n, seed):
        """Randomized greedy with different orderings."""
        rng = random.Random(seed)
        if not cliques:
            return []
        
        m = len(cliques)
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].append(i)
        
        uncovered = [True] * m
        result = set()
        
        while True:
            if not any(uncovered):
                break
            
            # Compute scores with random perturbation
            best_v = -1
            best_score = -1
            for v in range(n):
                if v in result:
                    continue
                count = sum(1 for i in vertex_to_cliques[v] if uncovered[i])
                if count == 0:
                    continue
                # Add small random noise to break ties
                score = count + rng.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                break
            
            result.add(best_v)
            for i in vertex_to_cliques[best_v]:
                uncovered[i] = False
        
        return sorted(result)

    def find_min_hitting_set(cliques, n, time_budget=30.0):
        """Try multiple methods and return the best hitting set."""
        if not cliques:
            return []
        
        start = time.time()
        solutions = {}
        
        # Method 1: Fast greedy
        sol_greedy = fast_greedy_hitting_set(cliques, n)
        solutions['greedy'] = sol_greedy
        
        # Method 2: LP rounding
        if time.time() - start < time_budget * 0.5:
            sol_lp = lp_rounding_hitting_set(cliques, n)
            solutions['lp'] = sol_lp
        
        # Method 3: Branch and bound
        if time.time() - start < time_budget * 0.7:
            bnb_time = min(5.0, time_budget - (time.time() - start) - 5.0)
            if bnb_time > 1.0:
                sol_bnb = branch_and_bound_hitting_set(cliques, n, time_limit=bnb_time)
                solutions['bnb'] = sol_bnb
        
        # Method 4: Randomized greedy (multiple restarts)
        if time.time() - start < time_budget * 0.8:
            for restart_seed in range(5):
                if time.time() - start > time_budget:
                    break
                sol_rand = randomized_greedy(cliques, n, restart_seed)
                solutions[f'rand_{restart_seed}'] = sol_rand
        
        # Pick the best initial solution
        best = min(solutions.values(), key=len)
        
        # Local search improvement
        if time.time() - start < time_budget:
            improved = local_search_improve(best, cliques, n, max_iterations=50)
            if len(improved) < len(best):
                best = improved
        
        return best

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # Fallback: return all vertices
            results.append(list(range(n)))
            continue
        
        edges = generate_edges(n, p, seed)
        cliques = enumerate_maximal_cliques(n, edges)
        
        if not cliques:
            results.append([])
            continue
        
        # Allocate time budget based on remaining time
        remaining_time = DEADLINE - time.time()
        time_budget = min(30.0, remaining_time * 0.9)
        
        hitting_set = find_min_hitting_set(cliques, n, time_budget=time_budget)
        results.append(hitting_set)
    
    return results
# EVOLVE-BLOCK-END
