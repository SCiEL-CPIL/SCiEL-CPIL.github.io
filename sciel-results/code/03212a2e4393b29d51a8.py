"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
        return adj

    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        adj_sets = [set(i for i in range(n) if adj[v][i]) for v in range(n)]

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return

            # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
            best_pivot = None
            best_count = -1
            for v in list(P) + list(X):
                count = len(P & adj_sets[v])
                if count > best_count:
                    best_count = count
                    best_pivot = v

            candidates = P
            if best_pivot is not None:
                candidates = P - adj_sets[best_pivot]

            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj_sets[v]
                new_X = X & adj_sets[v]
                bk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)

        bk(frozenset(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)

        if not cliques:
            return []

        # MILP formulation:
        # x_v in {0,1} for v in 0..n-1
        # For each clique C: sum_{v in C} x_v >= 1
        # Minimize: sum_v x_v

        num_cliques = len(cliques)
        # Build constraint matrix: A @ x >= 1
        # A is num_cliques x n
        rows = []
        for clique in cliques:
            row = [0] * n
            for v in clique:
                row[v] = 1
            rows.append(row)

        A = np.array(rows, dtype=float)
        b_lower = np.ones(num_cliques)
        b_upper = np.full(num_cliques, np.inf)

        c = np.ones(n, dtype=float)  # minimize sum x_v

        constraints = LinearConstraint(A, b_lower, b_upper)
        bounds = Bounds(0, 1)

        # Use milp with integer variables
        integrality = np.ones(n)  # all variables integer

        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

        if result.success:
            # Extract solution
            solution = result.x
            selected = [int(v) for v in range(n) if solution[v] > 0.5]
            return sorted(selected)
        else:
            # Fallback: greedy approach
            return greedy_transversal(cliques, n)

    def greedy_transversal(cliques, n):
        """Greedy set cover as fallback."""
        if not cliques:
            return []
        uncovered = set(cliques)
        selected = set()
        while uncovered:
            # Find vertex that covers most uncovered cliques
            best_v = -1
            best_count = 0
            for v in range(n):
                count = sum(1 for c in uncovered if v in c)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1 or best_count == 0:
                break
            selected.add(best_v)
            uncovered = {c for c in uncovered if best_v not in c}
        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        results.append(solve_instance(n, p, seed))
    return results
# EVOLVE-BLOCK-END
