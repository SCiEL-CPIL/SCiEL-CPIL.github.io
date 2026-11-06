"""Construct a large (5,5)-free graph by extending the Paley graph vertex by vertex."""
import random
from itertools import combinations

def construct():
    """
    Extends the Paley graph on 37 vertices by adding vertices one at a time,
    choosing edges to avoid K5 and independent 5-sets.
    """
    CLIQUE = 5
    INDEPENDENT = 5
    MAX_VERTICES = 46

    def paley_graph(n):
        """Constructs Paley graph on n vertices (n must be prime, n % 4 == 1)."""
        Q = set()
        for i in range(1, n):
            Q.add((i * i) % n)
        S = Q
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in S:
                    edges.append((i, j))
        return n, edges

    def find_all_k4(adj, n):
        """Find all K4s in the graph."""
        k4s = []
        for v0 in range(n):
            nbs = [v for v in range(n) if v != v0 and adj[v0][v]]
            if len(nbs) < 3:
                continue
            for combo in combinations(nbs, 3):
                if all(adj[a][b] for a, b in combinations(combo, 2)):
                    k4s.append(tuple(sorted([v0] + list(combo))))
        return k4s

    def find_all_indep4(adj, n):
        """Find all independent 4-sets in the graph."""
        indep4s = []
        for v0 in range(n):
            non_nbs = [v for v in range(n) if v != v0 and not adj[v0][v]]
            if len(non_nbs) < 3:
                continue
            for combo in combinations(non_nbs, 3):
                if all(not adj[a][b] for a, b in combinations(combo, 2)):
                    indep4s.append(tuple(sorted([v0] + list(combo))))
        return indep4s

    # Start with Paley graph on 37 vertices
    n0, edges0 = paley_graph(37)

    # Build adjacency matrix
    adj = [[False] * n0 for _ in range(n0)]
    for i, j in edges0:
        adj[i][j] = True
        adj[j][i] = True

    # Find all K4s and independent 4-sets in the initial graph
    k4s = find_all_k4(adj, n0)
    indep4s = find_all_indep4(adj, n0)

    best_n = n0
    best_edges = edges0[:]

    # Try to add vertices one at a time
    for new_n in range(n0 + 1, MAX_VERTICES + 1):
        random.seed(new_n * 7919)

        found_pattern = None

        # Try many random patterns
        for attempt in range(1000):
            pattern = [random.random() < 0.5 for _ in range(new_n - 1)]

            valid = True

            # Check K5: new vertex + K4 where new connects to all 4
            for k4 in k4s:
                if all(pattern[v] for v in k4):
                    valid = False
                    break

            if not valid:
                continue

            # Check independent 5: new vertex + indep4 where new connects to none
            for indep4 in indep4s:
                if all(not pattern[v] for v in indep4):
                    valid = False
                    break

            if valid:
                found_pattern = pattern
                break

        if found_pattern is not None:
            adj.append([False] * new_n)
            for v in range(new_n):
                adj[v].append(False)

            new_vertex = new_n - 1
            for v in range(new_n - 1):
                if found_pattern[v]:
                    adj[new_vertex][v] = True
                    adj[v][new_vertex] = True

            k4s = find_all_k4(adj, new_n)
            indep4s = find_all_indep4(adj, new_n)

            best_n = new_n
            best_edges = []
            for i in range(new_n):
                for j in range(i + 1, new_n):
                    if adj[i][j]:
                        best_edges.append((i, j))
        else:
            break

    return best_n, best_edges
