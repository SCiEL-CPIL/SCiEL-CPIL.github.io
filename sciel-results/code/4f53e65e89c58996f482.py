import time
import sys
from itertools import combinations
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    from random import Random
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    DEADLINE = 140.0  # safety margin before 150s kill
    start_time = time.time()

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def enumerate_cycles(n, adj, max_len=12):
        """Enumerate all simple cycles up to max_len vertices.
        Returns list of cycles, each cycle is a tuple of vertices (first == last not included, closing edge implied).
        We store each cycle as a sorted tuple of its edge set to deduplicate."""
        cycles = set()
        adj_set = [set() for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                adj_set[u].add(v)

        # For each starting vertex, do DFS to find cycles
        for start in range(n):
            # Only start from the smallest vertex in the cycle to avoid duplicates
            stack = [(start, [start], set([start]))]
            while stack:
                node, path, visited = stack.pop()
                for nb in sorted(adj_set[node]):
                    if nb == start and len(path) >= 3:
                        # Found a cycle: path + closing edge
                        cycle_edges = frozenset(
                            [tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)]
                            + [tuple(sorted((path[-1], start)))]
                        )
                        cycles.add(cycle_edges)
                    elif nb not in visited and nb > start and len(path) < max_len:
                        stack.append((nb, path + [nb], visited | {nb}))
        return [list(c) for c in cycles]

    def solve_decomposition(n, edges):
        """Find minimum piece decomposition using ILP for max edge-disjoint cycles."""
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        edge_list = [tuple(e) for e in edges]
        edge_index = {e: i for i, e in enumerate(edge_list)}
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)

        # Enumerate cycles
        cycles = enumerate_cycles(n, adj, max_len=10)
        m = len(edge_list)
        k = len(cycles)

        if k == 0:
            # No cycles found, all edges are individual pieces
            return [list(e) for e in edge_list]

        # ILP: maximize sum(x_i) subject to for each edge, sum of x_i over cycles using that edge <= 1
        # x_i in {0,1}
        
        # Objective: maximize sum(x_i) -> minimize -sum(x_i)
        c = -np.ones(k)

        # Constraints: for each edge j, sum_{i: edge j in cycle i} x_i <= 1
        A = np.zeros((m, k))
        for i, cycle_edges in enumerate(cycles):
            for e in cycle_edges:
                if e in edge_index:
                    A[edge_index[e], i] = 1.0

        cons = LinearConstraint(A, ub=np.ones(m), lb=-np.inf)
        bounds = Bounds(lb=np.zeros(k), ub=np.ones(k))

        try:
            res = milp(c=c, constraints=cons, integrality=np.ones(k), bounds=bounds,
                       options={"time_limit": 30})
            if res.status == 0:  # optimal
                x = res.x
                selected = [i for i in range(k) if x[i] > 0.5]
            elif res.status == 9:  # time limit reached, use best integer solution
                # scipy milp doesn't directly give partial, so fall back to greedy
                selected = greedy_cycle_packing(cycles, edge_index, m)
            else:
                selected = greedy_cycle_packing(cycles, edge_index, m)
        except Exception:
            selected = greedy_cycle_packing(cycles, edge_index, m)

        # Build result
        used_edges = set()
        pieces = []
        for i in selected:
            cycle_edges = cycles[i]
            # Reconstruct cycle vertex sequence from edges
            cycle_verts = reconstruct_cycle(cycle_edges)
            if cycle_verts:
                pieces.append(cycle_verts)
                used_edges.update(cycle_edges)

        # Remaining edges as individual pieces
        for e in edge_list:
            if e not in used_edges:
                pieces.append([e[0], e[1]])

        return pieces

    def greedy_cycle_packing(cycles, edge_index, m):
        """Greedy fallback: pick cycles that don't conflict, sorted by length (shorter first)."""
        used = [0] * m
        selected = []
        # Sort by number of edges (prefer shorter cycles)
        cycle_info = []
        for i, ce in enumerate(cycles):
            edge_indices = [edge_index[e] for e in ce if e in edge_index]
            cycle_info.append((len(edge_indices), i, edge_indices))
        cycle_info.sort(key=lambda x: (x[0], -x[1]))
        
        for length, idx, edge_indices in cycle_info:
            if all(used[j] == 0 for j in edge_indices):
                for j in edge_indices:
                    used[j] = 1
                selected.append(idx)
        return selected

    def reconstruct_cycle(edge_set):
        """Given a set of edges forming a cycle, reconstruct the vertex sequence."""
        # Build adjacency from just these edges
        adj = defaultdict(list)
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find a vertex with degree 2 (all vertices in a cycle have degree 2)
        if not adj:
            return None
        start = next(iter(adj))
        path = [start]
        visited_edges = set()
        current = start
        for _ in range(len(edge_set)):
            nexts = adj[current]
            nxt = None
            for nb in nexts:
                e = tuple(sorted((current, nb)))
                if e not in visited_edges:
                    nxt = nb
                    break
            if nxt is None:
                return None
            visited_edges.add(tuple(sorted((current, nxt))))
            path.append(nxt)
            current = nxt
        # path should end at start
        if path[-1] != path[0]:
            return None
        # Remove the repeated start at the end
        return path[:-1]

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - start_time > DEADLINE:
            # Fallback: all edges as individual pieces
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        pieces = solve_decomposition(spec[0], edges)
        results.append(pieces)

    return results

# EVOLVE-BLOCK-END
