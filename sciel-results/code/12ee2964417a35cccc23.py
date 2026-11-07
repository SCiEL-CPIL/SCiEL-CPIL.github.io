"""Rewritten program: packing edge-disjoint Hamiltonian cycles."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin under 90s kill

    n = P['n']
    edges = P['edges']
    rng = random.Random(12345)

    # Build adjacency and edge set
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    def ekey(u, v):
        return (min(u, v), max(u, v))

    def is_edge(u, v):
        return ekey(u, v) in edge_set

    def find_ham_cycle(used_edges, start_vertex=0, max_nodes=20000000):
        """DFS for a Hamiltonian cycle avoiding used_edges."""
        path = [start_vertex]
        visited = {start_vertex}
        nodes_explored = [0]

        def dfs():
            if time.time() > deadline:
                return None
            nodes_explored[0] += 1
            if nodes_explored[0] > max_nodes:
                return None

            if len(path) == n:
                last = path[-1]
                e = ekey(last, start_vertex)
                if is_edge(last, start_vertex) and e not in used_edges:
                    return path[:]
                return None

            last = path[-1]
            unvisited = [v for v in adj[last] if v not in visited]
            # Prune: if any unvisited vertex has no unused edge to any unvisited-or-last
            # (i.e., its only possible entry/exit is blocked), skip
            # Simpler pruning: sort by remaining unused-degree (warnsdorff)
            def rem_deg(v):
                c = 0
                for w in adj[v]:
                    if w not in visited or w == start_vertex:
                        if ekey(v, w) not in used_edges:
                            c += 1
                return c
            unvisited.sort(key=lambda v: rem_deg(v))
            for nb in unvisited:
                e = ekey(last, nb)
                if e in used_edges:
                    continue
                path.append(nb)
                visited.add(nb)
                res = dfs()
                if res is not None:
                    return res
                path.pop()
                visited.remove(nb)
            return None

        return dfs()

    # Greedy: find up to 3 edge-disjoint Hamiltonian cycles
    used_edges = set()
    cycles = []
    for attempt in range(3):
        if time.time() > deadline:
            break
        found = False
        # Try several start vertices / orderings for robustness
        starts = list(range(n))
        rng.shuffle(starts)
        for s in starts:
            if time.time() > deadline:
                break
            c = find_ham_cycle(used_edges, start_vertex=s)
            if c is not None:
                for i in range(len(c)):
                    u = c[i]
                    v = c[(i + 1) % len(c)]
                    used_edges.add(ekey(u, v))
                cycles.append(c)
                found = True
                break
        if not found:
            break

    # Validate: each cycle uses all n distinct vertices, edges are host edges, edge-disjoint
    def valid_solution(cycles):
        seen = set()
        for c in cycles:
            if len(c) != n:
                return False
            if len(set(c)) != n:
                return False
            for i in range(n):
                u = c[i]
                v = c[(i + 1) % n]
                e = ekey(u, v)
                if e not in edge_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
        return True

    if not valid_solution(cycles):
        # Fallback: single Hamiltonian cycle (seed)
        c = find_ham_cycle(set(), start_vertex=0)
        if c is not None and valid_solution([c]):
            cycles = [c]
        else:
            cycles = []

    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    return result
# EVOLVE-BLOCK-END
