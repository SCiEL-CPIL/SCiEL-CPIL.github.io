"""Packing edge-disjoint Hamiltonian cycles on the 21-vertex graph."""
import time
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}


def construct():
    deadline = time.time() + 80.0
    n = P['n']
    edges = P['edges']
    m = len(edges)

    # Build edge lookup: (u,v) -> edge index
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i

    # Adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def find_hamiltonian_cycle(frozen_edges):
        """Find a Hamiltonian cycle avoiding edges in frozen_edges (a set of edge indices)."""
        # Build available adjacency
        avail = defaultdict(set)
        for u in range(n):
            for v in adj[u]:
                ei = edge_idx[(u, v)]
                if ei not in frozen_edges:
                    avail[u].add(v)

        # Check minimum degree
        for v in range(n):
            if len(avail[v]) < 2:
                return None

        # DFS with pruning
        path = [0]
        used_e = set()
        visited = {0}

        def get_degree(v):
            return len(avail[v])

        def dfs():
            if time.time() > deadline:
                return False
            cur = path[-1]
            if len(path) == n:
                # Check closing edge
                if 0 in avail[cur]:
                    ei = edge_idx[(cur, 0)]
                    if ei not in frozen_edges and ei not in used_e:
                        return True
                return False

            # MRV: sort neighbors by remaining degree
            nbrs = sorted(avail[cur], key=lambda x: get_degree(x))
            for nb in nbrs:
                if nb in visited:
                    continue
                ei = edge_idx[(cur, nb)]
                if ei in frozen_edges or ei in used_e:
                    continue
                # Pruning: ensure nb can still connect to unvisited vertices
                # (basic connectivity check)
                if nb != 0 and len([x for x in avail[nb] if x not in visited]) < 1:
                    # nb would be isolated among unvisited if we go here, unless it's the last
                    if len(path) < n - 1:
                        continue
                visited.add(nb)
                path.append(nb)
                used_e.add(ei)
                if dfs():
                    return True
                used_e.remove(ei)
                path.pop()
                visited.remove(nb)
            return False

        if dfs():
            return path
        return None

    cycles = []
    frozen = set()

    # Find first cycle
    c1 = find_hamiltonian_cycle(frozen)
    if c1 is None:
        return [list(range(n))]

    cycles.append([int(v) for v in c1])
    for i in range(len(c1)):
        u, v = c1[i], c1[(i + 1) % len(c1)]
        frozen.add(edge_idx[(u, v)])

    # Find second cycle
    c2 = find_hamiltonian_cycle(frozen)
    if c2 is not None:
        cycles.append([int(v) for v in c2])

    return cycles
