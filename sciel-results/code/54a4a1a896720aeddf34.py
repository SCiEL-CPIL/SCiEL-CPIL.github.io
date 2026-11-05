import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    graphs = [instance(*spec) for spec in specs]

    def find_cycle_vertices(edges_set, rng):
        if not edges_set:
            return None

        adj = defaultdict(list)
        for u, v in edges_set:
            adj[u].append(v)
            adj[v].append(u)

        vertices = set()
        for u, v in edges_set:
            vertices.add(u)
            vertices.add(v)

        if not vertices:
            return None

        v_list = list(vertices)
        rng.shuffle(v_list)

        for start in v_list:
            state = {v: 0 for v in vertices}
            stack = [(start, iter(adj[start]))]
            path = [start]
            state[start] = 1

            while stack:
                u, neighbors = stack[-1]
                advanced = False
                for v in neighbors:
                    if state[v] == 0:
                        state[v] = 1
                        path.append(v)
                        stack.append((v, iter(adj[v])))
                        advanced = True
                        break
                    elif state[v] == 1:
                        idx = path.index(v)
                        cycle_nodes = path[idx:]
                        return cycle_nodes
                if not advanced:
                    state[u] = 2
                    path.pop()
                    stack.pop()

        return None

    def decompose(edges, rng):
        initial_edges = set(edges)
        best_pieces = None
        best_count = float('inf')

        # Run until deadline
        while time.time() < deadline:
            remaining_edges = set(initial_edges)
            pieces = []

            while remaining_edges:
                cycle_nodes = find_cycle_vertices(remaining_edges, rng)
                if cycle_nodes:
                    pieces.append(cycle_nodes)
                    for i in range(len(cycle_nodes)):
                        u1 = cycle_nodes[i]
                        u2 = cycle_nodes[(i + 1) % len(cycle_nodes)]
                        edge = tuple(sorted((u1, u2)))
                        remaining_edges.remove(edge)
                else:
                    e = next(iter(remaining_edges))
                    pieces.append([e[0], e[1]])
                    remaining_edges.remove(e)

            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces

        return best_pieces

    results = []
    for i, edges in enumerate(graphs):
        # Use a different random seed for each graph to decorrelate failures
        rng = random.Random(100 + i)
        pieces = decompose(edges, rng)
        results.append(pieces)

    return results
