"""Decompose instance graphs into edge-disjoint cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 140

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edge_set):
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_2_core(edge_set):
        if not edge_set:
            return set()
        adj = build_adj(edge_set)
        deg = {v: len(neigh) for v, neigh in adj.items()}
        q = deque([v for v in deg if deg[v] < 2])
        in_core = {v: True for v in deg}
        while q:
            v = q.popleft()
            if not in_core[v]:
                continue
            in_core[v] = False
            for nb in adj[v]:
                if in_core[nb]:
                    deg[nb] -= 1
                    if deg[nb] < 2:
                        q.append(nb)
        return {(u, v) for u, v in edge_set if in_core.get(u, False) and in_core.get(v, False)}

    def find_longest_cycle(adj, max_depth=30):
        if not adj:
            return None
        vertices = sorted(adj.keys())
        best = None
        best_len = 0

        for start in vertices:
            # Iterative DFS with proper backtracking
            visited = {start}
            path = [start]
            # stack holds (vertex, iterator_over_neighbors)
            stack = [(start, iter(sorted(adj[start])))]
            while stack:
                u, it = stack[-1]
                advanced = False
                for nb in it:
                    if nb == start and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best = path[:]
                        break
                    if nb not in visited and len(path) < max_depth:
                        visited.add(nb)
                        path.append(nb)
                        stack.append((nb, iter(sorted(adj[nb]))))
                        advanced = True
                        break
                if not advanced:
                    stack.pop()
                    path.pop()
                    if path:
                        visited.discard(path[-1])
                    else:
                        visited.discard(u)
            if best and best_len >= max_depth - 2:
                break
        return best

    def greedy_decompose(edge_set):
        rem = set(edge_set)
        cycles = []
        while rem:
            core = get_2_core(rem)
            if not core:
                break
            adj = build_adj(core)
            cycle = find_longest_cycle(adj, max_depth=25)
            if not cycle:
                break
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                rem.discard(tuple(sorted((u, v))))
            cycles.append(cycle)
        return cycles, rem

    def decompose(edges, n, time_budget):
        if not edges:
            return []
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        best_pieces = None
        best_count = len(edge_set)
        
        num_restarts = 12
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            cycles, rem = greedy_decompose(edge_set)
            count = len(cycles) + len(rem)
            if count < best_count:
                best_count = count
                best_pieces = [c[:] for c in cycles] + [list(e) for e in rem]
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edge_set]
        return best_pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() > deadline - 3:
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, deadline - time.time())
        results.append(pieces)
    return results
