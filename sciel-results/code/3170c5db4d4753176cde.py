import time
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)
import random


def construct():
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before the 150 s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_any_cycle(adj):
        """Guaranteed cycle finder (undirected DFS back-edge). Returns vertex list or None."""
        color = {}
        parent = {}

        def dfs(u):
            color[u] = 1
            for w in adj[u]:
                if w == parent.get(u):
                    continue
                if color.get(w, 0) == 0:
                    parent[w] = u
                    res = dfs(w)
                    if res is not None:
                        return res
                elif color.get(w, 0) == 1:
                    cyc = []
                    cur = u
                    while cur != w:
                        cyc.append(cur)
                        cur = parent[cur]
                    cyc.append(w)
                    cyc.reverse()
                    if len(cyc) >= 3:
                        return cyc
            color[u] = 2
            return None

        for v in list(adj.keys()):
            if color.get(v, 0) == 0:
                res = dfs(v)
                if res is not None:
                    return res
        return None

    def find_long_cycle(adj, budget=50000):
        """Bounded DFS searching for the longest simple cycle through high-degree starts."""
        starts = [v for v in adj if len(adj[v]) >= 2]
        if not starts:
            return None
        starts.sort(key=lambda v: -len(adj[v]))
        K = min(12, len(starts))
        sub_budget = max(500, budget // K)

        best_cycle = None
        best_len = 0
        for start in starts[:K]:
            state = {'visited': 0, 'path': [start], 'on_path': {start},
                     'best': None, 'best_len': 0}

            def dfs(u):
                if state['visited'] > sub_budget:
                    return
                nbrs = sorted(adj[u], key=lambda x: -len(adj[x]))
                for w in nbrs:
                    state['visited'] += 1
                    if state['visited'] > sub_budget:
                        return
                    if w == start and len(state['path']) >= 3:
                        if len(state['path']) > state['best_len']:
                            state['best_len'] = len(state['path'])
                            state['best'] = list(state['path'])
                    elif w not in state['on_path']:
                        state['on_path'].add(w)
                        state['path'].append(w)
                        dfs(w)
                        state['path'].pop()
                        state['on_path'].discard(w)

            dfs(start)
            if state['best'] is not None and state['best_len'] > best_len:
                best_len = state['best_len']
                best_cycle = state['best']
        return best_cycle

    def find_hamiltonian_cycle(adj, n):
        """Try to find a Hamiltonian cycle using DFS with pruning."""
        if n < 3:
            return None
        # Check if graph is connected and has enough edges
        if len(adj) < n:
            return None
        
        # Use a greedy + backtracking approach
        # Start from vertex 0 (or highest degree)
        start = max(adj.keys(), key=lambda v: len(adj[v])) if adj else None
        if start is None:
            return None
        
        visited = [False] * n
        visited[start] = True
        path = [start]
        found = False
        steps = [0]
        max_steps = 50000
        
        def hdfs(u, depth):
            nonlocal found
            if found:
                return
            steps[0] += 1
            if steps[0] > max_steps:
                return
            if depth == n:
                if start in adj[u]:
                    path.append(start)
                    found = True
                return
            
            # Pruning: if remaining unvisited vertices can't be connected
            unvisited_count = n - depth
            if unvisited_count > 0:
                # Check if any unvisited vertex is reachable
                has_unvisited_neighbor = False
                for w in adj[u]:
                    if not visited[w]:
                        has_unvisited_neighbor = True
                        break
                if not has_unvisited_neighbor:
                    return
            
            nbrs = sorted(adj[u], key=lambda x: -len(adj[x]))
            for w in nbrs:
                if w == start:
                    continue  # Only close at the end
                if not visited[w]:
                    visited[w] = True
                    path.append(w)
                    hdfs(w, depth + 1)
                    path.pop()
                    visited[w] = False
                    if found:
                        return
        
        hdfs(start, 1)
        if found:
            return path
        return None

    def decompose(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        remaining = set(edges)
        pieces = []
        
        # Try to find Hamiltonian cycle first
        n = max(max(u for u, v in edges), max(v for u, v in edges)) + 1 if edges else 0
        ham_cycle = find_hamiltonian_cycle(adj, n)
        if ham_cycle is not None:
            for i in range(len(ham_cycle)):
                a, b = ham_cycle[i], ham_cycle[(i + 1) % len(ham_cycle)]
                e = (min(a, b), max(a, b))
                remaining.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)
            pieces.append(list(ham_cycle))
        
        # Continue with long cycles for remaining edges
        while time.time() - start_time < DEADLINE:
            if not remaining:
                break
            cycle = find_long_cycle(adj)
            if cycle is None:
                cycle = find_any_cycle(adj)
            if cycle is None:
                break
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (min(a, b), max(a, b))
                remaining.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)
            pieces.append(list(cycle))
        
        for e in remaining:
            pieces.append([e[0], e[1]])
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
