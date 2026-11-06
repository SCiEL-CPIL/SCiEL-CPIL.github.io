"""Rewritten program: path decomposition via odd-vertex pairing + Eulerian circuit + greedy merge."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    t_start = time.time()
    DEADLINE = 135.0  # Safety margin from 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    def decompose(edges):
        if not edges:
            return []
        
        edge_set = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        all_verts = set()
        for u, v in edges:
            all_verts.add(u)
            all_verts.add(v)
        
        deg = {v: len(adj[v]) for v in all_verts}
        odd_verts = sorted([v for v in all_verts if deg[v] % 2 == 1])
        
        def find_path_bfs(start, end, forbidden):
            if start == end:
                return [start]
            visited = {start}
            parent = {start: (None, None)}
            queue = deque([start])
            while queue:
                v = queue.popleft()
                for nbr in sorted(adj[v]):
                    e = (min(v, nbr), max(v, nbr))
                    if e in forbidden:
                        continue
                    if nbr not in visited:
                        visited.add(nbr)
                        parent[nbr] = (v, e)
                        if nbr == end:
                            path = [nbr]
                            cur = nbr
                            while cur != start:
                                pv, pe = parent[cur]
                                path.append(pv)
                                cur = pv
                            path.reverse()
                            return path
                        queue.append(nbr)
            return None

        def eulerian_circuit(rem_edges, start):
            """Find Eulerian circuit in graph with all even degrees."""
            adj_e = defaultdict(list)
            for u, v in rem_edges:
                adj_e[u].append(v)
                adj_e[v].append(u)
            
            # Make mutable
            for v in adj_e:
                adj_e[v] = list(adj_e[v])
            
            stack = [start]
            circuit = []
            while stack:
                v = stack[-1]
                if adj_e.get(v):
                    nbr = adj_e[v].pop()
                    # Remove v from nbr's list
                    if v in adj_e.get(nbr, []):
                        adj_e[nbr].remove(v)
                    stack.append(nbr)
                else:
                    circuit.append(stack.pop())
            circuit.reverse()
            return circuit

        paths = []
        remaining = set(edges)
        
        if len(odd_verts) == 0:
            start = next(iter(all_verts))
            circuit = eulerian_circuit(remaining, start)
            paths.append(circuit)
        else:
            pairs = [(odd_verts[i], odd_verts[i + 1]) for i in range(0, len(odd_verts), 2)]
            
            for u, v in pairs:
                path = find_path_bfs(u, v, set())
                if path is None:
                    path = [u, v]  # fallback, shouldn't happen
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    e = (min(a, b), max(a, b))
                    remaining.discard(e)
                paths.append(path)
            
            if remaining:
                rem_verts = set()
                for u, v in remaining:
                    rem_verts.add(u)
                    rem_verts.add(v)
                if rem_verts:
                    start = next(iter(rem_verts))
                    circuit = eulerian_circuit(remaining, start)
                    paths.append(circuit)
        
        # Greedy merge
        def can_merge(p1, p2):
            cands = []
            if p1[-1] == p2[-1]:
                cands.append(p1 + p2[-2::-1])
            if p1[-1] == p2[0]:
                cands.append(p1 + p2[1:])
            if p1[0] == p2[0]:
                cands.append(p1[::-1] + p2[1:])
            if p1[0] == p2[-1]:
                cands.append(p1[::-1] + p2)
            for c in cands:
                if len(c) != len(set(c)):
                    continue
                ok = True
                for i in range(len(c) - 1):
                    a, b = c[i], c[i + 1]
                    if (min(a, b), max(a, b)) not in edge_set:
                        ok = False
                        break
                if ok:
                    return c
            return None
        
        changed = True
        while changed and (time.time() - t_start) < DEADLINE - 5:
            changed = False
            new_paths = list(paths)
            for i in range(len(new_paths)):
                for j in range(i + 1, len(new_paths)):
                    m = can_merge(new_paths[i], new_paths[j])
                    if m is not None:
                        new_paths[i] = m
                        new_paths[j] = None
                        changed = True
                        break
                if changed:
                    break
            paths = [p for p in new_paths if p is not None]
        
        return paths

    results = []
    for spec in specs:
        if (time.time() - t_start) > DEADLINE - 10:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
