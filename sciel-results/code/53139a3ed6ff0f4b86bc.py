import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    def greedy_longest_path(remaining_edges, n, rng, time_limit):
        """Find a long simple path in the graph formed by remaining_edges."""
        if not remaining_edges:
            return []
        
        edge_set = set(remaining_edges)
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        if not adj:
            return []
        
        best_path = []
        nodes = list(adj.keys())
        rng.shuffle(nodes)
        
        t0 = time.time()
        for start in nodes[:min(len(nodes), 25)]:
            if time.time() - t0 > time_limit:
                break
            path = [start]
            visited = {start}
            used_edges = set()
            current = start
            
            while True:
                extended = False
                neighbors = list(adj[current])
                # Prefer neighbors with fewer options (Warnsdorff)
                neighbors.sort(key=lambda x: len(adj[x]))
                rng.shuffle(neighbors)
                
                for nb in neighbors:
                    e = (min(current, nb), max(current, nb))
                    if e in edge_set and e not in used_edges and nb not in visited:
                        path.append(nb)
                        visited.add(nb)
                        used_edges.add(e)
                        current = nb
                        extended = True
                        break
                if not extended:
                    break
            
            if len(path) > len(best_path):
                best_path = path[:]
        
        return best_path

    def decompose(edges, n, rng, time_budget):
        """Greedy decomposition: repeatedly extract longest path."""
        edge_set = set(edges)
        paths = []
        remaining = set(edge_set)
        t0 = time.time()
        
        while remaining and (time.time() - t0) < time_budget:
            path = greedy_longest_path(remaining, n, rng, time_budget * 0.1)
            
            if not path:
                # Add remaining edges as single-edge paths
                for e in remaining:
                    paths.append(list(e))
                break
            
            path_edges = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                path_edges.add((min(u, v), max(u, v)))
            
            remaining -= path_edges
            paths.append(path)
        
        # Handle any leftover edges
        for e in remaining:
            paths.append(list(e))
        
        return paths

    def try_merge(paths, edge_set, n, rng, time_budget):
        """Try to merge paths that can be concatenated."""
        t0 = time.time()
        changed = True
        while changed and (time.time() - t0) < time_budget:
            changed = False
            for i in range(len(paths)):
                if (time.time() - t0) > time_budget:
                    break
                for j in range(i + 1, len(paths)):
                    pi = paths[i]
                    pj = paths[j]
                    # Try all 4 combinations of directions
                    for perm_i in [pi, pi[::-1]]:
                        for perm_j in [pj, pj[::-1]]:
                            if perm_i[-1] == perm_j[0]:
                                new_path = perm_i + perm_j[1:]
                                # Check simplicity
                                if len(set(new_path)) != len(new_path):
                                    continue
                                # Check all edges exist
                                valid = True
                                for k in range(len(new_path) - 1):
                                    u, v = new_path[k], new_path[k+1]
                                    e = (min(u, v), max(u, v))
                                    if e not in edge_set:
                                        valid = False
                                        break
                                if valid:
                                    paths[i] = new_path
                                    del paths[j]
                                    changed = True
                                    break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        return paths

    def validate(paths, edges):
        """Check if paths form a valid edge decomposition."""
        edge_set = set(edges)
        covered = set()
        for path in paths:
            if len(path) < 2:
                return False
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e = (min(u, v), max(u, v))
                if e not in edge_set:
                    return False
                covered.add(e)
        return covered == edge_set

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    for idx, spec in enumerate(specs):
        n, p, seed = spec
        edges = instance(n, p, seed)
        m = len(edges)

        if m == 0:
            all_results.append([])
            continue

        # Allocate ~45s per graph
        per_graph = 45.0
        graph_deadline = start_time + per_graph * (idx + 1)
        
        best_paths = None
        best_count = m + 1
        
        # Multiple randomized restarts
        for attempt in range(50):
            if time.time() >= graph_deadline - 2.0:
                break
            
            rng = random.Random(seed * 10000 + attempt * 7 + 42)
            time_budget = min(1.5, graph_deadline - time.time() - 1.0)
            if time_budget <= 0.1:
                break
            
            paths = decompose(edges, n, rng, time_budget)
            
            if not paths:
                continue
            
            # Local improvement: merge paths
            merge_budget = min(1.0, graph_deadline - time.time() - 1.0)
            if merge_budget > 0.1:
                paths = try_merge(paths, set(edges), n, rng, merge_budget)
            
            # Validate
            if validate(paths, edges):
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = [p[:] for p in paths]
        
        if best_paths is None:
            best_paths = [list(e) for e in edges]
        
        all_results.append(best_paths)

    return all_results
