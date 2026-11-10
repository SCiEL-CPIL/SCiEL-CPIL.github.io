"""
Path decomposition for Gallai's conjecture instances.
Combines the current program's structure with the inspiration's multi-strategy
restart approach and a more robust local improvement phase.
"""
import time
import random
from collections import defaultdict
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _greedy_walk(adj, remaining, deg, start_v, rng=None, randomize=False, mode="low_deg"):
        """
        Greedy walk that builds a simple path by extending from start_v.
        At each step, picks a neighbor that is not yet visited and has an edge in remaining.
        """
        path = [start_v]
        visited = {start_v}
        
        while True:
            v = path[-1]
            valid_neighbors = []
            for u in adj[v]:
                if u not in visited:
                    e = (min(v, u), max(v, u))
                    if e in remaining:
                        valid_neighbors.append(u)
            
            if not valid_neighbors:
                break
            
            if randomize and rng is not None:
                if mode == "high_deg":
                    weights = [float(deg.get(u, 0) + 1) for u in valid_neighbors]
                elif mode == "low_deg":
                    weights = [1.0 / (deg.get(u, 0) + 1) for u in valid_neighbors]
                else:
                    weights = [1.0 + deg.get(u, 0) * 0.5 for u in valid_neighbors]
                
                total_w = sum(weights)
                if total_w == 0:
                    next_v = valid_neighbors[0]
                else:
                    r = rng.random() * total_w
                    cum = 0
                    next_v = valid_neighbors[-1]
                    for i, u in enumerate(valid_neighbors):
                        cum += weights[i]
                        if r <= cum:
                            next_v = u
                            break
            else:
                if mode == "high_deg":
                    next_v = max(valid_neighbors, key=lambda u: deg.get(u, 0))
                elif mode == "low_deg":
                    next_v = min(valid_neighbors, key=lambda u: deg.get(u, 0))
                else:
                    next_v = max(valid_neighbors, key=lambda u: (deg.get(u, 0), -u))
            
            e = (min(v, next_v), max(v, next_v))
            remaining.discard(e)
            deg[v] -= 1
            deg[next_v] -= 1
            path.append(next_v)
            visited.add(next_v)
        
        return path

    def _path_cover_single(edges, deadline, rng=None, randomize=False, mode="low_deg", start_strategy="leaf", shuffle_adj=False):
        """
        Single pass greedy path cover.
        Repeatedly pick a start vertex and extend greedily to form a simple path,
        removing edges as they are used.
        """
        adj = _build_adj(edges)
        
        if shuffle_adj and rng is not None:
            for v in adj:
                adj[v] = set(adj[v])
                adj[v] = list(adj[v])
                rng.shuffle(adj[v])
                adj[v] = set(adj[v])

        remaining = set(map(tuple, edges))
        paths = []
        
        deg = defaultdict(int)
        for e in remaining:
            deg[e[0]] += 1
            deg[e[1]] += 1
        
        while remaining and time.time() < deadline:
            start_candidates = set()
            for e in remaining:
                start_candidates.add(e[0])
                start_candidates.add(e[1])
            
            if start_strategy == "leaf":
                leaves = [v for v in start_candidates if deg[v] == 1]
                if leaves:
                    if randomize and rng is not None:
                        start_v = rng.choice(leaves)
                    else:
                        start_v = leaves[0]
                else:
                    min_deg = min(deg[v] for v in start_candidates)
                    candidates = [v for v in start_candidates if deg[v] == min_deg]
                    if randomize and rng is not None:
                        start_v = rng.choice(candidates)
                    else:
                        start_v = min(candidates)
            elif start_strategy == "max_deg":
                max_deg = max(deg[v] for v in start_candidates)
                candidates = [v for v in start_candidates if deg[v] == max_deg]
                if randomize and rng is not None:
                    start_v = rng.choice(candidates)
                else:
                    start_v = min(candidates)
            elif start_strategy == "min_deg":
                min_deg = min(deg[v] for v in start_candidates)
                candidates = [v for v in start_candidates if deg[v] == min_deg]
                if randomize and rng is not None:
                    start_v = rng.choice(candidates)
                else:
                    start_v = min(candidates)
            else:
                if randomize and rng is not None:
                    start_v = rng.choice(list(start_candidates))
                else:
                    start_v = min(start_candidates)
            
            path = _greedy_walk(adj, remaining, deg, start_v, rng, randomize, mode)
            if len(path) >= 2:
                paths.append(path)
        
        # Add any remaining edges as single-edge paths
        for e in remaining:
            paths.append(list(e))
            
        return paths

    def _path_cover_with_restarts(edges, deadline):
        """
        Try multiple strategies and randomized restarts to find a good path cover.
        """
        best_paths = None
        best_count = len(edges)
        
        # 1. Deterministic baseline: leaf-start, low-degree greedy
        paths = _path_cover_single(edges, deadline, randomize=False, mode="low_deg", start_strategy="leaf")
        best_paths = paths
        best_count = len(paths)
        
        # 2. Deterministic variant: min-degree start
        if time.time() < deadline - 5:
            paths2 = _path_cover_single(edges, deadline, randomize=False, mode="low_deg", start_strategy="min_deg")
            if len(paths2) < best_count:
                best_count = len(paths2)
                best_paths = paths2
        
        # 3. Deterministic variant: high-degree greedy
        if time.time() < deadline - 5:
            paths3 = _path_cover_single(edges, deadline, randomize=False, mode="high_deg", start_strategy="leaf")
            if len(paths3) < best_count:
                best_count = len(paths3)
                best_paths = paths3
        
        # 4. Randomized restarts
        rng = random.Random(42)
        modes = ["low_deg", "low_deg", "balanced", "high_deg"]
        start_strats = ["leaf", "min_deg", "random"]
        
        trial = 0
        while time.time() < deadline - 2:
            mode = modes[trial % len(modes)]
            strat = start_strats[trial % len(start_strats)]
            
            paths = _path_cover_single(edges, deadline - 2, rng, randomize=True, mode=mode, start_strategy=strat, shuffle_adj=True)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
                if best_count <= 8:
                    break
            
            trial += 1
        
        return best_paths

    def _is_valid_path(path, edges):
        """Check if path is simple and uses only edges from the graph."""
        if len(set(path)) != len(path):
            return False
        for i in range(len(path) - 1):
            e = (min(path[i], path[i+1]), max(path[i], path[i+1]))
            if e not in edges:
                return False
        return True

    def _improve_by_merging(paths, edges, max_time=None):
        """Try to merge paths to reduce count."""
        if max_time is None:
            max_time = deadline - 3
        
        edges = set(edges)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            n = len(paths)
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        break
                    pi = paths[i]
                    pj = paths[j]
                    
                    # Try all 4 combinations of reversing
                    for rev_i in [False, True]:
                        for rev_j in [False, True]:
                            a = list(reversed(pi)) if rev_i else list(pi)
                            b = list(reversed(pj)) if rev_j else list(pj)
                            
                            # Try connecting end of a to start of b
                            if a[-1] == b[0]:
                                merged = a + b[1:]
                                if _is_valid_path(merged, edges):
                                    paths[i] = merged
                                    del paths[j]
                                    improved = True
                                    break
                            # Try connecting end of b to start of a
                            if b[-1] == a[0]:
                                merged = b + a[1:]
                                if _is_valid_path(merged, edges):
                                    paths[i] = merged
                                    del paths[j]
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
        
        return paths

    def _improve_by_splice(paths, edges, max_time=None):
        """
        Try to splice two paths by finding a common internal vertex.
        If path1 = [a, ..., x, ..., b] and path2 = [c, ..., x, ..., d],
        we can potentially form [a, ..., x, ..., d] and [c, ..., x, ..., b]
        or other combinations, but this is complex.
        
        A simpler approach: if two paths share an endpoint, we already handle that in merge.
        For now, let's try a different local improvement:
        For each path, try to extend it by "stealing" edges from other paths.
        """
        if max_time is None:
            max_time = deadline - 3
        
        edges = set(edges)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            n = len(paths)
            for i in range(n):
                if time.time() > max_time:
                    break
                pi = paths[i]
                if len(pi) < 2:
                    continue
                
                # Try to extend pi at either end
                for end in [0, -1]:
                    v = pi[end]
                    # Find edges incident to v that are used by other paths
                    for j in range(n):
                        if j == i or time.time() > max_time:
                            continue
                        pj = paths[j]
                        if len(pj) < 2:
                            continue
                        
                        # Check if v is an endpoint of pj
                        if pj[0] == v:
                            # Try to merge pi (extended at end) with pj (reversed if needed)
                            # This is essentially the same as the merge logic, but let's be explicit
                            if end == -1:
                                merged = pi + pj[1:]
                            else:
                                merged = pj[1:] + pi
                            if _is_valid_path(merged, edges):
                                paths[i] = merged
                                del paths[j]
                                improved = True
                                break
                        elif pj[-1] == v:
                            if end == -1:
                                merged = pi + [x for x in reversed(pj)][1:]
                            else:
                                merged = [x for x in reversed(pj)][1:] + pi
                            if _is_valid_path(merged, edges):
                                paths[i] = merged
                                del paths[j]
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        
        return paths

    # Process each graph
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        edges = instance(*spec)
        
        if time.time() < deadline - 5:
            paths = _path_cover_with_restarts(edges, max_time=deadline - 5)
            if time.time() < deadline - 3:
                paths = _improve_by_merging(paths, edges, max_time=deadline - 3)
        else:
            paths = [list(e) for e in edges]
        
        # Final validation and cleanup
        used_edges = set()
        valid_paths = []
        for path in paths:
            if len(path) < 2:
                continue
            path_edges = set()
            valid = True
            for i in range(len(path) - 1):
                e = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                if e in used_edges or e not in set(edges):
                    valid = False
                    break
                path_edges.add(e)
            if valid:
                valid_paths.append(path)
                used_edges.update(path_edges)
        
        # Add any missing edges as single-edge paths
        missing = set(edges) - used_edges
        for e in missing:
            valid_paths.append(list(e))
        
        results.append(valid_paths)
    
    return results

# EVOLVE-BLOCK-END
