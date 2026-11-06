import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def lower_bound(edges):
        if not edges:
            return 0
        degree = defaultdict(int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        odd_count = sum(1 for d in degree.values() if d % 2 == 1)
        return max(1, (odd_count + 1) // 2)

    def greedy_extend(start, adj, rng, deadline):
        """
        Greedy path extension with Warnsdorff-like heuristic and randomization.
        Returns the path found.
        """
        path = [start]
        visited = {start}
        
        for _ in range(100):  # max path length cap
            if time.time() > deadline:
                break
            current = path[-1]
            neighbors = [nb for nb in adj[current] if nb not in visited]
            
            if not neighbors:
                break
            
            # Score neighbors: prefer those with fewer unvisited neighbors (Warnsdorff)
            def score(nb):
                unvisited_count = sum(1 for x in adj[nb] if x not in visited)
                return unvisited_count
            
            if rng.random() < 0.85:
                neighbors.sort(key=score)
                # Pick from top 3 with slight randomness
                top_k = min(3, len(neighbors))
                choice = rng.choice(neighbors[:top_k])
            else:
                choice = rng.choice(neighbors)
            
            path.append(choice)
            visited.add(choice)
        
        return path

    def decompose_with_restarts(edges, deadline):
        """
        Decompose edges into minimum number of edge-disjoint simple paths.
        Uses randomized restarts with greedy path finding.
        """
        if not edges:
            return []
        
        best_paths = None
        best_count = len(edges)
        
        # Determine number of restarts based on time available
        # We want to finish well before deadline
        time_per_restart = 0.05  # estimate
        num_restarts = max(10, int((deadline - time.time()) * 10))
        num_restarts = min(num_restarts, 500)
        
        lb = lower_bound(edges)
        
        for attempt in range(num_restarts):
            if time.time() > deadline - 2:
                break
            
            rng = random.Random(attempt * 7919 + 13)
            paths = _greedy_decompose(edges, rng, deadline)
            
            if paths is None:
                continue
            
            # Post-process: merge paths if possible
            paths = merge_paths(paths, edges)
            
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                if count <= lb:
                    break
        
        if best_paths is None:
            return [list(e) for e in edges]
        
        return best_paths

    def _greedy_decompose(edges, rng, deadline):
        """
        Single pass greedy decomposition.
        """
        remaining = set(tuple(sorted(e)) for e in edges)
        if not remaining:
            return []
        
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        degree = defaultdict(int)
        for u, v in remaining:
            degree[u] += 1
            degree[v] += 1
        
        paths = []
        
        while remaining:
            if time.time() > deadline:
                return None
            
            # Pick start vertex
            # Prefer odd-degree vertices
            odd_verts = [v for v in adj if degree.get(v, 0) % 2 == 1 and adj[v]]
            if odd_verts:
                start = rng.choice(odd_verts)
            else:
                candidates = [v for v in adj if adj[v]]
                if not candidates:
                    break
                # Weight by degree
                weights = [max(1, degree.get(v, 0)) for v in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                start = candidates[-1]
                for v, w in zip(candidates, weights):
                    cum += w
                    if r <= cum:
                        start = v
                        break
            
            # Find long path
            path = greedy_extend(start, adj, rng, deadline)
            
            if len(path) < 2:
                # Fallback: take a single edge
                if adj[start]:
                    nb = rng.choice(list(adj[start]))
                    path = [start, nb]
                else:
                    e = next(iter(remaining))
                    path = [e[0], e[1]]
            
            # Remove edges from path
            path_edges = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e = tuple(sorted((u, v)))
                if e in remaining:
                    path_edges.add(e)
            
            if not path_edges:
                # Should not happen, but safety
                e = next(iter(remaining))
                path = [e[0], e[1]]
                path_edges = {e}
            
            for e in path_edges:
                remaining.discard(e)
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
                degree[u] -= 1
                degree[v] -= 1
            
            # Clean up adj
            for v in list(adj.keys()):
                if not adj[v]:
                    del adj[v]
            
            paths.append(path)
        
        return paths

    def merge_paths(paths, edges):
        """
        Try to merge paths that share endpoints.
        This is a simple post-processing step.
        """
        # Build a map from vertex to list of paths containing it (and position)
        # For simplicity, we just try to concatenate paths if they share an endpoint
        # and the resulting path is still simple and uses valid edges.
        
        # This is a heuristic merge. We try to find pairs (i, j) where
        # paths[i] ends at v and paths[j] starts at v (or vice versa)
        # and merge them.
        
        # To keep it simple and fast, we do a few passes of merging.
        merged = True
        current_paths = [list(p) for p in paths]
        
        # Build edge set for quick lookup
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        while merged:
            merged = False
            new_paths = []
            used = [False] * len(current_paths)
            
            for i in range(len(current_paths)):
                if used[i]:
                    continue
                p1 = current_paths[i]
                
                # Try to merge with any other path
                best_j = -1
                best_dir = 0 # 0: p1 + p2, 1: p2 + p1
                
                for j in range(len(current_paths)):
                    if i == j or used[j]:
                        continue
                    p2 = current_paths[j]
                    
                    # Check if p1 ends at p2 start
                    if p1[-1] == p2[0]:
                        # Check if merge is valid (simple path, valid edges)
                        if _is_valid_merge(p1, p2, edge_set):
                            best_j = j
                            best_dir = 0
                            break
                    
                    # Check if p2 ends at p1 start
                    if p2[-1] == p1[0]:
                        if _is_valid_merge(p2, p1, edge_set):
                            best_j = j
                            best_dir = 1
                            break
                
                if best_j != -1:
                    if best_dir == 0:
                        new_path = p1 + p2[1:]
                    else:
                        new_path = current_paths[best_j] + p1[1:]
                    
                    new_paths.append(new_path)
                    used[best_j] = True
                    merged = True
                else:
                    new_paths.append(p1)
            
            current_paths = new_paths
        
        return current_paths

    def _is_valid_merge(p1, p2, edge_set):
        """
        Check if p1 + p2[1:] is a valid simple path using edges from edge_set.
        """
        merged = p1 + p2[1:]
        
        # Check simplicity
        if len(merged) != len(set(merged)):
            return False
        
        # Check edges
        for i in range(len(merged) - 1):
            e = tuple(sorted((merged[i], merged[i+1])))
            if e not in edge_set:
                return False
        
        return True

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        paths = decompose_with_restarts(edges, deadline)
        results.append(paths)
    
    # Fallback for any missing graphs
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
