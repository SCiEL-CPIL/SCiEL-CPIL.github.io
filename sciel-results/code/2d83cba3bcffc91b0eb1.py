import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_greedy_path(adj, available_edges, rng):
        """
        Find a simple path using a randomized greedy walk.
        This is O(V) per path and avoids exponential backtracking.
        """
        if not available_edges:
            return []
        
        # Pick a random start vertex from those with available edges
        # To improve path length, pick a vertex with high degree in available edges
        degrees = defaultdict(int)
        for u, v in available_edges:
            degrees[u] += 1
            degrees[v] += 1
        
        if not degrees:
            return []
            
        # Create a list of candidates, weighted by degree
        # Simple approach: pick a random vertex from the set of vertices with deg > 0
        # To bias towards longer paths, we can pick a random edge and start from one end
        # Let's pick a random edge (u, v) and start from u, forcing the first step to v?
        # Or just pick a random start node with high degree.
        
        # Let's try picking a random start node from the available vertices
        start_candidates = list(degrees.keys())
        if not start_candidates:
            return []
        
        # Shuffle to break symmetry
        rng.shuffle(start_candidates)
        
        # Try a few starts to find a decent path
        best_path = []
        
        # We only try 1 start to keep it O(V) per call, or maybe 2-3
        for start in start_candidates[:2]:
            path = [start]
            visited = {start}
            current = start
            
            while True:
                # Find available neighbors
                neighbors = []
                for v in adj[current]:
                    e = tuple(sorted((current, v)))
                    if e in available_edges and v not in visited:
                        neighbors.append(v)
                
                if not neighbors:
                    break
                
                # Pick a random neighbor
                next_node = rng.choice(neighbors)
                visited.add(next_node)
                path.append(next_node)
                current = next_node
            
            if len(path) > len(best_path):
                best_path = path
        
        return best_path

    def merge_paths(paths, edge_set):
        """
        Try to merge paths to reduce the count.
        """
        if len(paths) <= 1:
            return paths
        
        improved = True
        while improved:
            improved = False
            new_paths = []
            merged_indices = set()
            
            for i in range(len(paths)):
                if i in merged_indices:
                    continue
                path_i = paths[i]
                merged = False
                
                for j in range(i + 1, len(paths)):
                    if j in merged_indices:
                        continue
                    path_j = paths[j]
                    
                    # Try merging: path_i + path_j
                    if path_i[-1] == path_j[0]:
                        candidate = path_i + path_j[1:]
                        if len(set(candidate)) == len(candidate):
                            # Check edges
                            valid = True
                            for k in range(len(candidate) - 1):
                                e = tuple(sorted((candidate[k], candidate[k+1])))
                                if e not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_paths.append(candidate)
                                merged_indices.add(j)
                                improved = True
                                merged = True
                                break
                    
                    # Try merging: path_j + path_i
                    elif path_j[-1] == path_i[0]:
                        candidate = path_j + path_i[1:]
                        if len(set(candidate)) == len(candidate):
                            valid = True
                            for k in range(len(candidate) - 1):
                                e = tuple(sorted((candidate[k], candidate[k+1])))
                                if e not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_paths.append(candidate)
                                merged_indices.add(i)
                                improved = True
                                merged = True
                                break
                    
                    # Try reversing path_j
                    path_j_rev = path_j[::-1]
                    if path_i[-1] == path_j_rev[0]:
                        candidate = path_i + path_j_rev[1:]
                        if len(set(candidate)) == len(candidate):
                            valid = True
                            for k in range(len(candidate) - 1):
                                e = tuple(sorted((candidate[k], candidate[k+1])))
                                if e not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_paths.append(candidate)
                                merged_indices.add(j)
                                improved = True
                                merged = True
                                break
                                
                    if path_j_rev[-1] == path_i[0]:
                        candidate = path_j_rev + path_i[1:]
                        if len(set(candidate)) == len(candidate):
                            valid = True
                            for k in range(len(candidate) - 1):
                                e = tuple(sorted((candidate[k], candidate[k+1])))
                                if e not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_paths.append(candidate)
                                merged_indices.add(i)
                                improved = True
                                merged = True
                                break
                            
                if not merged:
                    new_paths.append(path_i)
            
            paths = new_paths
        
        return paths

    def solve_graph(edges, seed):
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        rng = random.Random(seed + 100)
        
        paths = []
        available = set(edge_set)
        
        # Greedy path decomposition using fast greedy walks
        while available:
            if time.time() > deadline:
                break
            
            # Find a path
            path = find_greedy_path(adj, available, rng)
            
            if not path:
                # Fallback: pick any edge
                u, v = next(iter(available))
                path = [u, v]
            
            # Remove edges from available
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e in available:
                    available.remove(e)
            
            paths.append(path)
        
        # If we ran out of time and still have edges, add them as single-edge paths
        if available:
            for u, v in available:
                paths.append([u, v])
        
        # Simple merge to reduce path count further if time permits
        # The original merge was complex, let's do a simple one or skip if too slow
        # Given the speed of find_greedy_path, we have plenty of time for a simple merge
        if time.time() < deadline - 10:
            paths = merge_paths(paths, edge_set)
            
        return paths

    results = []
    for i, spec in enumerate([(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]):
        if time.time() > deadline:
            # Fallback
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        paths = solve_graph(edges, spec[2])
        results.append(paths)
        
    return results
