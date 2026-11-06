"""Path decomposition using ILP over candidate simple paths."""
import time
from itertools import combinations, permutations
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def get_adjacency(edges):
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj
    
    def find_all_simple_paths(adj, max_len=None):
        """Find all simple paths up to a certain length. For small graphs, find all."""
        paths = []
        nodes = sorted(adj.keys())
        
        def dfs(path, visited):
            if time.time() - start_time > deadline - 5:
                return
            if len(path) >= 2:
                paths.append(list(path))
            if max_len and len(path) >= max_len:
                return
            last = path[-1]
            for nb in sorted(adj.get(last, [])):
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(path, visited)
                    path.pop()
                    visited.remove(nb)
        
        for start in nodes:
            dfs([start], {start})
        
        return paths
    
    def solve_path_cover(edges, adj, all_paths, time_limit=30.0):
        """Solve minimum path cover using ILP."""
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        num_edges = len(edge_set)
        
        # Filter paths: only keep paths that are valid (all edges exist)
        valid_paths = []
        for path in all_paths:
            valid = True
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    valid = False
                    break
            if valid and len(path) >= 2:
                valid_paths.append(path)
        
        if not valid_paths:
            # Fallback: each edge as its own path
            return [list(e) for e in edges]
        
        num_paths = len(valid_paths)
        
        # Create incidence matrix: path i uses edge j
        # A[i, j] = 1 if path i uses edge j
        A = np.zeros((num_paths, num_edges), dtype=int)
        edge_index = {e: i for i, e in enumerate(edge_set)}
        
        for i, path in enumerate(valid_paths):
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                e = tuple(sorted((u, v)))
                A[i, edge_index[e]] = 1
        
        # Objective: minimize number of paths used
        c = np.ones(num_paths)
        
        # Constraints: each edge covered exactly once
        # A^T * x = 1 for all edges
        lower = np.ones(num_edges)
        upper = np.ones(num_edges)
        constraint = LinearConstraint(A.T, lb=lower, ub=upper)
        
        # Bounds: x_i in [0, 1]
        bounds = Bounds(lb=0, ub=1)
        
        try:
            result = milp(
                c=c,
                constraints=constraint,
                bounds=bounds,
                integrality=np.ones(num_paths)
            )
            
            if result.status == 0 and result.x is not None:
                selected = [i for i in range(num_paths) if result.x[i] > 0.5]
                if len(selected) > 0:
                    # Verify coverage
                    covered_edges = set()
                    for i in selected:
                        path = valid_paths[i]
                        for k in range(len(path) - 1):
                            u, v = path[k], path[k + 1]
                            covered_edges.add(tuple(sorted((u, v))))
                    
                    if covered_edges == edge_set:
                        return [valid_paths[i] for i in selected]
                else:
                    return [list(e) for e in edges]
            else:
                return [list(e) for e in edges]
        except Exception:
            return [list(e) for e in edges]
    
    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for n, p, seed in specs:
        if time.time() > deadline:
            # Emergency fallback
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        # Find candidate paths - limit to reasonable length to keep ILP tractable
        # For n=30, full enumeration may be too large. Use max_len heuristic.
        max_len = min(n, 12)  # Limit path length to keep number of paths manageable
        all_paths = find_all_simple_paths(adj, max_len=max_len)
        
        # If too many paths, use greedy fallback
        if len(all_paths) > 5000:
            # Greedy: repeatedly pick longest available path
            remaining_edges = set(tuple(sorted(e)) for e in edges)
            paths_found = []
            
            while remaining_edges:
                if time.time() > deadline - 5:
                    break
                best_path = None
                best_len = 0
                for path in all_paths:
                    # Check if path can be formed from remaining edges
                    valid = True
                    for k in range(len(path) - 1):
                        u, v = path[k], path[k + 1]
                        e = tuple(sorted((u, v)))
                        if e not in remaining_edges:
                            valid = False
                            break
                    if valid and len(path) > best_len:
                        best_path = path
                        best_len = len(path)
                
                if best_path is None:
                    # Pick any remaining edge
                    e = next(iter(remaining_edges))
                    best_path = list(e)
                
                paths_found.append(best_path)
                for k in range(len(best_path) - 1):
                    u, v = best_path[k], best_path[k + 1]
                    remaining_edges.discard(tuple(sorted((u, v))))
            
            # Add any remaining edges as single-edge paths
            for e in remaining_edges:
                paths_found.append(list(e))
            
            results.append(paths_found)
        else:
            # Use ILP
            time_left = deadline - time.time() - 5
            if time_left > 5:
                paths_found = solve_path_cover(edges, adj, all_paths, time_limit=time_left)
                results.append(paths_found)
            else:
                results.append([list(e) for e in edges])
    
    return results
